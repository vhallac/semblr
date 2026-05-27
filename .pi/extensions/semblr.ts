/**
 * semblr — Retrieval-Augmented Context Assembly
 *
 * At agent_end: save the completed round to .pi/rounds/ and embed it.
 * At context: embed the current user prompt, query the vector index,
 *             inject the top-matching rounds as context for the LLM.
 *
 * Replaces flashback-amnesia.ts — no wiping, just smart retrieval.
 */

import type { ExtensionAPI, ContextEvent } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import * as fs from "node:fs";
import * as crypto from "node:crypto";
import * as os from "node:os";
import { spawnSync } from "node:child_process";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

// PI_CODING_AGENT_DIR overrides the default ~/.pi/agent config directory.
// We store semblr rounds under that directory so they survive project moves.
const PI_CONFIG_DIR = process.env.PI_CODING_AGENT_DIR || `${os.homedir()}/.pi/agent`;
const ROUNDS_DIR = `${PI_CONFIG_DIR}/semblr/rounds`;
const INDEX_PATH = `${ROUNDS_DIR}/index.csv`;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";

const CONTEXT_BUDGET_RATIO = 0.5; // 50% of model context window for historical rounds

// Collapse mode: "full" injects complete historical rounds into context.
// "collapsed" injects only a compact index — the LLM must use
// get_round_details() / get_tool_details() to expand rounds it needs.
const ROUND_COLLAPSE_MODE: "full" | "collapsed" = "collapsed";

// ─────────────────────────────────────────────
// Causal Chain — in-memory buffer of session rounds
// ─────────────────────────────────────────────

interface ChainEntry {
  fileName: string;
  userPrompt: string;
  responseSequence: string;
  toolSummary: string;
}

/** In-memory buffer of rounds from the current session, in chronological order.
 *  Survives between agent cycles within the same session. Cleared on session_start.
 *  Injected into context so the model can resolve "it", "those changes", "the fix", etc.
 *  without needing to search the vector index. */
let causalChain: ChainEntry[] = [];

/** Format the causal chain buffer as a collapsed-style index block with n/a scores. */
function formatCausalChainBlock(chain: ChainEntry[]): string | null {
  if (chain.length === 0) return null;
  const lines: string[] = [];
  lines.push(`[CAUSAL CHAIN — recent rounds in this session, newest first]
The following list shows the most recent rounds in this session. When the current
prompt contains references to recent events ("it", "those changes", "the fix", etc.),
review this chain to discover the referent. Entries with n/a score are in-memory
only — not yet embedded in the vector index.`);
  lines.push("---");
  // Show newest first (reverse chronological)
  const reversed = [...chain].reverse();
  let idx = 0;
  for (const entry of reversed) {
    idx++;
    const promptLines = entry.userPrompt
      .split("\n")
      .map((line, i) => i === 0 ? `  user: ${line}` : `  ${line}`);
    lines.push(
      `${idx}. ${entry.fileName} [n/a | ${entry.toolSummary}]:`
    );
    lines.push(...promptLines);
    lines.push("  ---");
  }
  return lines.join("\n");
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function normalize(v: number[]): number[] {
  const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  return mag === 0 ? v : v.map((x) => x / mag);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function extractText(content: Array<{ type: string; text?: string }>): string {
  return content
    .filter((c) => c.type === "text" && c.text)
    .map((c) => c.text!)
    .join(" ");
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  return d.toTimeString().slice(0, 8); // HH:MM:SS
}

/**
 * Format a compact index of retrieved rounds for collapsed mode.
 * Numbered list style — preserves full prompt text and paragraph
 * structure so the LLM can weigh the full signal when deciding
 * which rounds to expand via get_round_details().
 *
 * Format:
 *   1. <hash> [0.78 | 8 tools]:
 *      user: this is line 1
 *
 * this is line 2
 *
 * this is line 3
 *      ---
 *   2. <hash2> ...
 */
function formatCollapsedIndex(
  rounds: Array<{ fileName: string; bestScore: number; data: RoundData }>
): string {
  const lines: string[] = [];
  let idx = 0;
  for (const round of rounds) {
    idx++;
    const toolCount = round.data.toolCallCount ?? 0;
    // Indent each line of the user prompt by 6 spaces so paragraphs
    // are visually distinct from the header line.
    const promptLines = round.data.userPrompt
      .split("\n")
      .map((line, i) => i === 0 ? `  user: ${line}` : `  ${line}`);
    lines.push(
      `${idx}. ${round.fileName} [${round.bestScore.toFixed(2)} | ${toolCount} tools]:`
    );
    lines.push(...promptLines);
    lines.push("  ---");
  }
  return lines.join("\n");
}

function formatDelta(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${Math.round(ms / 1000)}s`;
  if (ms < 3600_000) return `${Math.floor(ms / 60_000)}m${Math.round((ms % 60_000) / 1000)}s`;
  return `${Math.floor(ms / 3600_000)}h${Math.floor((ms % 3600_000) / 60_000)}m`;
}

function buildTurnTimeline(currentTurnIndex: number | null): string | null {
  if (agentTurnTimestamps.size === 0) return null;

  // Sort completed turns (those with index < current) chronologically
  const completed = Array.from(agentTurnTimestamps.entries())
    .filter(([ti]) => currentTurnIndex == null || ti < currentTurnIndex)
    .sort(([a], [b]) => a - b);

  if (completed.length === 0) return null;

  const lines: string[] = [];
  let prev: number | null = null;
  for (const [ti, ts] of completed) {
    const delta = prev != null ? ` (+${formatDelta(ts - prev)})` : "";
    lines.push(`Turn ${ti + 1}: ${formatTimestamp(ts)}${delta}`);
    prev = ts;
  }

  // Current turn marker
  if (currentTurnIndex != null && agentTurnTimestamps.has(currentTurnIndex)) {
    const currentTs = agentTurnTimestamps.get(currentTurnIndex)!;
    lines.push(`[Current: turn ${currentTurnIndex + 1} — started ${formatTimestamp(currentTs)}]`);
  }

  return `Turn timeline:\n${lines.join("\n")}`;
}

// ─────────────────────────────────────────────
// Index CSV format:
//   base64url(vector_json),filePath
//   (no header row)
//   filePath includes :prompt or :response suffix
// ─────────────────────────────────────────────

interface IndexEntry {
  filePath: string;
  vector: number[];
}

function loadIndex(): IndexEntry[] {
  if (!fs.existsSync(INDEX_PATH)) return [];
  const raw = fs.readFileSync(INDEX_PATH, "utf-8").trim();
  if (!raw) return [];
  return raw.split("\n").map((line) => {
    const comma = line.indexOf(",");
    const b64 = line.slice(0, comma);
    const filePath = line.slice(comma + 1);
    const decoded = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
    return { filePath, vector: Array.isArray(decoded) ? decoded : [] };
  });
}

interface ToolCallDetail {
  index: number;
  name: string;
  arguments: string;    // JSON string of arguments (abbreviated if >500 chars)
  result_summary: string; // first 300 chars of result text
}

interface RoundData {
  userPrompt: string;
  responseSequence: string;
  turnIndex: number;
  userTimestamp?: number;
  toolCallCount?: number;
  toolCallNames?: string[];
  toolCalls?: ToolCallDetail[];
}

function readRoundFile(
  filePath: string,
): RoundData | null {
  // filePath may be "xxx.json:prompt" or "xxx.json:response"
  // strip the :prompt/:response suffix to get the actual file
  const actualFile = filePath.replace(/:prompt$|:response$/, "");
  const fullPath = `${ROUNDS_DIR}/${actualFile}`;
  if (!fs.existsSync(fullPath)) return null;
  try {
    const data = JSON.parse(fs.readFileSync(fullPath, "utf-8"));
    return {
      userPrompt: data.userPrompt ?? "",
      responseSequence: data.responseSequence ?? "",
      turnIndex: data.turnIndex ?? 0,
      userTimestamp: data.userTimestamp,
      toolCallCount: data.toolCallCount,
      toolCallNames: data.toolCallNames,
      toolCalls: data.toolCalls,
    };
  } catch {
    return null;
  }
}

let lastRoundFileName: string | null = null; // tracks the most recent saved round (process-local)

// Per-agent accumulation (reset in agent_start, saved in agent_end)
let agentUserPrompt: string | null = null;
let agentTurnIndex: number | null = null;
let agentAccumulatedText: string[] = []
let agentToolCallCount: number = 0;
let agentToolCallNames: string[] = [];
let agentToolCalls: ToolCallDetail[] = [];
let agentPendingToolCallIds: Map<string, ToolCallDetail> = new Map(); // toolCallId → partial detail


// Turn timeline tracking — maps turnIndex → start timestamp (ms)
// Populated in agent_start, filtered in context to show only completed turns
let agentTurnTimestamps = new Map<number, number>();

// Context embedding cache — avoids redundant embedding API calls across tool turns
// within the same agent cycle. Reset in agent_start.
let lastContextUserPrompt: string | null = null;
let lastContextVec: number[] = [];
let agentPromptVec: number[] | null = null; // cached from context hook, reused in agent_end to avoid redundant embed

async function getApiKey(ctx?: { modelRegistry?: { getApiKeyForProvider(provider: string): Promise<string | undefined> } }): Promise<string | null> {
  // 1. Environment variable
  const envKey = process.env.OPENROUTER_API_KEY;
  if (envKey) return envKey;

  // 2. Pi's configured OpenRouter auth (e.g. /login, auth storage, models.json)
  try {
    const piKey = await ctx?.modelRegistry?.getApiKeyForProvider("openrouter");
    if (piKey) return piKey;
  } catch {
    // Pi auth lookup failed, fall through
  }

  // 3. Pass store
  try {
    const result = spawnSync("pass", ["show", "ai/openrouter"], {
      timeout: 5000,
      stdio: ["pipe", "pipe", "pipe"],
    });
    if (result.status === 0) {
      const key = result.stdout.toString().trim();
      if (key) return key;
    }
  } catch {
    // pass not available, fall through
  }

  return null;
}

async function embedText(text: string, apiKey: string): Promise<number[]> {
  const response = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: text,
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Embedding API error ${response.status}: ${errText}`);
  }

  const data = (await response.json()) as {
    data: Array<{ embedding: number[] }>;
  };
  return data.data[0].embedding;
}

function createRoundFilePath(userPrompt: string, responseText: string): string {
  const content = userPrompt + responseText;
  const hash = crypto.createHash("md5").update(content).digest("hex");
  return `${hash}.json`;
}

function appendToIndex(filePath: string, vector: number[]) {
  const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
  fs.mkdirSync(ROUNDS_DIR, { recursive: true });
  fs.appendFileSync(INDEX_PATH, `${b64},${filePath}\n`);
}

// ─────────────────────────────────────────────
// Extension
// ─────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // ────────────────────────────────────────────
  // 1. context — assemble context from round repository
  // ────────────────────────────────────────────
  pi.on("context", async (event: ContextEvent, ctx) => {
    const { messages } = event;

    // --- Extract system prompt + current round messages ---
    // We strip all prior rounds to prevent conversation bloat.
    // The retrieved historical context replaces the prior conversation.
    // Current round = everything from the last user message onward
    // (includes assistant responses, tool calls, tool results in-flight).
    const systemMsg = messages.find(
      (m) => m.role === "system" || m.role === "developer",
    ) ?? null;
    const lastUserIdx = messages.reduce((last, m, i) =>
      m.role === "user" ? i : last, -1);
    const currentMessages = lastUserIdx >= 0
      ? [...messages].slice(lastUserIdx)
      : [...messages];

    // --- Get the current user prompt (last user message) ---
    const userMessages = currentMessages.filter((m) => m.role === "user");
    if (userMessages.length === 0) return { messages };

    // Extract user prompt text — content may be a string or an array of content blocks
    const lastUserContent = userMessages[userMessages.length - 1].content;
    let userPrompt: string;
    if (typeof lastUserContent === "string") {
      userPrompt = lastUserContent.split(" ").slice(0, 200).join(" ");
    } else if (Array.isArray(lastUserContent)) {
      userPrompt = extractText(lastUserContent);
    } else {
      return { messages };
    }

    // agentPromptVec is stashed after embedding below for agent_end to reuse


    try {
      const apiKey = await getApiKey(ctx);
      if (!apiKey) return { messages };

      // Embed the user prompt — cached per agent cycle to avoid redundant API calls
      // across multiple tool turns within the same user prompt.
      let queryVec: number[];
      if (userPrompt === lastContextUserPrompt) {
        queryVec = lastContextVec;
      } else {
        queryVec = normalize(await embedText(userPrompt, apiKey));
        lastContextUserPrompt = userPrompt;
        lastContextVec = queryVec;
      }
      // Stash for agent_end to reuse (saves 1 embedding call per round)
      agentPromptVec = queryVec;


      // Load and score the index
      const index = loadIndex();
      if (index.length === 0) return { messages };

      const scored = index
        .map((entry) => ({
          ...entry,
          similarity: cosineSimilarity(queryVec, entry.vector),
        }))
        .sort((a, b) => b.similarity - a.similarity);
      const bestScore = scored.length > 0 ? scored[0].similarity : 0;

      // --- Dynamic budget ---
      const MIN_SIMILARITY = 0.30;
      const minBudget = 2000;
      const MAX_BUDGET = Math.floor(
        CONTEXT_BUDGET_RATIO *
          (event.contextWindowSize ?? 128_000),
      );
      // Linear interpolation: at MIN_SIMILARITY → minBudget, at 1.0 → MAX_BUDGET
      const t = Math.max(
        0,
        Math.min(1, (bestScore - MIN_SIMILARITY) / (1 - MIN_SIMILARITY)),
      );
      const budgetTokens = Math.floor(
        minBudget + t * (MAX_BUDGET - minBudget),
      );

      // Group by round file, take best score per round
      interface RoundScore {
        data: RoundData;
        fileName: string;
        bestScore: number;
      }
      const roundScores = new Map<string, RoundScore>();
      for (const entry of scored) {
        const roundFile = entry.filePath.replace(/:prompt$|:response$/, "");
        if (!roundFile.endsWith(".json")) continue;
        if (roundScores.has(roundFile)) continue;
        const roundData = readRoundFile(entry.filePath);
        if (!roundData) continue;
        roundScores.set(roundFile, {
          data: roundData,
          fileName: roundFile,
          bestScore: entry.similarity,
        });
      }

      const uniqueRounds = new Set(
        index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
      ).size;

      const scoredRounds = Array.from(roundScores.values()).sort(
        (a, b) => b.bestScore - a.bestScore,
      );

      // Select rounds within budget
      const selectedRounds: RoundScore[] = [];
      let usedTokens = 0;
      const addRound = (round: RoundScore) => {
        selectedRounds.push(round);
      };

      // 1. Score-based selection (below threshold stops)
      for (const round of scoredRounds) {
        if (round.bestScore < MIN_SIMILARITY) break;
        const roundTokens = estimateTokens(
          round.data.userPrompt + round.data.responseSequence,
        );
        if (usedTokens + roundTokens > budgetTokens) break;
        addRound(round);
        usedTokens += roundTokens;
      }

      // 2. Always add the last round (if not already there)
      // Content hash used later to move it to last position after dedup.
      let lastRoundContentHash: string | null = null;
      if (lastRoundFileName) {
        const lastData = readRoundFile(lastRoundFileName);
        if (lastData) {
          addRound({ data: lastData, fileName: lastRoundFileName, bestScore: 0 });
          lastRoundContentHash = crypto.createHash("md5")
            .update(lastData.userPrompt + lastData.responseSequence)
            .digest("hex");
        }
      }

      if (selectedRounds.length === 0) {
        ctx.ui.setStatus(
          "semblr",
          `🧠 no relevant context (best: ${bestScore.toFixed(3)})`,
        );
        return { messages };
      }

      // ── Branched context injection: full vs collapsed ──
      if (ROUND_COLLAPSE_MODE === "collapsed") {
        // Collapsed mode: inject only a compact index. The LLM expands
        // rounds it needs via get_round_details() / get_tool_details().
        const collapsedIndex = formatCollapsedIndex(
          selectedRounds.map(r => ({ fileName: r.fileName, bestScore: r.bestScore, data: r.data }))
        );

        const uniqueRounds = new Set(
          index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
        ).size;

        ctx.ui.setStatus(
          "semblr",
          `🧠 collapsed: ${selectedRounds.length} rounds indexed from ${uniqueRounds} total`,
        );

        const timeline = buildTurnTimeline(agentTurnIndex);

        const enrichedSystem = systemMsg
          ? {
              ...systemMsg,
              content: typeof systemMsg.content === "string"
                ? systemMsg.content + `

[ENVIRONMENT]
Host: ${os.hostname()}
CWD: ${process.cwd()}
Current date/time: ${new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+/, "")}`
                : systemMsg.content,
            }
          : null;

        // Find the last round (by fileName) — always added earlier.
        // Inject the user prompt + collapsed agent turns (tool calls redacted,
        // expandable via get_tool_details) + assistant text response.
        // This replaces the old approach of dumping raw responseSequence, which
        // mixed text and tool calls together. Now the model sees structured turns.
        let lastRoundFull: Array<{ role: string; content: Array<{ type: string; text: string }> }> = [];
        if (lastRoundFileName) {
          const last = selectedRounds.find(r => r.fileName === lastRoundFileName);
          if (last) {
            // Build collapsed tool call turns
            let toolTurnText = "";
            if (last.data.toolCallCount != null && last.data.toolCallCount > 0 && last.data.toolCalls && last.data.toolCalls.length > 0) {
              const turnLines = last.data.toolCalls.map((tc: ToolCallDetail) =>
                `  Turn ${tc.index}: ${tc.name} — [REDACTED: arguments and result collapsed. Use get_tool_details("${last!.fileName}", ${tc.index}) to expand.]`
              );
              toolTurnText = "\n--- Agent turns (all tool calls redacted — use get_tool_details to expand) ---\n" + turnLines.join("\n");
            } else if (last.data.toolCallCount === 0) {
              toolTurnText = "\n--- Agent turns ---\n  (no tool calls — discussion only)";
            }

            lastRoundFull = [
              {
                role: "user",
                content: [{ type: "text", text: `[LAST ROUND — ${lastRoundFileName} (collapsed)]
User asked: ${last.data.userPrompt}${toolTurnText}` }],
              },
              {
                role: "assistant",
                content: [{ type: "text", text: `${last.data.responseSequence}` }],
              },
            ];
          }
        }

        // Build causal chain system message from in-memory buffer
        const chainBlock = formatCausalChainBlock(causalChain);

        const finalMessages = [
          ...(enrichedSystem ? [enrichedSystem] : []),
          // Preamble for collapsed mode — use "user" role because pi's convertToLlm
          // strips messages with role "system". The system prompt is handled entirely
          // via the separate systemPrompt field. Using "user" ensures the model sees
          // this historical context.
          {
            role: "user",
            content: [{
              type: "text",
              text: `[HISTORICAL ROUND INDEX — use get_round_details("hash.json") to expand any round]
Numbered list. Each entry: N. hash.json [score | N tools]: followed by full user prompt (indented).
Use get_tool_details("hash.json", N) to inspect tool call N within a round.
${timeline ? timeline + "\n" : ""}---`,
            }],
          },
          // Causal chain — in-memory rounds from this session
          ...(chainBlock
            ? [{
                role: "user" as const,
                content: [{ type: "text" as const, text: chainBlock }],
              }]
            : []),
          // Compact index (also as "user" role)
          {
            role: "user",
            content: [{
              type: "text",
              text: collapsedIndex,
            }],
          },
          // Last round in full (sans tool calls) — for prompt parsing
          ...lastRoundFull,
          ...currentMessages,
        ];

        return { messages: finalMessages };
      }

      // ── Full mode (default): inject complete round content ──
      // Build context messages — chronological order (oldest first) for coherence
      // Chronological: prefer turnIndex within session, then userTimestamp as tiebreaker
      selectedRounds.sort((a, b) => {
        const ti = a.data.turnIndex - b.data.turnIndex;
        if (ti !== 0) return ti;
        const tsA = (a.data as Record<string, unknown>).userTimestamp as number ?? 0;
        const tsB = (b.data as Record<string, unknown>).userTimestamp as number ?? 0;
        return tsA - tsB;
      });

      // Dedup by MD5 content hash — last round may duplicate a scored round
      const seenHashes = new Set<string>();
      const dedupedRounds: typeof selectedRounds = [];
      for (const round of selectedRounds) {
        const hash = crypto.createHash("md5").update(round.data.userPrompt + round.data.responseSequence).digest("hex");
        if (seenHashes.has(hash)) continue;
        seenHashes.add(hash);
        dedupedRounds.push(round);
      }

      // Post-sort: ensure last round is always the LAST element in the injected context.
      // This bridges the referential gap — the model sees the preceding round's response
      // immediately before the current prompt, making "these", "it", "that" resolvable.
      if (lastRoundContentHash) {
        const lastIdx = dedupedRounds.findIndex(r => {
          const h = crypto.createHash("md5").update(r.data.userPrompt + r.data.responseSequence).digest("hex");
          return h === lastRoundContentHash;
        });
        if (lastIdx !== -1 && lastIdx !== dedupedRounds.length - 1) {
          const [lastRound] = dedupedRounds.splice(lastIdx, 1);
          dedupedRounds.push(lastRound);
        }
      }

      // Rebuild contextMessages from deduped rounds
      // Each round is wrapped in a clear delimiter so the model knows these
      // are historical records — not the current conversation. This prevents
      // the model from mimicking tool calls or phrasing from past rounds.
      const contextMessages: Array<{ role: string; content: Array<{ type: string; text: string }> }> = [];

      // ── Preamble: explain the historical round format ──
      // Injected before the first round so the model understands what it's seeing.
      contextMessages.push({
        role: "system",
        content: [{
          type: "text",
          text: `[HISTORICAL ROUND INDEX — relevance score (0.0–1.0) | tool count]
Use get_round_details("hash.json") to expand a round's full conversation.
Use get_tool_details("hash.json", N) to inspect tool call N within a round.
The responses shown below are historical context, not your own voice.
---`,
        }],
      });

      // Inject turn timeline — gives the agent a sense
      // of pacing and when completed turns happened (1-based display).
      const timeline = buildTurnTimeline(agentTurnIndex);
      if (timeline) {
        contextMessages.push({
          role: "system",
          content: [{ type: "text", text: timeline }],
        });
      }

      for (const round of dedupedRounds) {
        // Build tool metadata line to distinguish real work from discussion
        let toolMeta = "";
        if (round.data.toolCallCount != null && round.data.toolCallCount > 0) {
          const names = round.data.toolCallNames?.length ? round.data.toolCallNames.join(", ") : "unknown";
          toolMeta = `${round.data.toolCallCount} tools`;
          // Add tool detail hints if available
          if (round.data.toolCalls && round.data.toolCalls.length > 0) {
            const hints = round.data.toolCalls.slice(0, 5).map(tc =>
              `${tc.index}:${tc.name}`
            ).join(" ");
            const more = round.data.toolCalls.length > 5 ? ` ...+${round.data.toolCalls.length - 5} more` : "";
            toolMeta += `\nUse get_tool_details("${round.fileName}", N) to inspect any call. Indexes: ${hints}${more}`;
          }
        } else if (round.data.toolCallCount === 0) {
          toolMeta = `0 tools`;
        }

        // Hash-first format: the round filename is the critical reference
        // for get_tool_details() — put it first so it's easy to copy-paste.
        contextMessages.push({
          role: "user",
          content: [{ type: "text", text: `[HISTORICAL ROUND — ${round.fileName} | similarity: ${round.bestScore.toFixed(3)} [TOOLS USED: ${toolMeta}]
User asked: ${round.data.userPrompt}` }],
        });
        contextMessages.push({
          role: "assistant",
          content: [{ type: "text", text: `[END OF HISTORICAL ROUND — this was a past response, not the current one. Stay focused on the user's most recent message above.]
${round.data.responseSequence}` }],
        });
      }

      ctx.ui.setStatus(
        "semblr",
        `🧠 retrieved ${dedupedRounds.length} rounds (${usedTokens} tok) from ${uniqueRounds} indexed`,
      );

      const enrichedSystem = systemMsg
        ? {
            ...systemMsg,
            content: typeof systemMsg.content === "string"
              ? systemMsg.content + `

[ENVIRONMENT]
Host: ${os.hostname()}
CWD: ${process.cwd()}
Current date/time: ${new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+/, "")}`
              : systemMsg.content,
          }
        : null;

      return {
        messages: [
          ...(enrichedSystem ? [enrichedSystem] : []),
          ...contextMessages,
          ...currentMessages,
        ],
      };
    } catch (err) {
      ctx.ui.setStatus("semblr", `🧠 error: ${(err as Error).message}`);
    }
  });

  // ────────────────────────────────────────────
  // 2. agent_start + message_end + agent_end — Save round + embed it
  // ────────────────────────────────────────────
  // agent_start/agent_end fire once per user prompt (unlike turn_start/turn_end
  // which fire per inner LLM call within a tool-calling loop). By saving at
  // agent_end we capture the FULL assistant response across all tool iterations.
  pi.on("agent_start", async (event, _ctx) => {
    const { messages } = event;
    // Extract the first user message content as the prompt for this agent cycle
    const firstUser = messages?.find((m: { role: string }) => m.role === "user");
    if (firstUser) {
      const content = firstUser.content;
      if (typeof content === "string") {
        agentUserPrompt = content;
      } else if (Array.isArray(content)) {
        agentUserPrompt = extractText(content as Array<{ type: string; text?: string }>);
      }
    }
    agentTurnIndex = event.turnIndex ?? null;
    if (agentTurnIndex != null) {
      agentTurnTimestamps.set(agentTurnIndex, Date.now());
    }
    agentAccumulatedText = [];
    agentToolCallCount = 0;
    agentToolCallNames = [];
    agentToolCalls = [];

    // Reset context embedding cache — new agent cycle = new user prompt
    lastContextUserPrompt = null;
    lastContextVec = [];
    agentPromptVec = null;
  });

  pi.on("message_end", async (event, _ctx) => {
    const msg = event.message;
    if (!msg) return;

    if (msg.role === "user") {
      // User sent something -- don't reset the accumulator, this is a new agent
      // cycle (agent_start will reset it). Keep safe.
    } else if (msg.role === "assistant") {
      // Extract text from this assistant message
      const content = msg.content as Array<{ type: string; text?: string }> | undefined;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === "text" && block.text) {
            agentAccumulatedText.push(block.text);
          } else if (block.type === "toolCall") {
            agentToolCallCount++;
            const blockRec = block as Record<string, unknown>;
            const name = blockRec.name as string | undefined;
            const id = blockRec.id as string | undefined;
            if (name && !agentToolCallNames.includes(name)) {
              agentToolCallNames.push(name);
            }
            if (id && name) {
              const detail: ToolCallDetail = {
                index: agentToolCalls.length,
                name,
                arguments: JSON.stringify(blockRec.arguments ?? {}),
                result_summary: "",
              };
              agentToolCalls.push(detail);
            }
          }
        }
      }
    } else if (msg.role === "toolResult") {
      // Pair tool results with their calls
      const toolCallId = msg.toolCallId as string | undefined;
      if (toolCallId) {
        // Find the matching ToolCallDetail by matching the last call without a result
        // (pi sessions don't expose the toolCallId -> toolCall mapping directly, so
        // we match sequentially — results arrive in order)
        for (let i = agentToolCalls.length - 1; i >= 0; i--) {
          if (agentToolCalls[i].result_summary === "") {
            const resultContent = msg.content as Array<{ type: string; text?: string }> | undefined;
            const resultText = resultContent ? extractText(resultContent) : "";
            agentToolCalls[i].result_summary = resultText.slice(0, 300);
            break;
          }
        }
      }
    }
  });

  pi.on("agent_end", async (event, ctx) => {
    const { messages } = event;

    // Note: we intentionally do NOT delete the timestamp here.
    // The context hook filters by ti < currentTurnIndex to show only completed
    // turns. If we deleted here, the next agent's context would have no
    // timestamps to show (agent_end fires before the next agent_start).

    // Get user prompt -- prefer agent_start cached value, fall back to messages
    let userPrompt = agentUserPrompt ?? "";
    if (!userPrompt && messages) {
      const lastUser = [...messages].reverse().find((m: { role: string }) => m.role === "user");
      if (lastUser) {
        const content = lastUser.content;
        if (typeof content === "string") {
          userPrompt = content;
        } else if (Array.isArray(content)) {
          userPrompt = extractText(content as Array<{ type: string; text?: string }>);
        }
      }
    }

    if (!userPrompt) {
      ctx.ui.setStatus("semblr", "\u{1f9e0} agent_end: no user prompt to save");
      return;
    }

    // Build response text from accumulated assistant text across all tool iterations
    let responseText = agentAccumulatedText.join("\n\n").trim();
    if (!responseText) {
      // Fallback: extract text from messages (last assistant message)
      const lastAssistant = messages ? [...messages].reverse().find((m: { role: string }) => m.role === "assistant") : null;
      if (lastAssistant) {
        const content = lastAssistant.content;
        if (typeof content === "string") {
          responseText = content;
        } else if (Array.isArray(content)) {
          responseText = extractText(content as Array<{ type: string; text?: string }>);
        }
      }
    }

    if (!responseText) {
      ctx.ui.setStatus("semblr", "\u{1f9e0} agent_end: no response text");
      return;
    }

    fs.mkdirSync(ROUNDS_DIR, { recursive: true });

    const roundFileName = createRoundFilePath(userPrompt, responseText);
    const roundPath = `${ROUNDS_DIR}/${roundFileName}`;

    // Push to causal chain — even on dedup, this ensures the in-memory buffer
    // tracks every round seen in this session.
    {
      const toolCalls = agentToolCalls.length;
      const names = agentToolCallNames;
      const toolSummary = toolCalls > 0
        ? `${toolCalls} tools (${names.join(", ")})`
        : "0 tools (discussion)";
      causalChain.push({
        fileName: roundFileName,
        userPrompt,
        responseSequence: responseText,
        toolSummary,
      });
    }

    // Skip if already saved (deduplication by content hash)
    if (fs.existsSync(roundPath)) {
      ctx.ui.setStatus("semblr", `\u{1f9e0} round already saved (${roundFileName})`);
      lastRoundFileName = roundFileName;
      agentAccumulatedText = [];
      agentUserPrompt = null;
      agentTurnIndex = null;
      return;
    }

    // Write round file
    const roundData = {
      id: crypto.createHash("md5").update(userPrompt + responseText).digest("hex"),
      userPrompt,
      responseSequence: responseText,
      turnIndex: agentTurnIndex ?? 0,
      userTimestamp: Date.now(),
      toolCallCount: agentToolCallCount,
      toolCallNames: agentToolCallNames,
      toolCalls: agentToolCalls,
    };

    try {
      fs.writeFileSync(roundPath, JSON.stringify(roundData, null, 2));
    } catch (err) {
      ctx.ui.setStatus("semblr", `\u{1f9e0} write error: ${(err as Error).message}`);
      agentAccumulatedText = [];
      agentUserPrompt = null;
      agentTurnIndex = null;
      return;
    }

    // Embed prompt and response separately
    const apiKey = await getApiKey(ctx);
    if (!apiKey) {
      ctx.ui.setStatus("semblr", "\u{1f9e0} saved but not embedded (no API key)");
      lastRoundFileName = roundFileName;
      agentAccumulatedText = [];
      agentUserPrompt = null;
      agentTurnIndex = null;
      return;
    }

    try {
      const [promptVec, responseVec] = await Promise.all([
        agentPromptVec
          ? Promise.resolve(agentPromptVec)
          : embedText(userPrompt, apiKey),
        embedText(responseText, apiKey),
      ]);
      appendToIndex(`${roundFileName}:prompt`, promptVec);
      appendToIndex(`${roundFileName}:response`, responseVec);
      ctx.ui.setStatus(
        "semblr",
        `\u{1f9e0} saved + embedded round (${roundFileName})`,
      );
    } catch (err) {
      ctx.ui.setStatus("semblr", `\u{1f9e0} embedding error: ${(err as Error).message}`);
    }

    lastRoundFileName = roundFileName;
    agentAccumulatedText = [];
    agentUserPrompt = null;
    agentTurnIndex = null;
  });

  // ────────────────────────────────────────────
  // 4. Cancel pi's internal compaction
  // ────────────────────────────────────────────
  // We save complete rounds via agent_end and retrieve them via semantic search.
  // Letting pi compact would throw away message-level detail we need for
  // accurate retrieval and tool call metadata. Cancelling keeps the full chain.
  pi.on("session_before_compact", async (_event, _ctx) => {
    return { cancel: true };
  });

  // ─────────────────────────────────────────────
  // 5. Startup — register tool + show status
  // ─────────────────────────────────────────────
  // registerTool is called inside session_start because factory-level
  // registration doesn't reliably make tools visible to the LLM.
  pi.on("session_start", async (_event, ctx) => {
    // Clear causal chain — new session starts fresh
    causalChain = [];

    const indexExists = fs.existsSync(INDEX_PATH);
    const index = indexExists ? loadIndex() : [];
    const uniqueRounds = new Set(
      index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
    ).size;
    ctx.ui.notify(
      `🧠 semblr loaded — ${uniqueRounds} rounds indexed`,
      "info",
    );

    // Register the search_interactions tool here, not at factory level
    pi.registerTool({
      name: "search_interactions",
      label: "Search Interactions",
      description: "Search all past user interactions for topics, questions, or discussions. Unlike the built-in search_memory (which searches within the current session), this searches across ALL sessions the user has ever had — every conversation round ever indexed. Use this when you need to find something from a past session, recall prior discussions, or reconnect with knowledge that was established a long time ago.\n\nYou can optionally scope the search to specific round files by passing the `turns` parameter. This is useful when you want to drill down into a specific subset of rounds.",
      promptSnippet: "Search past interactions for relevant context",
      parameters: Type.Object({
        query: Type.String({ description: "The search query — what you want to find in past conversations" }),
        minSimilarity: Type.Optional(Type.Number({ description: "Minimum similarity threshold (0.0 to 1.0). Default 0.25. Lower to get broader matches." })),
        rounds: Type.Optional(Type.Array(Type.String(), { description: "Optional list of round filenames to scope the search to (e.g., ['abc.json', 'def.json']). When provided, only these round files are searched." })),
      }),
      async execute(toolCallId, params, signal, onUpdate, ctx2) {
        const p = params as { query: string; minSimilarity?: number; rounds?: string[] };
        const query = p.query;
        if (!query) {
          return {
            content: [{ type: "text", text: "No query provided." }],
            details: {},
          };
        }
        const threshold = p.minSimilarity ?? 0.25;
        const scopeRounds = p.rounds ?? null;

        const apiKey = await getApiKey(ctx2);
        if (!apiKey) {
          return {
            content: [{ type: "text", text: "No API key available for embedding. Skipping search." }],
            details: {},
          };
        }

        // Embed the query
        const queryVec = normalize(await embedText(query, apiKey));

        // Load index and score
        let index = loadIndex();
        if (index.length === 0) {
          return {
            content: [{ type: "text", text: "The round index is empty. No conversations have been saved yet." }],
            details: {},
          };
        }

        // If rounds[] is provided, scope the search to only those round files
        if (scopeRounds && scopeRounds.length > 0) {
          const scopeSet = new Set(scopeRounds);
          index = index.filter((entry) => {
            const roundFile = entry.filePath.replace(/:prompt$|:response$/, "");
            return scopeSet.has(roundFile);
          });
          if (index.length === 0) {
            return {
              content: [{ type: "text", text: `No indexed vectors found for the specified rounds: ${scopeRounds.join(", ")}. They may not be embedded yet.` }],
              details: {},
            };
          }
        }

        const scored = index
          .map((entry) => ({ ...entry, similarity: cosineSimilarity(queryVec, entry.vector) }))
          .sort((a, b) => b.similarity - a.similarity);

        // Group by round file, take best score per round
        const roundScores = new Map<string, { fileName: string; data: RoundData; bestScore: number }>();
        for (const entry of scored) {
          const roundFile = entry.filePath.replace(/:prompt$|:response$/, "");
          if (!roundFile.endsWith(".json")) continue;
          if (roundScores.has(roundFile)) continue;
          const roundData = readRoundFile(entry.filePath);
          if (!roundData) continue;
          roundScores.set(roundFile, { fileName: roundFile, data: roundData, bestScore: entry.similarity });
        }

        const sorted = Array.from(roundScores.values())
          .sort((a, b) => b.bestScore - a.bestScore);

        if (sorted.length === 0) {
          return {
            content: [{ type: "text", text: "No matching turns found in the index." }],
            details: {},
          };
        }

        // Build result text — top 5 rounds with score.
        // Branch: collapsed mode shows full prompt + tool turn index + full response;
        // full mode keeps the existing truncated format.
        const MIN_SIMILARITY = threshold;
        const lines: string[] = [];
        let count = 0;
        for (const round of sorted) {
          if (round.bestScore < MIN_SIMILARITY) break;
          if (count >= 5) break;
          count++;
          const toolStr = round.data.toolCallCount != null && round.data.toolCallCount > 0
            ? ` | ${round.data.toolCallCount} tools (${(round.data.toolCallNames ?? []).join(", ")})`
            : (round.data.toolCallCount === 0 ? " | 0 tools (discussion only)" : "");

          if (ROUND_COLLAPSE_MODE === "collapsed") {
            // Collapsed: full prompt + tool turn index + full response.
            // Exposes the round filename so the model can call get_round_details().
            lines.push(`--- Round ${round.fileName} (score: ${round.bestScore.toFixed(3)}${toolStr}) ---`);
            lines.push(`User: ${round.data.userPrompt}`);

            // Build collapsed tool call turns (mirrors last-round-context injection)
            if (round.data.toolCallCount != null && round.data.toolCallCount > 0 && round.data.toolCalls && round.data.toolCalls.length > 0) {
              const turnLines = round.data.toolCalls.map((tc: ToolCallDetail) =>
                `  Turn ${tc.index}: ${tc.name} — [REDACTED: arguments and result collapsed. Use get_tool_details("${round.fileName}", ${tc.index}) to expand.]`
              );
              lines.push("--- Agent turns (all tool calls redacted — use get_tool_details to expand) ---");
              lines.push(...turnLines);
            } else if (round.data.toolCallCount === 0) {
              lines.push("--- Agent turns ---");
              lines.push("  (no tool calls — discussion only)");
            }

            lines.push(`Assistant: ${round.data.responseSequence}`);
            lines.push("");
          } else {
            // Full mode: truncated format (existing behavior)
            lines.push(`--- Round (score: ${round.bestScore.toFixed(3)}${toolStr}) ---`);
            lines.push(`User: ${round.data.userPrompt.slice(0, 500)}`);
            lines.push(`Assistant: ${round.data.responseSequence.slice(0, 1000)}`);
            lines.push("");
          }
        }

        if (count === 0) {
          return {
            content: [{ type: "text", text: `No relevant rounds found (best score: ${sorted[0].bestScore.toFixed(3)}).` }],
            details: {},
          };
        }

        return {
          content: [{ type: "text", text: `Found ${count} relevant rounds:\n\n${lines.join("\n")}` }],
          details: { matched: count, topScore: sorted[0].bestScore },
        };
      },
    });

    // Register get_round_details — retrieve the full content of a round file
    pi.registerTool({
      name: "get_round_details",
      label: "Get Round Details",
      description: "Retrieve the full content of a past conversation round by its filename hash. Unlike the truncated previews injected into context (which show only the first portion of the user prompt and assistant response), this returns the complete userPrompt and responseSequence for that round, plus all tool call metadata. Use this when you need to see the full conversation from a historical round.\n\nParameters:\n- round: the round filename (e.g., 'abc123.json')",
      promptSnippet: "Get full details of a past conversation round",
      parameters: Type.Object({
        round: Type.String({ description: "The round filename to look up (e.g., 'abc123def456.json')" }),
      }),
      async execute(toolCallId, params, signal, onUpdate, ctx2) {
        const p = params as { round: string };

        const fullPath = `${ROUNDS_DIR}/${p.round}`;
        if (!fs.existsSync(fullPath)) {
          return {
            content: [{ type: "text", text: `Round file not found: ${p.round}` }],
            details: {},
          };
        }

        let roundData: Record<string, unknown>;
        try {
          roundData = JSON.parse(fs.readFileSync(fullPath, "utf-8"));
        } catch {
          return {
            content: [{ type: "text", text: `Failed to parse round file: ${p.round}` }],
            details: {},
          };
        }

        // Build a readable summary of the full round
        let toolMeta = "";
        if (roundData.toolCallCount != null && Number(roundData.toolCallCount) > 0) {
          const names = (roundData.toolCallNames as string[])?.join(", ") ?? "unknown";
          toolMeta = `\n  Tools used: ${roundData.toolCallCount} (${names})`;
          if (roundData.toolCalls && Array.isArray(roundData.toolCalls) && (roundData.toolCalls as any[]).length > 0) {
            const hints = (roundData.toolCalls as any[]).slice(0, 10).map((tc: any) =>
              `${tc.index}:${tc.name}`
            ).join(" ");
            const more = (roundData.toolCalls as any[]).length > 10 ? ` ...+${(roundData.toolCalls as any[]).length - 10} more` : "";
            toolMeta += `\n  Indexes: ${hints}${more}\n  ALL tool call arguments and results are REDACTED — use get_tool_details("${p.round}", N) to expand individual calls.`;
          }
        } else if (roundData.toolCallCount === 0) {
          toolMeta = "\n  Tools used: 0 (discussion only)";
        }

        // Collapse tool call arguments and results in the details object.
        // Even when a round is "expanded" via get_round_details, the tool call
        // internals stay redacted — you must drill in with get_tool_details().
        const collapsedDetails = { ...roundData };
        if (collapsedDetails.toolCalls && Array.isArray(collapsedDetails.toolCalls)) {
          collapsedDetails.toolCalls = (collapsedDetails.toolCalls as any[]).map((tc: any) => ({
            ...tc,
            arguments: `[REDACTED — use get_tool_details("${p.round}", ${tc.index}) to expand]`,
            result_summary: `[REDACTED — use get_tool_details("${p.round}", ${tc.index}) to expand]`,
          }));
        }

        return {
          content: [{
            type: "text",
            text: `=== Round: ${p.round} ===\n` +
              `User: ${roundData.userPrompt ?? "(empty)"}\n` +
              `Assistant: ${roundData.responseSequence ?? "(empty)"}${toolMeta}`,
          }],
          details: collapsedDetails,
        };
      },
    });

    // Register get_tool_details — retrieve full tool call details from a round file
    pi.registerTool({
      name: "get_tool_details",
      label: "Get Tool Details",
      description: "Retrieve the full arguments and result of a specific tool call from a past conversation round. When you see historical rounds injected into context with TOOLS USED markers, you can call this tool to inspect what a specific tool did — its full arguments and the complete output (not just the preview stored in the round file).\n\nParameters:\n- round: the round filename (e.g., 'abc123.json')\n- index: the 0-based index of the tool call within that round's toolCalls array",
      promptSnippet: "Get details of a specific tool call from a past round",
      parameters: Type.Object({
        round: Type.String({ description: "The round filename to look up (e.g., 'abc123def456.json')" }),
        index: Type.Number({ description: "The 0-based index of the tool call within the round" }),
      }),
      async execute(toolCallId, params, signal, onUpdate, ctx2) {
        const p = params as { round: string; index: number };

        const fullPath = `${ROUNDS_DIR}/${p.round}`;
        if (!fs.existsSync(fullPath)) {
          return {
            content: [{ type: "text", text: `Round file not found: ${p.round}` }],
            details: {},
          };
        }

        let roundData: RoundData;
        try {
          roundData = JSON.parse(fs.readFileSync(fullPath, "utf-8"));
        } catch {
          return {
            content: [{ type: "text", text: `Failed to parse round file: ${p.round}` }],
            details: {},
          };
        }

        if (!roundData.toolCalls || roundData.toolCalls.length === 0) {
          return {
            content: [{ type: "text", text: `This round has no tool calls stored. It may have been indexed before tool call metadata was added. Consider re-indexing.` }],
            details: {},
          };
        }

        if (p.index < 0 || p.index >= roundData.toolCalls.length) {
          return {
            content: [{ type: "text", text: `Invalid index ${p.index}. This round has ${roundData.toolCalls.length} tool calls (indices 0–${roundData.toolCalls.length - 1}).` }],
            details: {},
          };
        }

        const tc = roundData.toolCalls[p.index];

        // Parse arguments back to object for display
        let argsParsed: unknown = tc.arguments;
        try {
          argsParsed = JSON.parse(tc.arguments);
        } catch { /* keep as string */ }

        return {
          content: [{
            type: "text",
            text: `Tool call #${tc.index} in round ${p.round}\n` +
              `  Name: ${tc.name}\n` +
              `  Arguments: ${JSON.stringify(argsParsed, null, 2)}\n` +
              `  Result (preview): ${tc.result_summary || "(empty)"}`,
          }],
          details: { name: tc.name, arguments: tc.arguments, result_preview: tc.result_summary },
        };
      },
    });
  });


}
