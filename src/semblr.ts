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
import * as path from "node:path";
import { spawn, spawnSync } from "node:child_process";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

// PI_CODING_AGENT_DIR overrides the default ~/.pi/agent config directory.
// We store semblr rounds under that directory so they survive project moves.
const PI_CONFIG_DIR = process.env.PI_CODING_AGENT_DIR || `${os.homedir()}/.pi/agent`;
const ROUNDS_DIR = `${PI_CONFIG_DIR}/semblr/rounds`;
const INDEX_PATH = `${ROUNDS_DIR}/index.csv`;
const SEMBLR_DIR = `${PI_CONFIG_DIR}/semblr`;
const STATS_PATH = `${SEMBLR_DIR}/chain-read-stats.json`

// ◈ Causal-chain read statistics — global, never injected into context
//   Tracks all 5 causal-chain display positions (1-5, where 1 = most recent round).
//   positionScores[i]: presentedCount vs readCount for display position i+1
//   (i=0 = most recent round, i=4 = oldest in the 5-entry window)
//   Flushed atomically at agent_end. NOT reset on /new.
const TRACK_POSITIONS = 5; // hard-coded per user request — re-evaluate if readRate on any position > 50%
let statsState = loadStats();
// Hashes presented at each display position (1-5) in the current context.
// Set in context hook, consumed by recordRead / recordPresented.
// Index 0 = display position 1 (most recent), index 4 = display position 5 (oldest).
let statsPresentedHashes: (string | null)[] = [null, null, null, null, null];

function loadStats() {
  try {
    if (fs.existsSync(STATS_PATH)) {
      const loadedStats = JSON.parse(fs.readFileSync(STATS_PATH, "utf-8"));
      // Migrate from v1 (position5 scalar) to v2 (positionScores array)
      if (loadedStats.version === 1 && loadedStats.position5) {
        const old = loadedStats.position5;
        loadedStats.version = 2;
        loadedStats.positionScores = [
          { presentedCount: 0, readCount: 0, presentedHash: null },
          { presentedCount: 0, readCount: 0, presentedHash: null },
          { presentedCount: 0, readCount: 0, presentedHash: null },
          { presentedCount: 0, readCount: 0, presentedHash: null },
          { presentedCount: old.presentedCount, readCount: old.readCount, presentedHash: null },
        ];
        delete loadedStats.position5;
        return loadedStats;
      }
      return loadedStats;
    }
  } catch { /* corrupt file, reset */ }
  return {
    version: 2,
    positionScores: [
      { presentedCount: 0, readCount: 0, presentedHash: null },
      { presentedCount: 0, readCount: 0, presentedHash: null },
      { presentedCount: 0, readCount: 0, presentedHash: null },
      { presentedCount: 0, readCount: 0, presentedHash: null },
      { presentedCount: 0, readCount: 0, presentedHash: null },
    ],
    lastUpdated: new Date().toISOString(),
  };
}

function flushStats() {
  statsState.lastUpdated = new Date().toISOString();
  const tmp = `${STATS_PATH}.tmp.${process.pid}`;
  try {
    fs.mkdirSync(SEMBLR_DIR, { recursive: true });
    fs.writeFileSync(tmp, JSON.stringify(statsState, null, 2));
    fs.renameSync(tmp, STATS_PATH); // atomic on POSIX
  } catch (e) {
    // Best-effort; stats collection must never break the extension
    try { if (fs.existsSync(tmp)) fs.unlinkSync(tmp); } catch {}
  }
}

/** Record a read — increment readCount for any position whose hash matches. */
function recordRead(hash: string) {
  for (let i = 0; i < TRACK_POSITIONS; i++) {
    if (statsPresentedHashes[i] && statsPresentedHashes[i] === hash) {
      statsState.positionScores[i].readCount++;
    }
  }
}

/**
 * Record presentation for all 5 causal-chain positions; also bin the best score.
 * `chainEntries` is the chronological causal chain (oldest first, newest last).
 * Display positions (1-5 = index 0 to 4) map to reversed entries:
 *   index 0 = newest (last element of chain)
 *   index 4 = oldest in the 5-entry window (element at chain.length - 5, if available)
 */
function recordPresented(chainEntries: { fileName: string }[]) {
  // Build the reversed (newest-first) view
  const reversed = [...chainEntries].reverse();
  for (let i = 0; i < TRACK_POSITIONS; i++) {
    const entry = i < reversed.length ? reversed[i] : null;
    const hash = entry ? entry.fileName : null;
    statsPresentedHashes[i] = hash;
    if (hash) {
      statsState.positionScores[i].presentedCount++;
    }
  }
}

/**
 * Format the chain-read statistics for TUI display as percentages.
 * Example: "🧠 chain-read: p1→2/16(13%) p2→0/0 p3→0/0 p4→0/0 p5→0/0"
 */
function formatChainStats(): string {
  const parts = statsState.positionScores.map((ps, i) => {
    const pct = ps.presentedCount > 0
      ? `(${Math.round((ps.readCount / ps.presentedCount) * 100)}%)`
      : "(—%)";
    return `p${i + 1}→${ps.readCount}/${ps.presentedCount}${pct}`;
  });
  return `chain-read: ${parts.join(" ")}`;
}

const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";

const CONTEXT_BUDGET_RATIO = 0.5; // 50% of model context window for historical rounds

// Collapsed-only mode. All rounds are injected via the Recency and Relevance Lists.
// Use get_round_details() to expand. The Recency List contains the in-memory causal
// chain from the current session; the Relevance List contains semantically similar
// rounds from all past sessions.

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

/**
 * Format a single round entry — identical structure for both Recency and Relevance lists.
 * Produces:
 *   N. hash.json [score | N tools]:
 *     user: first line of prompt
 *     subsequent lines indented
 *     ---
 */
function formatRoundEntry(
  idx: number,
  fileName: string,
  score: string,
  toolSummary: string,
  userPrompt: string,
  sizeStr?: string,
): string[] {
  const promptLines = userPrompt
    .split("\n")
    .map((line, i) => i === 0 ? `  user: ${line}` : `  ${line}`);
  const sizePart = sizeStr ? ` | ${sizeStr}` : "";
  return [
    `${idx}. ${fileName} [${score} | ${toolSummary}${sizePart}]:`,
    ...promptLines,
    "  ---",
  ];
}

/** Build the Recency List section from the in-memory causal chain. */
function buildRecencyList(chain: ChainEntry[]): string | null {
  if (chain.length === 0) return null;
  const lines: string[] = [];
  const header = `--- RECENCY LIST (current session, newest first) ---
These rounds have n/a scores because they are presented by recency — they form
the immediate conversational context from this session.

IMPORTANT: This list shows ONLY the user's questions from past rounds.
You do NOT have the assistant responses or tool results unless you expand a
round. If you answer based on these prompts alone, you are hallucinating.

Use this list when the current prompt ...:
- ... asks about past work, decisions, code, or findings from prior sessions
- ... is unusually short or lacks clear context/goals/outputs
- ... uses references with no clear antecedent in the causal chain ("that fix",
  "the plan", "where we left off")
- ... asks you to remember, verify, continue, or build upon prior work
- ... requires cross-session continuity (same project, recurring topic,
  long-running task)
- ... is ambiguous: lacks proper context or references, and seems to assume
  knowledge was established

When this happens:
1. Scan the list prompts for relevance. Higher score = stronger match.
2. If a round looks relevant, expand ONLY that round via get_round_details.
3. Stop as soon as the expanded round gives you enough context to answer.
4. If no round looks relevant but the query clearly needs past context,
   use search_interactions.

When NOT to expand:
- The query is fully self-contained (clear context, goals, and outputs present).
- The prompts in the context already provides sufficient information.

Rule: When in doubt, expand. A verification tool call is cheaper than a wrong
answer.`;
  lines.push(header);
  lines.push("");

  // Show newest first (reverse chronological)
  const reversed = [...chain].reverse();
  let idx = 0;
  for (const entry of reversed) {
    idx++;
    const sizeStr = getRoundSize(entry.fileName) ?? undefined;
    lines.push(...formatRoundEntry(
      idx,
      entry.fileName,
      "n/a",
      entry.toolSummary,
      entry.userPrompt,
      sizeStr,
    ));
  }
  return lines.join("\n");
}

/** Build the Relevance List section from scored rounds. */
function buildRelevanceList(
  rounds: Array<{ fileName: string; bestScore: number; data: RoundData }>,
): string | null {
  if (rounds.length === 0) return null;
  const lines: string[] = [];
  const header = `--- RELEVANCE LIST (all sessions, by similarity) ---
These rounds have numeric similarity scores (0.0–1.0). Higher = stronger
semantic match. They come from ALL past sessions, not just the current one.

The extension has pre-run a semantic search against your prompt. The results
are below. If something here rings a bell, expand it via get_round_details.
If nothing rings a bell, ignore this list — it's a pre-filter, not a map.`;
  lines.push(header);
  lines.push("");

  let idx = 0;
  for (const round of rounds) {
    idx++;
    const toolCount = round.data.toolCallCount ?? 0;
    const sizeStr = getRoundSize(round.fileName) ?? undefined;

    // Build per-tool size tags where data is available
    let toolSummary = `${toolCount} tools`;
    if (round.data.toolCalls && round.data.toolCalls.length > 0) {
      const toolSizes = round.data.toolCalls.map((tc: ToolCallDetail) => {
        const sourceText = tc.result_full ?? tc.result_summary ?? "";
        return sourceText.length > 0
          ? `${tc.name} (${formatFileSize(Buffer.byteLength(sourceText, "utf-8"))})`
          : tc.name;
      });
      toolSummary = `${toolCount} tools (${toolSizes.join(", ")})`;
    }

    lines.push(...formatRoundEntry(
      idx,
      round.fileName,
      round.bestScore.toFixed(2),
      toolSummary,
      round.data.userPrompt,
      sizeStr,
    ));
  }
  return lines.join("\n");
}

/** Build the Context Building References preamble section. */
function buildContextPreamble(
  hasRecency: boolean,
  hasRelevance: boolean,
): string | null {
  if (!hasRecency && !hasRelevance) return null;
  return `[CONTEXT BUILDING REFERENCES]
The lists below show past conversation rounds. Each entry contains only the user prompt — responses and tool calls are collapsed.
Use get_round_details("hash.json") to expand a round's full conversation.
Use get_tool_details("hash.json", N) to inspect tool call N within a round.

Format: N. hash.json [score | N tools | size]: followed by the full user prompt (indented).
Number 1 in the list is the most recent round.`;
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

/** Format a byte count into human-readable KB/MB. */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${Math.round(bytes / 10.24) / 100}KB`;       // 0.5KB
  if (bytes < 10240) return `${Math.round(bytes / 1024)}KB`;             // 5KB
  if (bytes < 1048576) return `${Math.round(bytes / 1024)}KB`;           // 54KB
  return `${Math.round(bytes / 10485.76) / 100}MB`;                      // 1.1MB
}

/** Stat a round file and return its formatted size string, or null on failure. */
function getRoundSize(fileName: string): string | null {
  try {
    const stat = fs.statSync(`${ROUNDS_DIR}/${fileName}`);
    return formatFileSize(stat.size);
  } catch {
    return null;
  }
}

// formatCollapsedIndex removed — replaced by buildRecencyList / buildRelevanceList / buildContextPreamble

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
  result_summary: string; // first 300 chars of result text (legacy field, kept for backward compat)
  result_full?: string;   // full tool output (no cap, for post-step-2 rounds)
  result_truncated?: boolean; // true if result exceeds cap (only false for post-step-2 rounds since no cap)
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

// Interleaved response segments — preserves tool call positions within assistant text
let agentResponseSegments: Array<{ type: "text" | "toolCall"; text?: string; toolCallIndex?: number }> = [];

// Context embedding cache — avoids redundant embedding API calls across tool turns
// within the same agent cycle. Reset in agent_start.
let lastContextUserPrompt: string | null = null;
let lastContextVec: number[] = [];
let agentPromptVec: number[] | null = null; // cached from context hook, reused in agent_end to avoid redundant embed
let roundPresentedRecorded = false; // dedup: recordPresented only once per agent cycle

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

function splitCommandArgs(args: string): string[] {
  const out: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;
  let escaping = false;

  for (const ch of args) {
    if (escaping) {
      current += ch;
      escaping = false;
      continue;
    }
    if (ch === "\\") {
      escaping = true;
      continue;
    }
    if (quote) {
      if (ch === quote) quote = null;
      else current += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (/\s/.test(ch)) {
      if (current) {
        out.push(current);
        current = "";
      }
      continue;
    }
    current += ch;
  }
  if (escaping) current += "\\";
  if (current) out.push(current);
  return out;
}

function extensionRoot(): string {
  // src/semblr.ts -> project root. __dirname is available in pi's jiti runtime.
  return path.resolve(__dirname, "..");
}

// ─────────────────────────────────────────────
// Extension
// ─────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  pi.registerCommand("semblr:import-claude", {
    description: "Import Claude Code history into Semblr (/semblr:import-claude --dry-run, --limit N, --include-sidechains)",
    getArgumentCompletions: (prefix: string) => {
      const options = ["--dry-run", "--include-sidechains", "--limit"];
      const matches = options.filter((opt) => opt.startsWith(prefix));
      return matches.length ? matches.map((value) => ({ value, label: value })) : null;
    },
    handler: async (args, ctx) => {
      const root = extensionRoot();
      const script = path.resolve(root, "scripts", "import-claude-code.ts");
      if (!fs.existsSync(script)) {
        ctx.ui.notify(`Semblr import script not found: ${script}`, "error");
        return;
      }

      const parsedArgs = splitCommandArgs(args);
      ctx.ui.setStatus("semblr", `🧠 importing Claude Code history ${parsedArgs.join(" ")}`.trim());
      ctx.ui.notify(`Starting Claude Code import${parsedArgs.length ? ` (${parsedArgs.join(" ")})` : ""}...`, "info");

      const apiKey = await getApiKey(ctx as Parameters<typeof getApiKey>[0]);
      const child = spawn("npx", ["tsx", script, ...parsedArgs], {
        cwd: root,
        env: {
          ...process.env,
          ...(apiKey ? { OPENROUTER_API_KEY: apiKey } : {}),
          SEMBLR_ROUNDS_DIR: ROUNDS_DIR,
        },
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      const keepTail = (s: string) => s.length > 4000 ? s.slice(-4000) : s;
      child.stdout.on("data", (chunk) => { stdout = keepTail(stdout + chunk.toString()); });
      child.stderr.on("data", (chunk) => {
        const text = chunk.toString();
        stderr = keepTail(stderr + text);
        const lastLine = text.trim().split("\n").filter(Boolean).pop();
        if (lastLine) ctx.ui.setStatus("semblr", `🧠 ${lastLine.slice(0, 120)}`);
      });

      const code = await new Promise<number | null>((resolve) => {
        child.on("close", resolve);
      });

      const output = [stdout.trim(), stderr.trim()].filter(Boolean).join("\n");
      const tail = output.split("\n").slice(-8).join("\n");
      if (code === 0) {
        ctx.ui.setStatus("semblr", "🧠 Claude Code import complete");
        ctx.ui.notify(`Claude Code import complete${tail ? `:\n${tail}` : ""}`, "info");
      } else {
        ctx.ui.setStatus("semblr", `🧠 Claude Code import failed (${code})`);
        ctx.ui.notify(`Claude Code import failed (${code})${tail ? `:\n${tail}` : ""}`, "error");
      }
    },
  });

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

    // --- Prepend environment info to the current user prompt ---
    const envPreamble = `[ENVIRONMENT]\nHost: ${os.hostname()}\nCWD: ${process.cwd()}\nCurrent date/time: ${new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+/, "")}`;

    const augmentedMessages = [...messages];
    if (lastUserIdx >= 0) {
      const userMsg = augmentedMessages[lastUserIdx];
      if (typeof userMsg.content === "string") {
        augmentedMessages[lastUserIdx] = { ...userMsg, content: `${envPreamble}\n\n${userMsg.content}` };
      } else if (Array.isArray(userMsg.content) && userMsg.content.length > 0 && userMsg.content[0].type === "text") {
        const newContent = [...userMsg.content];
        newContent[0] = { ...newContent[0], text: `${envPreamble}\n\n${newContent[0].text}` };
        augmentedMessages[lastUserIdx] = { ...userMsg, content: newContent };
      } else if (Array.isArray(userMsg.content)) {
        augmentedMessages[lastUserIdx] = { ...userMsg, content: [{ type: "text", text: `${envPreamble}\n\n` }, ...userMsg.content] };
      }
    }

    const currentMessages = lastUserIdx >= 0
      ? augmentedMessages.slice(lastUserIdx)
      : [...augmentedMessages];

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
      return { messages: augmentedMessages };
    }

    // agentPromptVec is stashed after embedding below for agent_end to reuse


    try {
      const apiKey = await getApiKey(ctx);
      if (!apiKey) return { messages: augmentedMessages };

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
      if (index.length === 0) return { messages: augmentedMessages };

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
      if (lastRoundFileName) {
        const lastData = readRoundFile(lastRoundFileName);
        if (lastData) {
          addRound({ data: lastData, fileName: lastRoundFileName, bestScore: 0 });
        }
      }

      if (selectedRounds.length === 0) {
        ctx.ui.setStatus(
          "semblr",
          `🧠 no relevant context (best: ${bestScore.toFixed(3)})`,
        );
        return { messages: augmentedMessages };
      }

      // ── Build the three-section context block ──
        const dropRelevance = process.env.DROP_RELEVANCE_LIST === "1" || process.env.DROP_RELEVANCE_LIST === "true";
        const relevanceList = dropRelevance
          ? null
          : buildRelevanceList(
              selectedRounds.map(r => ({ fileName: r.fileName, bestScore: r.bestScore, data: r.data }))
            );
        const recencyList = buildRecencyList(causalChain);
        const preamble = buildContextPreamble(!!recencyList, !!relevanceList);

        // ══ Stats: record all 5 positions presented ══
        {
          if (!roundPresentedRecorded) {
            recordPresented(causalChain);
            roundPresentedRecorded = true;
          }
          const uniqueRounds = new Set(
            index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
          ).size;
          ctx.ui.setStatus(
            "semblr",
            `🧠 collapsed: ${selectedRounds.length} matched / ${uniqueRounds} total | ${formatChainStats()}`,
          );
        }

        const finalMessages: Array<{ role: string; content: Array<{ type: string; text: string }> }> = [];
        if (systemMsg) finalMessages.push(systemMsg);

        // Section 1: Context Building References (explains format and names the two lists)
        if (preamble) {
          finalMessages.push({
            role: "user" as const,
            content: [{ type: "text" as const, text: preamble }],
          });
        }

        // Section 2: Recency List (if current session has prior rounds)
        if (recencyList) {
          finalMessages.push({
            role: "user" as const,
            content: [{ type: "text" as const, text: recencyList }],
          });
        }

        // Section 3: Relevance List (if semantic search returned matches)
        if (relevanceList) {
          finalMessages.push({
            role: "user" as const,
            content: [{ type: "text" as const, text: relevanceList }],
          });
        }

        finalMessages.push(...currentMessages);

        return { messages: finalMessages };
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
    agentAccumulatedText = [];
    agentToolCallCount = 0;
    agentToolCallNames = [];
    agentToolCalls = [];
    agentResponseSegments = [];

    // Reset context embedding cache — new agent cycle = new user prompt
    lastContextUserPrompt = null;
    lastContextVec = [];
    agentPromptVec = null;
    roundPresentedRecorded = false;
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
            agentResponseSegments.push({ type: "text", text: block.text });
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
            // Record this tool call's position in the response stream
            agentResponseSegments.push({ type: "toolCall", toolCallIndex: agentToolCalls.length - 1 });
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
            agentToolCalls[i].result_full = resultText;
            agentToolCalls[i].result_truncated = false;
            break;
          }
        }
      }
    }
  });

  pi.on("agent_end", async (event, ctx) => {
    const { messages } = event;

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
      flushStats(); // causal chain was pushed, so position scores may have changed
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
      responseSegments: agentResponseSegments,
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

    // ◈ Flush stats to disk (atomically) and show in TUI
    flushStats();

    // Combine chain-read stats with total indexed rounds count
    const indexExists = fs.existsSync(INDEX_PATH);
    const idx = indexExists ? loadIndex() : [];
    const totalRounds = new Set(
      idx.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
    ).size;
    ctx.ui.setStatus("semblr", `🧠 ${totalRounds} total indexed | ${formatChainStats()}`);
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
    ctx.ui.setStatus(
      "semblr",
      `🧠 semblr loaded — ${uniqueRounds} rounds indexed`,
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

          // search_interactions always shows full round content.
          const roundSizeStr = getRoundSize(round.fileName);
          const sizeTag = roundSizeStr ? ` | ${roundSizeStr}` : "";
          lines.push(`--- Round ${round.fileName} (score: ${round.bestScore.toFixed(3)}${toolStr}${sizeTag}) ---`);
          lines.push(`User: ${round.data.userPrompt}`);

          if (round.data.toolCallCount != null && round.data.toolCallCount > 0 && round.data.toolCalls && round.data.toolCalls.length > 0) {
            const turnLines = round.data.toolCalls.map((tc: ToolCallDetail) => {
              // Calculate tool output size from result_full or result_summary
              const sourceText = tc.result_full ?? tc.result_summary ?? "";
              const sizeLabel = sourceText.length > 0 ? formatFileSize(Buffer.byteLength(sourceText, "utf-8")) : null;
              const sizeTag = sizeLabel ? ` (${sizeLabel})` : "";
              return `  Turn ${tc.index}: ${tc.name}${sizeTag} — [REDACTED: use get_tool_details("${round.fileName}", ${tc.index}) to expand.]`;
            });
            lines.push("--- Agent turns (all tool calls redacted — use get_tool_details to expand) ---");
            lines.push(...turnLines);
          } else if (round.data.toolCallCount === 0) {
            lines.push("--- Agent turns ---");
            lines.push("  (no tool calls — discussion only)");
          }

          lines.push(`Assistant: ${round.data.responseSequence}`);
          lines.push("");
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
      description: "Retrieve the full content of a past conversation round by its filename hash. Unlike the truncated previews injected into context (which show only the first portion of the user prompt and assistant response), this returns the complete userPrompt and responseSequence for that round, plus all tool call metadata. Use this when you need to see the full conversation from a historical round.\n\nParameters:\n- round: the round filename (e.g., 'abc123.json')\n- from_line: optional 1-based line offset into the assistant response (default: 1). When specified, the response sequence is paginated on line boundaries. Mutually exclusive with match.\n- line_count: optional max lines of assistant response to return. Default: 200 when from_line is specified, 0 when match is specified (just the matched line). Omit both for full response.\n- match: optional regexp pattern to find within the round (user prompt + assistant response). Mutually exclusive with from_line. When present, line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return.\n- max_matches: max matches to return (default: 1). Has no effect without match.",
      promptSnippet: "Get full details of a past conversation round",
      parameters: Type.Object({
        round: Type.String({ description: "The round filename to look up (e.g., 'abc123def456.json')" }),
        from_line: Type.Optional(Type.Number({ description: "1-based line offset into the assistant response. Default: 1 (start). When specified, the response is paginated on line boundaries. Mutually exclusive with match." })),
        line_count: Type.Optional(Type.Number({ description: "Max lines of assistant response to return. Default: 200 when from_line is specified, 0 when match is specified. Omit both for full response." })),
        match: Type.Optional(Type.String({ description: "A regexp pattern to search within the round (user prompt + assistant response). Mutually exclusive with from_line. When provided, line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return." })),
        max_matches: Type.Optional(Type.Number({ description: "Max number of matches to return (default: 1). Has no effect without match." })),
      }),
      async execute(toolCallId, params, signal, onUpdate, ctx2) {
        const p = params as { round: string; from_line?: number; line_count?: number; match?: string; max_matches?: number };

        // ◈ Stats: check if this hash matches any presented position
        recordRead(p.round);

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

        // Build a readable summary of the full round.
        // If responseSegments exists (interleaved format), use it to produce
        // tool-call-at-position output. Otherwise fall back to flat format.
        let toolMeta = "";
        if (roundData.toolCallCount != null && Number(roundData.toolCallCount) > 0) {
          const names = (roundData.toolCallNames as string[])?.join(", ") ?? "unknown";
          toolMeta = `\n  Tools used: ${roundData.toolCallCount} (${names})`;
        } else if (roundData.toolCallCount === 0) {
          toolMeta = "\n  Tools used: 0 (discussion only)";
        }

        // Build interleaved assistant output using responseSegments when available
        let assistantOutput: string;
        if (roundData.responseSegments && Array.isArray(roundData.responseSegments) && (roundData.responseSegments as any[]).length > 0) {
          const parts: string[] = [];
          for (const seg of roundData.responseSegments as Array<{ type: string; text?: string; toolCallIndex?: number }>) {
            if (seg.type === "text" && seg.text) {
              parts.push(seg.text);
            } else if (seg.type === "toolCall" && seg.toolCallIndex != null) {
              parts.push(`[Tool call REDACTED: use get_tool_details("${p.round}", ${seg.toolCallIndex}) to expand]`);
            }
          }
          assistantOutput = parts.join("\n");
        } else {
          // Fall back to flat responseSequence for old round files
          assistantOutput = (roundData.responseSequence as string) ?? "(empty)";
        }

        // ── Pagination or Regexp Matching ──
        // Three modes:
        // 1. from_line (+ optional line_count) — position-based line slicing
        // 2. match (+ optional line_count, max_matches) — regexp-based context windows
        // 3. Neither — full response
        // If match is provided, it's mutually exclusive with from_line.
        const useMatch = p.match !== undefined && p.match.length > 0;
        const useFromLine = p.from_line !== undefined;
        let responseTotalLines = 0;
        let paginationMarker = "";
        let matchHeader = "";

        if (useMatch && useFromLine) {
          return {
            content: [{ type: "text", text: `Error: match and from_line are mutually exclusive. Use one or the other, not both.` }],
            details: {},
          };
        }

        if (useFromLine) {
          // Position-based pagination
          const flatText = (roundData.responseSequence as string) ?? "";
          const allLines = flatText.split("\n");
          responseTotalLines = allLines.length;
          const fromLine = p.from_line ?? 1;
          const lineCount = p.line_count ?? 200;
          const startIdx = Math.max(0, fromLine - 1);
          const endIdx = Math.min(responseTotalLines, startIdx + lineCount);
          const pageLines = allLines.slice(startIdx, endIdx);
          const remaining = responseTotalLines - endIdx;

          assistantOutput = pageLines.length > 0
            ? pageLines.join("\n")
            : "(empty)";

          if (remaining > 0) {
            paginationMarker = `[Truncated — use from_line=${endIdx + 1}, line_count=${lineCount} to continue]`;
          }
        } else if (useMatch) {
          // Regexp-based matching
          try {
            const regex = new RegExp(p.match!, "gm");
            const flatText = (roundData.responseSequence as string) ?? "";
            const allLines = flatText.split("\n");
            responseTotalLines = allLines.length;
            const lineCount = p.line_count ?? 0; // default: just the matched line
            const maxMatches = p.max_matches ?? 1;

            // Search both user prompt lines and assistant response lines
            // Build combined array with source markers
            const userLines = ((roundData.userPrompt as string) ?? "").split("\n");
            const searchLines: Array<{ text: string; source: "user" | "assistant"; originalIndex: number }> = [];
            for (let i = 0; i < userLines.length; i++) {
              searchLines.push({ text: userLines[i], source: "user", originalIndex: i });
            }
            // separator between user and assistant
            searchLines.push({ text: "", source: "assistant", originalIndex: -1 });
            for (let i = 0; i < allLines.length; i++) {
              searchLines.push({ text: allLines[i], source: "assistant", originalIndex: i });
            }

            // Find matches
            const matchResults: Array<{ lineIdx: number; source: "user" | "assistant"; text: string }> = [];
            for (let i = 0; i < searchLines.length; i++) {
              if (regex.test(searchLines[i].text)) {
                matchResults.push({
                  lineIdx: searchLines[i].originalIndex,
                  source: searchLines[i].source,
                  text: searchLines[i].text,
                });
                // Reset lastIndex for 'g' flag
                regex.lastIndex = 0;
              }
            }

            // Clip: if a match is in the user section, only show that one (cheapest)
            const userMatchCount = matchResults.filter(m => m.source === "user").length;
            let shownMatches = matchResults.slice(0, maxMatches);
            if (userMatchCount > 0) {
              // Only the first user match; clip early
              shownMatches = [matchResults[0]];
            }

            const totalMatches = matchResults.length;
            const matchCount = shownMatches.length;

            // Build output with matched lines and context windows
            const matchParts: string[] = [];
            for (let mi = 0; mi < matchCount; mi++) {
              const m = shownMatches[mi];
              if (m.source === "user") {
                // Show user prompt line
                matchParts.push(`[M ${mi + 1}/${matchCount} in user prompt] ${m.text}`);
              } else {
                // In assistant response: show matched line + lineCount lines after
                const startIdx = m.lineIdx;
                const endIdx = Math.min(responseTotalLines, startIdx + 1 + lineCount);
                const ctxLines = allLines.slice(startIdx, endIdx);
                const context = ctxLines.join("\n");
                if (lineCount > 0) {
                  matchParts.push(`[M ${mi + 1}/${matchCount} at assistant line ${m.lineIdx + 1} (${lineCount} lines of context)]\n${context}`);
                } else {
                  matchParts.push(`[M ${mi + 1}/${matchCount} at assistant line ${m.lineIdx + 1}] ${m.text}`);
                }
              }
            }

            const remainingMatches = totalMatches - matchCount;
            assistantOutput = matchParts.length > 0
              ? matchParts.join("\n\n")
              : "(no matches)";

            if (remainingMatches > 0) {
              matchHeader = ` (${matchCount} match${matchCount !== 1 ? "es" : ""} shown of ${totalMatches} total)`;
            } else if (totalMatches > 0) {
              const mLabel = totalMatches === 1 ? "match" : "matches";
              matchHeader = ` (${totalMatches} ${mLabel})`;
            }
          } catch (err) {
            return {
              content: [{ type: "text", text: `Invalid regexp pattern: ${(err as Error).message}. Provide a valid JavaScript regexp string.` }],
              details: {},
            };
          }
        }

        // Collapse tool call arguments and results in the details object.
        // Even when a round is "expanded" via get_round_details, the tool call
        // internals stay redacted — you must drill in with get_tool_details().
        // Include per-tool output sizes so the agent knows which call is expensive.
        const collapsedDetails = { ...roundData };
        if (collapsedDetails.toolCalls && Array.isArray(collapsedDetails.toolCalls)) {
          collapsedDetails.toolCalls = (collapsedDetails.toolCalls as any[]).map((tc: any) => {
            // Calculate tool output size from result_full or result_summary
            const sourceText = tc.result_full ?? tc.result_summary ?? "";
            const sizeLabel = formatFileSize(Buffer.byteLength(sourceText, "utf-8"));
            return {
              ...tc,
              arguments: `[REDACTED — use get_tool_details("${p.round}", ${tc.index}) to expand]`,
              result_summary: `[REDACTED — size: ${sizeLabel}; use get_tool_details("${p.round}", ${tc.index}) to expand]`,
              result_full: undefined, // strip inline to keep details compact
            };
          });
        }

        // Build header with position or match info
        let headerSuffix = "";
        if (useFromLine) {
          headerSuffix = ` (lines ${p.from_line ?? 1}–${Math.min((p.from_line ?? 1) - 1 + (p.line_count ?? 200), responseTotalLines)} of ${responseTotalLines})`;
        } else if (useMatch) {
          headerSuffix = matchHeader;
        }

        return {
          content: [{
            type: "text",
            text: `=== Round: ${p.round}${headerSuffix} ===\n` +
              `User: ${roundData.userPrompt ?? "(empty)"}\n` +
              `Assistant: ${assistantOutput}` +
              `${paginationMarker ? `\n${paginationMarker}` : ""}` +
              `${toolMeta}`,
          }],
          details: collapsedDetails,
        };
      },
    });

    // Register get_tool_details — retrieve full tool call details from a round file
    pi.registerTool({
      name: "get_tool_details",
      label: "Get Tool Details",
      description: "Retrieve the full arguments and result of a specific tool call from a past conversation round. When you see historical rounds injected into context with TOOLS USED markers, you can call this tool to inspect what a specific tool did — its full arguments and the complete output (not just the preview stored in the round file).\n\nParameters:\n- round: the round filename (e.g., 'abc123.json')\n- index: the 0-based index of the tool call within that round's toolCalls array\n- out__from_line: optional 1-based line offset into the result (default: 1). Mutually exclusive with match.\n- out_line_count: optional max lines to return (default: all, up to 200 when pagination is active)\n- match: optional regexp pattern to find within the tool result. Mutually exclusive with out__from_line. When present, out_line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return.\n- max_matches: max matches to return (default: 1). Has no effect without match.",
      promptSnippet: "Get details of a specific tool call from a past round",
      parameters: Type.Object({
        round: Type.String({ description: "The round filename to look up (e.g., 'abc123def456.json')" }),
        index: Type.Number({ description: "The 0-based index of the tool call within the round" }),
        out__from_line: Type.Optional(Type.Number({ description: "1-based line offset into the result (default: 1). Mutually exclusive with match." })),
        out_line_count: Type.Optional(Type.Number({ description: "Max lines to return (default: all, up to 200 when pagination is active, or 0 when match is specified)" })),
        match: Type.Optional(Type.String({ description: "A regexp pattern to search within the tool result text. Mutually exclusive with out__from_line. When provided, out_line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return." })),
        max_matches: Type.Optional(Type.Number({ description: "Max number of matches to return (default: 1). Has no effect without match." })),
      }),
      async execute(toolCallId, params, signal, onUpdate, ctx2) {
        const p = params as { round: string; index: number; out__from_line?: number; out_line_count?: number; match?: string; max_matches?: number };

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

        // Determine the result text (prefer result_full, fall back to result_summary)
        const resultText = tc.result_full ?? tc.result_summary ?? "";
        const useMatch = p.match !== undefined && p.match.length > 0;
        const useFromLine = p.out__from_line !== undefined;

        if (useMatch && useFromLine) {
          return {
            content: [{ type: "text", text: `Error: match and out__from_line are mutually exclusive. Use one or the other, not both.` }],
            details: {},
          };
        }

        if (useFromLine) {
          // Position-based pagination
          const allLines = resultText.split("\n");
          const totalLines = allLines.length;
          const fromLine = p.out__from_line ?? 1;
          const lineCount = p.out_line_count ?? 200;
          const startIdx = Math.max(0, fromLine - 1);
          const endIdx = Math.min(totalLines, startIdx + lineCount);
          const pageLines = allLines.slice(startIdx, endIdx);
          const remaining = totalLines - endIdx;

          const resultBlock = pageLines.length > 0
            ? pageLines.join("\n")
            : "(empty)";

          const footer = remaining > 0
            ? `\n\n[Truncated — lines remaining: ${remaining}. Use out__from_line=${endIdx + 1} and out_line_count=${lineCount} to continue.]`
            : "";

          return {
            content: [{
              type: "text",
              text: `[Showing lines ${fromLine}–${endIdx} of ${totalLines} for tool call #${tc.index} (${tc.name})]\n` +
                `Tool name: ${tc.name}\n` +
                `Arguments: ${JSON.stringify(argsParsed, null, 2)}\n` +
                `Result:\n` +
                `  ${resultBlock}${footer}`,
            }],
            details: { name: tc.name, arguments: tc.arguments, lines_shown: { from: fromLine, to: endIdx, of: totalLines } },
          };
        }

        if (useMatch) {
          // Regexp-based matching in tool result
          try {
            const regex = new RegExp(p.match!, "gm");
            const allLines = resultText.split("\n");
            const totalLines = allLines.length;
            const lineCount = p.out_line_count ?? 0;
            const maxMatches = p.max_matches ?? 1;

            // Find matching lines
            const matchResults: number[] = [];
            for (let i = 0; i < allLines.length; i++) {
              if (regex.test(allLines[i])) {
                matchResults.push(i);
                regex.lastIndex = 0;
              }
            }

            const totalMatches = matchResults.length;
            const shownIndices = matchResults.slice(0, maxMatches);
            const matchCount = shownIndices.length;

            // Build output with context windows
            const matchParts: string[] = [];
            for (let mi = 0; mi < matchCount; mi++) {
              const lineIdx = shownIndices[mi];
              const startIdx = lineIdx;
              const endIdx = Math.min(totalLines, startIdx + 1 + lineCount);
              const ctxLines = allLines.slice(startIdx, endIdx);
              const context = ctxLines.join("\n");
              if (lineCount > 0) {
                matchParts.push(`[M ${mi + 1}/${matchCount} at line ${lineIdx + 1} (${lineCount} lines of context)]\n${context}`);
              } else {
                matchParts.push(`[M ${mi + 1}/${matchCount} at line ${lineIdx + 1}] ${allLines[lineIdx]}`);
              }
            }

            const resultBlock = matchParts.length > 0
              ? matchParts.join("\n\n")
              : "(no matches)";

            let matchSummary = "";
            const remainingMatches = totalMatches - matchCount;
            if (remainingMatches > 0) {
              matchSummary = ` (${matchCount} match${matchCount !== 1 ? "es" : ""} shown of ${totalMatches} total)`;
            } else if (totalMatches > 0) {
              const mLabel = totalMatches === 1 ? "match" : "matches";
              matchSummary = ` (${totalMatches} ${mLabel})`;
            }

            return {
              content: [{
                type: "text",
                text: `[Match results for tool call #${tc.index} (${tc.name})${matchSummary}]\n` +
                  `Tool name: ${tc.name}\n` +
                  `Arguments: ${JSON.stringify(argsParsed, null, 2)}\n` +
                  `Result:\n` +
                  `  ${resultBlock}`,
              }],
              details: { name: tc.name, arguments: tc.arguments, matches: { shown: matchCount, total: totalMatches } },
            };
          } catch (err) {
            return {
              content: [{ type: "text", text: `Invalid regexp pattern: ${(err as Error).message}. Provide a valid JavaScript regexp string.` }],
              details: {},
            };
          }
        }

        // Unpaginated — show full result
        const resultBlock = resultText.length > 0
          ? `  ${resultText}`
          : "  (empty)";

        const truncatedFlag = tc.result_truncated
          ? "\n\n[Output exceeds storage cap — showing entire stored result]"
          : "";

        return {
          content: [{
            type: "text",
            text: `Tool call #${tc.index} in round ${p.round}\n` +
              `  Name: ${tc.name}\n` +
              `  Arguments: ${JSON.stringify(argsParsed, null, 2)}\n` +
              `  Result:\n${resultBlock}${truncatedFlag}`,
          }],
          details: { name: tc.name, arguments: tc.arguments, result_full: tc.result_full ?? tc.result_summary },
        };
      },
    });
  });


}
