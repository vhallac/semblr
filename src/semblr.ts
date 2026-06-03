/**
 * semblr — Retrieval-Augmented Context Assembly
 *
 * At agent_end: save the completed round to .pi/rounds/ and embed it.
 * At context: embed the current user prompt, query the vector index,
 *             inject the top-matching rounds as context for the LLM.
 *
 * Replaces flashback-amnesia.ts — no wiping, just smart retrieval.
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { ContextEvent, ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import {
	buildContextPreamble,
	buildGroupedRecencyList,
	buildRelevanceList,
	formatFileSize,
	splitCommandArgs,
} from "../lib/context-format.ts";
import { buildContextMessagePrefix, prepareContextMessages, shouldDropRelevanceList } from "../lib/context-messages.ts";
import { embedText, getApiKey } from "../lib/embedding-client.ts";
import { assignToGroup, formatGroupStats, parseGroupThreshold, type SemanticGroup } from "../lib/grouping.ts";
import { createRoundFilePath } from "../lib/hash.ts";
import {
	appendToIndexPath,
	buildSessionStartStatus,
	type IndexEntry,
	loadIndexFromPath as loadIndexFromPathCore,
	loadSessionStartIndex as loadSessionStartIndexCore,
} from "../lib/index-storage.ts";
import {
	applyMessageEndToState,
	buildAgentEndChainEntry,
	buildAgentEndEmbeddingTexts,
	buildAgentEndRoundData,
	extractAgentEndResponseText,
	extractAgentEndUserPrompt,
	getAgentEndParentId,
	getRelatedParentIdFromGroup,
	type MessageEndProcessingState,
} from "../lib/round-capture.ts";
import type { ChainEntry, ResponseSegment, RoundData, ToolCallDetail } from "../lib/round-data.ts";
import {
	loadRoundDataForToolDetails,
	renderRoundDetailsToolResult,
	renderToolDetailsToolResult,
	type ToolDetailsParams,
} from "../lib/round-tool-results.ts";
import {
	collectSearchRoundScores,
	computeContextBudget,
	filterSearchIndexByRounds,
	normalizeSearchInteractionsParams,
	renderSearchInteractionsToolResult,
	type SearchInteractionsParams,
	selectContextRounds,
} from "../lib/search-interactions.ts";
import {
	flushStatsFile,
	formatChainReadStatsReport,
	loadStatsFile,
	recordPresented,
	recordRead,
} from "../lib/stats.ts";
import { normalize } from "../lib/vector.ts";

export {
	buildContextMessagePrefix,
	countWordsInMessageContent,
	extractContextPrompt,
	prepareContextMessages,
	shouldDropRelevanceList,
	startsWithEnvironmentPreamble,
} from "../lib/context-messages.ts";
export { embedText, getApiKey } from "../lib/embedding-client.ts";
export { appendToIndexPath, buildSessionStartStatus, countUniqueIndexedRounds } from "../lib/index-storage.ts";
export type { MessageEndProcessingState } from "../lib/round-capture.ts";
export {
	applyMessageEndToState,
	buildAgentEndChainEntry,
	buildAgentEndEmbeddingTexts,
	buildAgentEndRoundData,
	buildAgentEndToolSummary,
	extractAgentEndResponseText,
	extractAgentEndUserPrompt,
	getAgentEndParentId,
	getRelatedParentIdFromGroup,
} from "../lib/round-capture.ts";
export type { RoundData } from "../lib/round-data.ts";
export {
	buildRoundAssistantOutput,
	collapseRoundDetails,
	formatRoundToolMeta,
	loadRoundDataForToolDetails,
	renderRoundDetailsToolResult,
	renderToolDetailsToolResult,
} from "../lib/round-tool-results.ts";
export {
	collectSearchRoundScores,
	computeContextBudget,
	filterSearchIndexByRounds,
	normalizeSearchInteractionsParams,
	renderSearchInteractionsToolResult,
	selectContextRounds,
} from "../lib/search-interactions.ts";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

// PI_CODING_AGENT_DIR overrides the default ~/.pi/agent config directory.
// We store semblr rounds under that directory so they survive project moves.
const PI_CONFIG_DIR = process.env.PI_CODING_AGENT_DIR || `${os.homedir()}/.pi/agent`;
const ROUNDS_DIR = `${PI_CONFIG_DIR}/semblr/rounds`;
const INDEX_PATH = `${ROUNDS_DIR}/index.csv`;
const SEMBLR_DIR = `${PI_CONFIG_DIR}/semblr`;
const STATS_PATH = `${SEMBLR_DIR}/chain-read-stats.json`;

// ◈ Causal-chain read statistics — global, never injected into context
//   Tracks all 5 causal-chain display positions (1-5, where 1 = most recent round).
//   Flushed atomically at agent_end. NOT reset on /new.
const TRACK_POSITIONS = 5; // hard-coded per user request — re-evaluate if readRate on any position > 50%
const statsState = loadStatsFile(STATS_PATH);
// Hashes presented at each display position (1-5) in the current context.
// Set in context hook, consumed by recordRead / recordPresented.
// Index 0 = display position 1 (most recent), index 4 = display position 5 (oldest).
const statsPresentedHashes: (string | null)[] = [null, null, null, null, null];

// Collapsed-only mode. All rounds are injected via the Recency and Relevance Lists.
// Use get_round_details() to expand. The Recency List contains the in-memory causal
// chain from the current session; the Relevance List contains semantically similar
// rounds from all past sessions.

// ─────────────────────────────────────────────
// Causal Chain — in-memory buffer of session rounds
// ─────────────────────────────────────────────

/** In-memory buffer of rounds from the current session, in chronological order.
 *  Survives between agent cycles within the same session. Cleared on session_start.
 *  Injected into context so the model can resolve "it", "those changes", "the fix", etc.
 *  without needing to search the vector index. */
let causalChain: ChainEntry[] = [];

// ─────────────────────────────────────────────
// Round Groups — centroid-based semantic grouping
// ─────────────────────────────────────────────

type RoundGroup = SemanticGroup<ChainEntry>;

/** In-memory groups of semantically similar rounds. Not persisted across sessions.
 *  Reset on session_start. Built incrementally at agent_end after each round save. */
let roundGroups: RoundGroup[] = [];

const SEMBLR_GROUP_THRESHOLD = parseGroupThreshold(process.env.SEMBLR_GROUP_THRESHOLD);

// Context formatting helpers live in lib/context-format.ts.

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** Stat a round file and return its formatted size string, or null on failure. */
function getRoundSize(fileName: string): string | null {
	try {
		const stat = fs.statSync(`${ROUNDS_DIR}/${fileName}`);
		return formatFileSize(stat.size);
	} catch {
		return null;
	}
}

// formatCollapsedIndex removed — replaced by buildGroupedRecencyList / buildRelevanceList / buildContextPreamble

// ─────────────────────────────────────────────
// Index CSV format:
//   base64url(vector_json),filePath
//   (no header row)
//   filePath includes :prompt, :response, or :round suffix
// ─────────────────────────────────────────────

export function loadIndexFromPath(
	indexPath: string = INDEX_PATH,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): IndexEntry[] {
	return loadIndexFromPathCore(indexPath, fsImpl);
}

function loadIndex(): IndexEntry[] {
	return loadIndexFromPathCore(INDEX_PATH);
}

export function loadSessionStartIndex(
	indexPath: string = INDEX_PATH,
	deps: { existsSync?: (filePath: string) => boolean; loadIndex?: () => IndexEntry[] } = {},
): IndexEntry[] {
	return loadSessionStartIndexCore(indexPath, deps);
}

export function readRoundFileFromDir(
	filePath: string,
	roundsDir: string = ROUNDS_DIR,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): RoundData | null {
	// filePath may be "xxx.json:prompt", "xxx.json:response", or "xxx.json:round"
	// strip the suffix to get the actual file
	const actualFile = filePath.replace(/(:prompt|:response|:round)$/, "");
	const fullPath = `${roundsDir}/${actualFile}`;
	if (!fsImpl.existsSync(fullPath)) return null;
	try {
		const data = JSON.parse(fsImpl.readFileSync(fullPath, "utf-8"));
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

function readRoundFile(filePath: string): RoundData | null {
	return readRoundFileFromDir(filePath, ROUNDS_DIR);
}

let lastRoundFileName: string | null = null; // tracks the most recent saved round (process-local)

// Per-agent accumulation (reset in agent_start, saved in agent_end)
let agentUserPrompt: string | null = null;
let agentTurnIndex: number | null = null;
let agentAccumulatedText: string[] = [];
let agentToolCallCount = 0;
let agentToolCallNames: string[] = [];
let agentToolCalls: ToolCallDetail[] = [];
const _agentPendingToolCallIds: Map<string, ToolCallDetail> = new Map(); // toolCallId → partial detail

// Interleaved response segments — preserves tool call positions within assistant text
let agentResponseSegments: ResponseSegment[] = [];

// Context embedding cache — avoids redundant embedding API calls across tool turns
// within the same agent cycle. Reset in agent_start.
let lastContextUserPrompt: string | null = null;
let lastContextVec: number[] = [];
let agentPromptVec: number[] | null = null; // cached from context hook, reused in agent_end to avoid redundant embed
let roundPresentedRecorded = false; // dedup: recordPresented only once per agent cycle

// Full-message cache — the envPreamble plus context-building-reference sections
// (recency list, relevance list, preamble) are computed once per agent cycle on
// the first context hook invocation and reused for subsequent tool turns.
// This fixes a prompt-cache-busting bug where new Date() timestamps in the
// envPreamble changed on every inner LLM call.
let cachedEnvPreamble: string | null = null;
let cachedContextMessages: unknown[] | null = null;
let cachedUserPromptForContext: string | null = null; // the extracted user prompt used to build cachedContextMessages

function appendToIndex(filePath: string, vector: number[]) {
	appendToIndexPath(INDEX_PATH, ROUNDS_DIR, filePath, vector);
}

function extensionRoot(): string {
	// src/semblr.ts -> project root. __dirname is available in pi's jiti runtime.
	return path.resolve(__dirname, "..");
}

export function getImportClaudeArgumentCompletions(prefix: string): Array<{ value: string; label: string }> | null {
	const options = ["--dry-run", "--include-sidechains", "--limit"];
	const matches = options.filter((opt) => opt.startsWith(prefix));
	return matches.length ? matches.map((value) => ({ value, label: value })) : null;
}

export function buildImportClaudeCommandPlan(args: string, root: string) {
	const parsedArgs = splitCommandArgs(args);
	return {
		root,
		script: path.resolve(root, "scripts", "import-claude-code.ts"),
		parsedArgs,
		statusText: `🧠 importing Claude Code history ${parsedArgs.join(" ")}`.trim(),
		startNotification: `Starting Claude Code import${parsedArgs.length ? ` (${parsedArgs.join(" ")})` : ""}...`,
	};
}

export function buildImportClaudeSpawnRequest(
	root: string,
	script: string,
	parsedArgs: readonly string[],
	apiKey: string | null,
	env: Record<string, string | undefined> = process.env,
) {
	const childEnv: Record<string, string | undefined> = {
		...env,
		...(apiKey ? { OPENROUTER_API_KEY: apiKey } : {}),
		SEMBLR_ROUNDS_DIR: ROUNDS_DIR,
	};
	return {
		command: "npx",
		args: ["tsx", script, ...parsedArgs],
		options: {
			cwd: root,
			env: childEnv,
			stdio: ["ignore", "pipe", "pipe"] as ["ignore", "pipe", "pipe"],
		},
	};
}

export function appendImportClaudeOutputTail(current: string, chunk: unknown, limit = 4000): string {
	const next = current + String(chunk);
	return next.length > limit ? next.slice(-limit) : next;
}

export function extractImportClaudeStatusLine(chunk: unknown): string | null {
	const lastLine = String(chunk).trim().split("\n").filter(Boolean).pop();
	return lastLine ? lastLine.slice(0, 120) : null;
}

export function combineImportClaudeOutput(stdout: string, stderr: string): string {
	return [stdout.trim(), stderr.trim()].filter(Boolean).join("\n");
}

export function buildImportClaudeCompletionReport(code: number | null, output: string) {
	const tail = output.split("\n").slice(-8).join("\n");
	if (code === 0) {
		return {
			statusText: "🧠 Claude Code import complete",
			notification: `Claude Code import complete${tail ? `:\n${tail}` : ""}`,
			level: "info" as const,
		};
	}
	return {
		statusText: `🧠 Claude Code import failed (${code})`,
		notification: `Claude Code import failed (${code})${tail ? `:\n${tail}` : ""}`,
		level: "error" as const,
	};
}

// ─────────────────────────────────────────────
// Extension
// ─────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	pi.registerCommand("semblr:import-claude", {
		description:
			"Import Claude Code history into Semblr (/semblr:import-claude --dry-run, --limit N, --include-sidechains)",
		getArgumentCompletions: getImportClaudeArgumentCompletions,
		handler: async (args, ctx) => {
			const root = extensionRoot();
			const { script, parsedArgs, statusText, startNotification } = buildImportClaudeCommandPlan(args, root);
			if (!fs.existsSync(script)) {
				ctx.ui.notify(`Semblr import script not found: ${script}`, "error");
				return;
			}

			ctx.ui.setStatus("semblr", statusText);
			ctx.ui.notify(startNotification, "info");

			const apiKey = await getApiKey(ctx as Parameters<typeof getApiKey>[0]);
			const spawnRequest = buildImportClaudeSpawnRequest(root, script, parsedArgs, apiKey);
			const child = spawn(spawnRequest.command, spawnRequest.args, spawnRequest.options);

			let stdout = "";
			let stderr = "";
			child.stdout.on("data", (chunk) => {
				stdout = appendImportClaudeOutputTail(stdout, chunk);
			});
			child.stderr.on("data", (chunk) => {
				stderr = appendImportClaudeOutputTail(stderr, chunk);
				const lastLine = extractImportClaudeStatusLine(chunk);
				if (lastLine) ctx.ui.setStatus("semblr", `🧠 ${lastLine}`);
			});

			const code = await new Promise<number | null>((resolve) => {
				child.on("close", resolve);
			});

			const output = combineImportClaudeOutput(stdout, stderr);
			const report = buildImportClaudeCompletionReport(code, output);
			ctx.ui.setStatus("semblr", report.statusText);
			ctx.ui.notify(report.notification, report.level);
		},
	});

	// ────────────────────────────────────────────
	// 1. context — assemble context from round repository
	// ────────────────────────────────────────────
	pi.on("context", async (event: ContextEvent, ctx: ExtensionContext) => {
		const { messages } = event;

		// --- Prepend environment info to the current user prompt ---
		// Computed once per agent cycle (cachedEnvPreamble) so the timestamp is
		// stable across tool turns — avoids busting the LLM prompt cache.
		const envPreamble =
			cachedEnvPreamble ??
			`[ENVIRONMENT]\nHost: ${os.hostname()}\nCWD: ${process.cwd()}\nCurrent date/time: ${new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+/, "")}`;
		cachedEnvPreamble = envPreamble; // pin early so guard below can use it

		// --- Extract system prompt + current round messages ---
		// We strip all prior rounds to prevent conversation bloat.
		// The retrieved historical context replaces the prior conversation.
		// Current round = everything from the last user message onward
		// (includes assistant responses, tool calls, tool results in-flight).
		// Also compute raw word count from the original un-augmented user prompt;
		// this is used below to skip the relevance list for very short prompts.
		const { systemMsg, augmentedMessages, currentMessages, hasUserMessage, userPrompt, rawPromptWordCount } =
			prepareContextMessages(messages, envPreamble);
		if (!hasUserMessage) return { messages } as any;
		if (userPrompt === null) return { messages: augmentedMessages } as any;

		// --- Reuse cached context blocks if this is a subsequent context call within
		//     the same agent cycle. The recency list, relevance list, and preamble
		//     are stable across tool turns; only the currentMessages section changes.
		if (cachedContextMessages && cachedEnvPreamble && userPrompt === cachedUserPromptForContext) {
			// Compose: system + preamble + recency + relevance + current turn messages
			const finalMessages = [...cachedContextMessages];
			finalMessages.push(...currentMessages);
			return { messages: finalMessages } as any;
		}

		// agentPromptVec is stashed after embedding below for agent_end to reuse

		try {
			const apiKey = await getApiKey(ctx);
			if (!apiKey) return { messages: augmentedMessages } as any;

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
			if (index.length === 0) {
				// Env preamble + current messages only (no context lists)
				cachedEnvPreamble = envPreamble;
				cachedUserPromptForContext = userPrompt;
				cachedContextMessages = systemMsg ? [systemMsg] : [];
				return { messages: augmentedMessages } as any;
			}

			const scoredRounds = collectSearchRoundScores(index, queryVec, readRoundFile);
			const bestScore = scoredRounds.length > 0 ? scoredRounds[0].bestScore : 0;

			// --- Dynamic budget ---
			const MIN_SIMILARITY = 0.3;
			const budgetTokens = computeContextBudget(
				bestScore,
				(ctx as ExtensionContext).model?.contextWindow ?? 128_000,
				MIN_SIMILARITY,
			);

			const selectedRounds = selectContextRounds(scoredRounds, lastRoundFileName, readRoundFile, {
				minSimilarity: MIN_SIMILARITY,
				budgetTokens,
			});

			if (selectedRounds.length === 0) {
				ctx.ui.setStatus("semblr", `🧠 no relevant context (best: ${bestScore.toFixed(3)})`);
				// Cache the empty-context result so subsequent turns reuse it
				cachedEnvPreamble = envPreamble;
				cachedUserPromptForContext = userPrompt;
				cachedContextMessages = systemMsg ? [systemMsg] : [];
				return { messages: augmentedMessages } as any;
			}

			// ── Build the three-section context block ──
			const dropRelevance = shouldDropRelevanceList(rawPromptWordCount);
			const relevanceList = dropRelevance
				? null
				: buildRelevanceList(
						selectedRounds.map((r) => ({ fileName: r.fileName, bestScore: r.bestScore, data: r.data })),
						getRoundSize,
					);
			const recencyList = buildGroupedRecencyList(roundGroups, causalChain, getRoundSize);
			const preamble = buildContextPreamble(!!recencyList, !!relevanceList);

			// ══ Stats: record all 5 positions presented ══
			{
				if (!roundPresentedRecorded) {
					recordPresented(statsState, statsPresentedHashes, causalChain, TRACK_POSITIONS);
					roundPresentedRecorded = true;
				}
				const uniqueRounds = new Set(
					index.map((e: { filePath: string }) => e.filePath.replace(/(:prompt|:response|:round)$/, "")),
				).size;
				ctx.ui.setStatus(
					"semblr",
					`🧠 collapsed: ${selectedRounds.length} matched / ${uniqueRounds} total | ${formatGroupStats(roundGroups, SEMBLR_GROUP_THRESHOLD)}`,
				);
			}

			const finalMessages = buildContextMessagePrefix(systemMsg, preamble, recencyList, relevanceList);

			// Cache the stable prefix (system + preamble + recency + relevance) for
			// subsequent context calls within this agent cycle. currentMessages is
			// appended fresh each time.
			cachedEnvPreamble = envPreamble;
			cachedUserPromptForContext = userPrompt;
			cachedContextMessages = [...finalMessages]; // snapshot before mutation below

			finalMessages.push(...currentMessages);

			return { messages: finalMessages } as any;
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
	pi.on("agent_start", async (_event, _ctx) => {
		// agent_start no longer carries messages/turnIndex in current pi API.
		// Hooks that need this info use agentPromptVec from context embedding
		// or before_agent_start prompt. turnIndex defaults to 0 in round data.
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

		// Reset full-message cache — env preamble and context blocks rebuilt on first context call
		cachedEnvPreamble = null;
		cachedContextMessages = null;
		cachedUserPromptForContext = null;
	});

	pi.on("message_end", async (event, _ctx) => {
		const state: MessageEndProcessingState = {
			accumulatedText: agentAccumulatedText,
			toolCallCount: agentToolCallCount,
			toolCallNames: agentToolCallNames,
			toolCalls: agentToolCalls,
			responseSegments: agentResponseSegments,
		};
		applyMessageEndToState(event.message, state);
		agentAccumulatedText = state.accumulatedText;
		agentToolCallCount = state.toolCallCount;
		agentToolCallNames = state.toolCallNames;
		agentToolCalls = state.toolCalls;
		agentResponseSegments = state.responseSegments;
	});

	pi.on("agent_end", async (event, ctx) => {
		const { messages } = event;

		// Get user prompt -- prefer agent_start cached value, fall back to messages
		const userPrompt = extractAgentEndUserPrompt(agentUserPrompt, messages);

		if (!userPrompt) {
			ctx.ui.setStatus("semblr", "\u{1f9e0} agent_end: no user prompt to save");
			return;
		}

		// Build response text from accumulated assistant text across all tool iterations
		const responseText = extractAgentEndResponseText(agentAccumulatedText, messages);

		if (!responseText) {
			ctx.ui.setStatus("semblr", "\u{1f9e0} agent_end: no response text");
			return;
		}

		fs.mkdirSync(ROUNDS_DIR, { recursive: true });

		const roundFileName = createRoundFilePath(userPrompt, responseText, agentToolCalls);
		const roundPath = `${ROUNDS_DIR}/${roundFileName}`;

		// Push to causal chain — even on dedup, this ensures the in-memory buffer
		// tracks every round seen in this session.
		causalChain.push(
			buildAgentEndChainEntry(roundFileName, userPrompt, responseText, agentToolCalls.length, agentToolCallNames),
		);

		// Skip if already saved (deduplication by content hash)
		if (fs.existsSync(roundPath)) {
			// Even on dedup, run grouping if the round has a combined embedding
			try {
				const existing = JSON.parse(fs.readFileSync(roundPath, "utf-8"));
				if (existing.promptEmbedding) {
					assignToGroup(
						roundGroups,
						causalChain[causalChain.length - 1],
						existing.promptEmbedding,
						SEMBLR_GROUP_THRESHOLD,
					);
				}
			} catch {
				/* best-effort */
			}
			ctx.ui.setStatus("semblr", `\u{1f9e0} round already saved (${roundFileName})`);
			lastRoundFileName = roundFileName;
			agentAccumulatedText = [];
			agentUserPrompt = null;
			agentTurnIndex = null;
			flushStatsFile(statsState, STATS_PATH, SEMBLR_DIR); // causal chain was pushed, so position scores may have changed
			return;
		}

		// Compute parentId from causalChain
		const parentId = getAgentEndParentId(causalChain);

		// Write round file
		const roundData = buildAgentEndRoundData({
			userPrompt,
			responseText,
			turnIndex: agentTurnIndex,
			toolCallCount: agentToolCallCount,
			toolCallNames: agentToolCallNames,
			toolCalls: agentToolCalls,
			responseSegments: agentResponseSegments,
			parentId,
		});

		try {
			fs.writeFileSync(roundPath, JSON.stringify(roundData, null, 2));
		} catch (err) {
			ctx.ui.setStatus("semblr", `\u{1f9e0} write error: ${(err as Error).message}`);
			agentAccumulatedText = [];
			agentUserPrompt = null;
			agentTurnIndex = null;
			return;
		}

		// Three embeddings for each round:
		//   1. prompt embedding → index.csv as :prompt
		//   2. response (truncated, REDACTED dropped) embedding → index.csv as :response
		//   3. concat(prompt, truncated+dropped response) embedding → round.json (for grouping)
		// text-embedding-3-small silently truncates at 8K tokens (~24KB), so we clip
		// the response explicitly to control what gets included.
		// Embedding a single concatenated text (rather than averaging separate prompt
		// and response vectors) preserves the semantic relationship between them.
		// See https://github.com/vhallac/semblr/issues/36
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
			// Strip context-injection REDACTED markers and clip to ~24KB (8K tokens)
			const { clippedResponse, combinedText } = buildAgentEndEmbeddingTexts(userPrompt, responseText);

			// Embedding #1: prompt (reuse cached agentPromptVec if available)
			const promptVec = agentPromptVec ?? normalize(await embedText(userPrompt, apiKey));

			// Embedding #2 + #3 in parallel
			const [responseVec, combinedVec] = await Promise.all([
				embedText(clippedResponse, apiKey),
				embedText(combinedText, apiKey),
			]);

			// Save to index: :prompt and :response (normalized for cosine similarity)
			appendToIndex(`${roundFileName}:prompt`, normalize(promptVec));
			appendToIndex(`${roundFileName}:response`, normalize(responseVec));

			// Update round.json with combined embedding
			const existing = JSON.parse(fs.readFileSync(roundPath, "utf-8"));
			existing.promptEmbedding = combinedVec;
			fs.writeFileSync(roundPath + ".tmp." + process.pid, JSON.stringify(existing, null, 2));
			fs.renameSync(roundPath + ".tmp." + process.pid, roundPath);

			ctx.ui.setStatus("semblr", `\u{1f9e0} saved + embedded round (${roundFileName})`);

			// — Grouping + metadata update —
			// Assign to a semantic group using the combined embedding.
			const roundEntry = causalChain[causalChain.length - 1];
			const groupIdx = assignToGroup(roundGroups, roundEntry, combinedVec, SEMBLR_GROUP_THRESHOLD);
			const group = roundGroups[groupIdx];
			// Find the most recent round in the same group *before* this round
			const relatedParentId = getRelatedParentIdFromGroup(group, roundEntry);
			if (relatedParentId) {
				// Re-read, update, re-write atomically
				try {
					const updated = JSON.parse(fs.readFileSync(roundPath, "utf-8"));
					updated.relatedParentId = relatedParentId;
					fs.writeFileSync(roundPath + ".tmp." + process.pid, JSON.stringify(updated, null, 2));
					fs.renameSync(roundPath + ".tmp." + process.pid, roundPath);
				} catch {
					/* best-effort */
				}
			}
		} catch (err) {
			ctx.ui.setStatus("semblr", `\u{1f9e0} embedding error: ${(err as Error).message}`);
		}

		lastRoundFileName = roundFileName;
		agentAccumulatedText = [];
		agentUserPrompt = null;
		agentTurnIndex = null;

		// ◈ Flush stats to disk (atomically) and show in TUI
		flushStatsFile(statsState, STATS_PATH, SEMBLR_DIR);

		// Combine chain-read stats with total indexed rounds count
		const indexExists = fs.existsSync(INDEX_PATH);
		const idx = indexExists ? loadIndex() : [];
		const totalRounds = new Set(
			idx.map((e: { filePath: string }) => e.filePath.replace(/(:prompt|:response|:round)$/, "")),
		).size;
		ctx.ui.setStatus(
			"semblr",
			`🧠 ${totalRounds} total indexed | ${formatGroupStats(roundGroups, SEMBLR_GROUP_THRESHOLD)}`,
		);
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
		// Clear causal chain and round groups — new session starts fresh
		causalChain = [];
		roundGroups = [];

		const index = loadSessionStartIndex();
		ctx.ui.setStatus("semblr", buildSessionStartStatus(index));

		// Register the search_interactions tool here, not at factory level
		pi.registerTool({
			name: "search_interactions",
			label: "Search Interactions",
			description:
				"Search all past user interactions for topics, questions, or discussions. Unlike the built-in search_memory (which searches within the current session), this searches across ALL sessions the user has ever had — every conversation round ever indexed. Use this when you need to find something from a past session, recall prior discussions, or reconnect with knowledge that was established a long time ago.\n\nYou can optionally scope the search to specific round files by passing the `turns` parameter. This is useful when you want to drill down into a specific subset of rounds.",
			promptSnippet: "Search past interactions for relevant context",
			parameters: Type.Object({
				query: Type.String({ description: "The search query — what you want to find in past conversations" }),
				minSimilarity: Type.Optional(
					Type.Number({
						description: "Minimum similarity threshold (0.0 to 1.0). Default 0.25. Lower to get broader matches.",
					}),
				),
				rounds: Type.Optional(
					Type.Array(Type.String(), {
						description:
							"Optional list of round filenames to scope the search to (e.g., ['abc.json', 'def.json']). When provided, only these round files are searched.",
					}),
				),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx2) {
				const { query, threshold, scopeRounds } = normalizeSearchInteractionsParams(
					params as SearchInteractionsParams,
				);
				if (!query) {
					return {
						content: [{ type: "text", text: "No query provided." }],
						details: {},
					};
				}

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
				const index = loadIndex();
				if (index.length === 0) {
					return {
						content: [{ type: "text", text: "The round index is empty. No conversations have been saved yet." }],
						details: {},
					};
				}

				// If rounds[] is provided, scope the search to only those round files
				const scopedIndex = filterSearchIndexByRounds(index, scopeRounds);
				if (scopeRounds && scopeRounds.length > 0 && scopedIndex.length === 0) {
					return {
						content: [
							{
								type: "text",
								text: `No indexed vectors found for the specified rounds: ${scopeRounds.join(", ")}. They may not be embedded yet.`,
							},
						],
						details: {},
					};
				}

				const sorted = collectSearchRoundScores(scopedIndex, queryVec, readRoundFile);
				return renderSearchInteractionsToolResult(sorted, threshold, getRoundSize);
			},
		});

		// Register get_round_details — retrieve the full content of a round file
		pi.registerTool({
			name: "get_round_details",
			label: "Get Round Details",
			description:
				"Retrieve the full content of a past conversation round by its filename hash. Unlike the truncated previews injected into context (which show only the first portion of the user prompt and assistant response), this returns the complete userPrompt and responseSequence for that round, plus all tool call metadata. Use this when you need to see the full conversation from a historical round.\n\nParameters:\n- round: the round filename (e.g., 'abc123.json')\n- from_line: optional 1-based line offset into the assistant response (default: 1). When specified, the response sequence is paginated on line boundaries. Mutually exclusive with match.\n- line_count: optional max lines of assistant response to return. Default: 200 when from_line is specified, 0 when match is specified (just the matched line). Omit both for full response.\n- match: optional regexp pattern to find within the round (user prompt + assistant response). Mutually exclusive with from_line. When present, line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return.\n- max_matches: max matches to return (default: 1). Has no effect without match.",
			promptSnippet: "Get full details of a past conversation round",
			parameters: Type.Object({
				round: Type.String({ description: "The round filename to look up (e.g., 'abc123def456.json')" }),
				from_line: Type.Optional(
					Type.Number({
						description:
							"1-based line offset into the assistant response. Default: 1 (start). When specified, the response is paginated on line boundaries. Mutually exclusive with match.",
					}),
				),
				line_count: Type.Optional(
					Type.Number({
						description:
							"Max lines of assistant response to return. Default: 200 when from_line is specified, 0 when match is specified. Omit both for full response.",
					}),
				),
				match: Type.Optional(
					Type.String({
						description:
							"A regexp pattern to search within the round (user prompt + assistant response). Mutually exclusive with from_line. When provided, line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return.",
					}),
				),
				max_matches: Type.Optional(
					Type.Number({
						description: "Max number of matches to return (default: 1). Has no effect without match.",
					}),
				),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const p = params as {
					round: string;
					from_line?: number;
					line_count?: number;
					match?: string;
					max_matches?: number;
				};

				// ◈ Stats: check if this hash matches any presented position
				recordRead(statsState, statsPresentedHashes, p.round, TRACK_POSITIONS);

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

				return renderRoundDetailsToolResult(p, roundData);
			},
		});

		// Register get_tool_details — retrieve full tool call details from a round file
		pi.registerTool({
			name: "get_tool_details",
			label: "Get Tool Details",
			description:
				"Retrieve the full arguments and result of a specific tool call from a past conversation round. When you see historical rounds injected into context with TOOLS USED markers, you can call this tool to inspect what a specific tool did — its full arguments and the complete output (not just the preview stored in the round file).\n\nParameters:\n- round: the round filename (e.g., 'abc123.json')\n- index: the 0-based index of the tool call within that round's toolCalls array\n- out__from_line: optional 1-based line offset into the result (default: 1). Mutually exclusive with match.\n- out_line_count: optional max lines to return (default: all, up to 200 when pagination is active)\n- match: optional regexp pattern to find within the tool result. Mutually exclusive with out__from_line. When present, out_line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return.\n- max_matches: max matches to return (default: 1). Has no effect without match.",
			promptSnippet: "Get details of a specific tool call from a past round",
			parameters: Type.Object({
				round: Type.String({ description: "The round filename to look up (e.g., 'abc123def456.json')" }),
				index: Type.Number({ description: "The 0-based index of the tool call within the round" }),
				out__from_line: Type.Optional(
					Type.Number({
						description: "1-based line offset into the result (default: 1). Mutually exclusive with match.",
					}),
				),
				out_line_count: Type.Optional(
					Type.Number({
						description:
							"Max lines to return (default: all, up to 200 when pagination is active, or 0 when match is specified)",
					}),
				),
				match: Type.Optional(
					Type.String({
						description:
							"A regexp pattern to search within the tool result text. Mutually exclusive with out__from_line. When provided, out_line_count specifies lines of context after each match (default: 0). Use with max_matches to control how many matches to return.",
					}),
				),
				max_matches: Type.Optional(
					Type.Number({
						description: "Max number of matches to return (default: 1). Has no effect without match.",
					}),
				),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const p = params as ToolDetailsParams;
				const loaded = loadRoundDataForToolDetails(`${ROUNDS_DIR}/${p.round}`, p.round);
				if (!loaded.ok) return loaded.result;
				return renderToolDetailsToolResult(p, loaded.roundData);
			},
		});

		// Register the /semblr-recent-read-stats command
		pi.registerCommand("semblr:recent-read-stats", {
			description: "Display chain-read statistics (how often the agent read rounds from each display position)",
			handler: async (_args, commandCtx) => {
				const report = formatChainReadStatsReport(statsState, TRACK_POSITIONS);
				commandCtx.ui.notify(report, "info");
			},
		});
	});
}
