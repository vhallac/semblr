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
	buildCheckpointSectionContent,
	buildContextPreamble,
	buildFinalResponseContract,
	buildFollowUpSectionContent,
	buildGroupedRecencyList,
	buildRelevanceList,
	buildRoutingInstructions,
	buildSessionArchitecture,
	buildWorkingMemorySection,
	splitCommandArgs,
} from "../lib/context-format.ts";
import {
	assembleContextPrefix,
	prepareContextMessages,
	shouldDropEmbedding,
	shouldDropRelevanceList,
} from "../lib/context-messages.ts";
import { embedText, getApiKey } from "../lib/embedding-client.ts";
import { assignToGroup, formatGroupStats } from "../lib/grouping.ts";
import { createRoundFilePath } from "../lib/hash.ts";
import { indexRoundFileFromPath } from "../lib/index-io.ts";
import {
	appendToIndexPath,
	buildSessionStartStatus,
	type IndexEntry,
	loadIndexFromPath as loadIndexFromPathCore,
	loadSessionStartIndex as loadSessionStartIndexCore,
} from "../lib/index-storage.ts";
import { resolveModelId } from "../lib/resolve-model-id.ts";
import {
	applyMessageEndToState,
	buildAgentEndChainEntry,
	buildAgentEndEmbeddingTexts,
	buildAgentEndRoundData,
	embeddingMaxTokensToResponseBytes,
	extractAgentEndResponseText,
	extractAgentEndUserPrompt,
	extractAndStripFollowupMarker,
	getAgentEndParentId,
	getRelatedParentIdFromGroup,
	type MessageEndProcessingState,
} from "../lib/round-capture.ts";
import type { RoundData } from "../lib/round-data.ts";
import { getRoundFileSize, readRoundFileFromDir as readRoundFileFromDirLib, readRoundJson } from "../lib/round-io.ts";
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
import { loadSemblrConfig, type PhaseName, type SemblrConfig } from "../lib/semblr-config.ts";
import type { CheckpointSummary, ToolCallDetail } from "../lib/state.ts";
import {
	contextCacheSnapshot,
	contextCacheStore,
	contextCacheValid,
	createRound,
	createSession,
} from "../lib/state.ts";
import {
	flushStatsFile,
	formatChainReadStatsReport,
	loadStatsFile,
	recordPresented,
	recordRead,
} from "../lib/stats.ts";
import { estimateMessagesTokens } from "../lib/tokens.ts";
import { normalize } from "../lib/vector.ts";
import {
	addSlot,
	deleteSlot,
	formatMiniMemSlot,
	getAndDeleteSlot,
	getSlot,
	updateSlot,
} from "../lib/working-memory.ts";

export {
	assembleContextPrefix,
	countWordsInMessageContent,
	extractContextPrompt,
	prepareContextMessages,
	shouldDropEmbedding,
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
export { readRoundFileFromDir } from "../lib/round-io.ts";
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

const SEMBLR_CONFIG = loadSemblrConfig();
const ROUNDS_DIR = SEMBLR_CONFIG.roundsDir;
const INDEX_PATH = SEMBLR_CONFIG.indexPath;
const SEMBLR_DIR = path.dirname(ROUNDS_DIR);
const STATS_PATH = path.join(SEMBLR_DIR, "chain-read-stats.json");
const EMBEDDING_RESPONSE_MAX_BYTES = embeddingMaxTokensToResponseBytes(SEMBLR_CONFIG.embeddingMaxTokens);
const ROUTING_INSTRUCTIONS = SEMBLR_CONFIG.routing.enabled ? buildRoutingInstructions() : null;

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
// Session — state that survives between rounds within the same session.
// Reset at session_start.
// ─────────────────────────────────────────────

let session = createSession();
let round = createRound();

const SEMBLR_GROUP_THRESHOLD = SEMBLR_CONFIG.groupThreshold;

/** Build a flat text representation of a checkpoint summary for embedding. */
function buildCheckpointSummaryText(summary: CheckpointSummary): string {
	const lines: string[] = [];
	lines.push(`Current Task: ${summary.currentTask}`);
	if (summary.progressMade.length > 0) {
		lines.push("Progress Made:");
		for (const item of summary.progressMade) lines.push(`- ${item}`);
	}
	if (summary.currentState.length > 0) {
		lines.push("Current State:");
		for (const item of summary.currentState) lines.push(`- ${item}`);
	}
	if (summary.nextSteps.length > 0) {
		lines.push("Next Steps:");
		for (const item of summary.nextSteps) lines.push(`- ${item}`);
	}
	if (summary.keyFindings.length > 0) {
		lines.push("Key Findings / Decisions:");
		for (const item of summary.keyFindings) lines.push(`- ${item}`);
	}
	return lines.join("\n");
}

// Context formatting helpers live in lib/context-format.ts.

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** Stat a round file and return its formatted size string, or null on failure. */
function getRoundSize(fileName: string): string | null {
	return getRoundFileSize(ROUNDS_DIR, fileName);
}

// formatCollapsedIndex removed — replaced by buildGroupedRecencyList / buildRelevanceList / buildContextPreamble

/**
 * Post-embedding tail: update round.json with the embedding vector,
 * run semantic grouping, and optionally backfill relatedParentId.
 * Shared by both the skip-prompt-embedding and full-embedding paths
 * in agent_end. Only the embedding vector and status message differ.
 */
function finalizeRoundEmbedding({
	roundPath,
	embeddingVec,
	needsFollowup,
}: {
	roundPath: string;
	embeddingVec: number[];
	needsFollowup: boolean;
}): number {
	// Atomic update of round.json with the embedding vector
	const existing = JSON.parse(fs.readFileSync(roundPath, "utf-8"));
	existing.promptEmbedding = embeddingVec;
	fs.writeFileSync(roundPath + ".tmp." + process.pid, JSON.stringify(existing, null, 2));
	fs.renameSync(roundPath + ".tmp." + process.pid, roundPath);

	// Assign to a semantic group.
	// If this round is tagged needsFollowup, remember the group index so
	// the next round is auto-assigned to the same group without semantic matching.
	const roundEntry = session.causalChain[session.causalChain.length - 1];
	const groupIdx = assignToGroup(
		session.roundGroups,
		roundEntry,
		embeddingVec,
		SEMBLR_GROUP_THRESHOLD,
		needsFollowup ? session.lastFollowupGroupIdx : null,
	);
	if (needsFollowup) {
		session.lastFollowupGroupIdx = groupIdx;
	}

	// Find the most recent round in the same group *before* this round
	const group = session.roundGroups[groupIdx];
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

	return groupIdx;
}

/**
 * Check if the last round has `needsFollowup: true`. If so, build a
 * follow-up section and clear the flag atomically.
 * Returns null if no follow-up is needed.
 */
function buildFollowUpContext(fileName: string): string | null {
	const round = readRoundJson(ROUNDS_DIR, fileName);
	if (!round?.needsFollowup) return null;
	return buildFollowUpSectionContent(
		fileName,
		(round.userPrompt as string) ?? "",
		(round.responseSequence as string) ?? "",
	);
}

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

function readRoundFile(filePath: string): RoundData | null {
	return readRoundFileFromDirLib(filePath, ROUNDS_DIR);
}

let lastRoundFileName: string | null = null; // tracks the most recent saved round (process-local)
// Used in context hook to gate follow-up injection: checks metadata + in-memory state
function needsFollowupInjection(fileName: string): boolean {
	const round = readRoundJson(ROUNDS_DIR, fileName);
	return round?.needsFollowup === true && !session.injectedFollowupRounds.has(fileName);
}

/** Check if the last round has a checkpoint summary that hasn't been injected yet. */
function needsCheckpointInjection(fileName: string): boolean {
	const round = readRoundJson(ROUNDS_DIR, fileName);
	return round?.summary != null && !session.injectedCheckpointRounds.has(fileName);
}

/** Build the checkpoint injection section for context. */
function buildCheckpointContext(fileName: string): string | null {
	const round = readRoundJson(ROUNDS_DIR, fileName);
	if (!round?.summary) return null;
	return buildCheckpointSectionContent(fileName, round.summary as CheckpointSummary);
}

// Per-agent accumulation state now lives in `round` object (createRound()).
// Reset at each agent_start.

// Thread-local pending tool call IDs — cleared per tool call, not per round
const _agentPendingToolCallIds: Map<string, ToolCallDetail> = new Map(); // toolCallId → partial detail

function appendToIndex(filePath: string, vector: number[], model?: string) {
	appendToIndexPath(INDEX_PATH, ROUNDS_DIR, filePath, vector, {}, model);
}

function embeddingClientDeps(ctx: ExtensionContext) {
	return { config: SEMBLR_CONFIG, modelRegistry: ctx.modelRegistry };
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
	config: SemblrConfig = SEMBLR_CONFIG,
) {
	const childEnv: Record<string, string | undefined> = {
		...env,
		...(apiKey ? { OPENROUTER_API_KEY: apiKey } : {}),
		SEMBLR_ROUNDS_DIR: config.roundsDir,
		SEMBLR_EMBEDDING_PROVIDER: config.embeddingProvider,
		SEMBLR_EMBEDDING_MODEL: config.embeddingModel,
		SEMBLR_EMBEDDING_MAX_TOKENS: String(config.embeddingMaxTokens),
		...(config.embeddingApiUrl ? { SEMBLR_EMBEDDING_API_URL: config.embeddingApiUrl } : {}),
		SEMBLR_GROUP_THRESHOLD: String(config.groupThreshold),
		SEMBLR_MIN_SIMILARITY: String(config.minSimilarity),
		SEMBLR_EMBED_TIMEOUT: String(config.embedTimeoutMs),
		SEMBLR_EMBED_RETRIES: String(config.embedMaxRetries),
		SEMBLR_EMBED_BACKOFF: String(config.embedBackoffMs),
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

			const apiKey = await getApiKey(ctx, { config: SEMBLR_CONFIG });
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

	/** Compute the context-size warning level based on non-system tokens.
	 *  Returns 0 (no warning), 1 (70% threshold), 2 (85%), or 3 (100%+).
	 *  The warning is appended after currentMessages (last thing the LLM
	 *  sees, minimal KV-cache disruption).
	 *
	 *  Cache coherence: the prefix is snapshotted WITHOUT the warning (to
	 *  keep the KV-cache hit across tool turns). On every call, the warning
	 *  is ALWAYS re-appended when level > 0 — NOT just on escalation. This
	 *  guarantees the warning never silently drops when the cache is reused.
	 *  See https://github.com/vhallac/semblr/issues/72 */
	function applyContextSizeWarning(
		prefixMessages: unknown[],
		currentMsgs: unknown[],
		systemMsg: unknown | null,
	): unknown[] {
		const threshold = SEMBLR_CONFIG.summaryThresholdExtra;
		if (threshold <= 0) return [...prefixMessages, ...currentMsgs];

		// Measure all messages including current (this is what the LLM will see)
		const allMessages = [...prefixMessages, ...currentMsgs];
		const totalTokens = estimateMessagesTokens(allMessages);

		// Subtract system message tokens — user wants to ignore system prompt
		const systemTokens = systemMsg ? estimateMessagesTokens([systemMsg]) : 0;
		const nonSystemTokens = totalTokens - systemTokens;

		// Compute warning level
		let newLevel = 0;
		if (nonSystemTokens >= threshold) {
			newLevel = 3;
		} else if (nonSystemTokens >= threshold * 0.85) {
			newLevel = 2;
		} else if (nonSystemTokens >= threshold * 0.7) {
			newLevel = 1;
		}

		// Update the tracked level (so we know we've warned at this level).
		// Even if the level hasn't escalated, we re-inject the warning below —
		// the cache snapshot lacks the warning, so dropping it would be a bug.
		round.contextWarningIssued = newLevel;

		// Always inject when level > 0 — not just on escalation.
		// Without this, the warning silently disappears on subsequent tool turns
		// because the cached prefix was snapshotted before the warning was appended.
		if (newLevel === 0) return allMessages;

		// Build the warning message
		const levelLabel = newLevel === 3 ? "3 — IMMEDIATE" : String(newLevel);
		const urgency =
			newLevel === 3
				? "You MUST stop IMMEDIATELY. Do not make any further tool calls except `semblr_checkpoint`. Call it now with your progress summary, then stop."
				: newLevel === 2
					? "You MUST wrap up your current work after this round. Before finishing, call the `semblr_checkpoint` tool with a summary of your progress. Then stop — do not start new work."
					: "You SHOULD wrap up your current work after this round. Before finishing, call the `semblr_checkpoint` tool with a summary of your progress. Then stop — do not start new work.";

		const warningText =
			`[CONTEXT SIZE WARNING — LEVEL ${levelLabel}]

` +
			`Your context has grown to ${nonSystemTokens} non-system tokens (threshold: ${threshold}). Context size ${newLevel >= 3 ? "has exceeded" : "is approaching"} the configured limit.

` +
			`${urgency}

` +
			`The semblr_checkpoint tool parameters are: currentTask (string), progressMade (string[]), currentState (string[]), nextSteps (string[]), keyFindings (string[]).`;

		// Append warning message at the end — after current messages (including tool results).
		// This positions the warning as the LAST thing the LLM sees, giving it full
		// attention and minimising KV-cache disruption (only the tail shifts).
		const warningMsg = {
			role: "user" as const,
			content: [{ type: "text" as const, text: warningText }],
		};

		return [...prefixMessages, ...currentMsgs, warningMsg];
	}
	/**
	 * Resolve checkpoint + follow-up injection for the last round.
	 * Returns {followUpMsg, checkpointMsg} for the context prefix.
	 * When a checkpoint is present, follow-up is suppressed (checkpoint
	 * already carries sufficient state for the agent to resume work).
	 *
	 * This was duplicated verbatim (~30 lines) between the short-prompt fast
	 * path and the full embedding path before the #1 refactoring.
	 */
	function resolveCompoundInjections(): {
		followUpMsg: unknown | null;
		checkpointMsg: unknown | null;
	} {
		let checkpointMsg: unknown | null = null;
		let hasCheckpoint = false;
		if (lastRoundFileName && needsCheckpointInjection(lastRoundFileName)) {
			const checkpointSection = buildCheckpointContext(lastRoundFileName);
			if (checkpointSection) {
				checkpointMsg = {
					role: "user" as const,
					content: [{ type: "text" as const, text: checkpointSection }],
				};
				session.injectedCheckpointRounds.add(lastRoundFileName);
				hasCheckpoint = true;
			}
		}

		let followUpMsg: unknown | null = null;
		if (!hasCheckpoint && lastRoundFileName && needsFollowupInjection(lastRoundFileName)) {
			const followUpSection = buildFollowUpContext(lastRoundFileName);
			if (followUpSection) {
				followUpMsg = {
					role: "user" as const,
					content: [{ type: "text" as const, text: followUpSection }],
				};
				session.injectedFollowupRounds.add(lastRoundFileName);
			}
		}

		return { followUpMsg, checkpointMsg };
	}

	pi.on("context", async (event: ContextEvent, ctx: ExtensionContext) => {
		const { messages } = event;

		// --- Prepend environment info to the current user prompt ---
		// Computed once per agent cycle (round.contextCache.envPreamble) so the timestamp is
		// stable across tool turns — avoids busting the LLM prompt cache.
		const envPreamble =
			round.contextCache.envPreamble ??
			`[ENVIRONMENT]\nHost: ${os.hostname()}\nCWD: ${process.cwd()}\nCurrent date/time: ${new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+/, "")}`;
		round.contextCache.envPreamble = envPreamble; // pin early so guard below can use it

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
		if (contextCacheValid(round.contextCache, userPrompt)) {
			// Compose: system + preamble + recency + relevance + current turn messages
			const finalMessages = applyContextSizeWarning(round.contextCache.messages!, currentMessages, systemMsg);
			return { messages: finalMessages } as any;
		}

		// --- Short-prompt fast path: skip embedding and retrieval but keep ───
		//     recency list, follow-up injection, and preamble (all zero API cost).
		// Short prompts ("yes", "do it", "continue") produce noisy embeddings.
		if (shouldDropEmbedding(rawPromptWordCount)) {
			// Short prompt: skip prompt embedding (noisy) but still embed response (significant).
			// Per https://github.com/vhallac/semblr/issues/38#issuecomment-4629826478
			round.skipPromptEmbedding = true;
			round.promptVec = null;
			round.contextCache.envPreamble = envPreamble;
			round.contextCache.userPrompt = userPrompt;

			// Build non-embedding context sections (all in-memory / disk, zero API cost)
			const recencyList = buildGroupedRecencyList(session.roundGroups, session.causalChain, getRoundSize);
			const preamble = buildContextPreamble(!!recencyList, false);

			const { followUpMsg, checkpointMsg } = resolveCompoundInjections();

			const workingMem = buildWorkingMemorySection(session.miniMemStore);

			const sessionArchitecture = buildSessionArchitecture();

			const contractMsg = {
				role: "user" as const,
				content: [{ type: "text" as const, text: buildFinalResponseContract() }],
			};

			// Assemble context prefix via assembleContextPrefix — single call, consistent order
			const prefixMsgs = assembleContextPrefix({
				systemMsg,
				sessionArchitecture,
				workingMemory: workingMem,
				routingInstructions: ROUTING_INSTRUCTIONS,
				preamble,
				recencyList,
				relevanceList: null,
				followUpMsg,
				checkpointMsg,
				contractMsg,
			});
			contextCacheSnapshot(round.contextCache, [...prefixMsgs]);

			// ══ Stats: record all 5 positions presented ══
			if (!round.presentedRecorded) {
				recordPresented(statsState, statsPresentedHashes, session.causalChain, TRACK_POSITIONS);
				round.presentedRecorded = true;
			}

			const finalMessages: unknown[] = applyContextSizeWarning(
				round.contextCache.messages!,
				currentMessages,
				systemMsg,
			);
			return { messages: finalMessages } as any;
		}

		// round.promptVec is stashed after embedding below for agent_end to reuse

		try {
			const apiKey = await getApiKey(ctx, { config: SEMBLR_CONFIG });
			if (!apiKey) {
				const prefixMsgs = assembleContextPrefix({
					systemMsg,
					sessionArchitecture: buildSessionArchitecture(),
					workingMemory: buildWorkingMemorySection(session.miniMemStore),
					routingInstructions: ROUTING_INSTRUCTIONS,
					preamble: null,
					recencyList: null,
					relevanceList: null,
					followUpMsg: null,
					checkpointMsg: null,
					contractMsg: {
						role: "user" as const,
						content: [{ type: "text" as const, text: buildFinalResponseContract() }],
					},
				});
				const finalMessages: unknown[] = applyContextSizeWarning(prefixMsgs, currentMessages, systemMsg);
				return { messages: finalMessages } as any;
			}

			// Embed the user prompt — cached per agent cycle to avoid redundant API calls
			// across multiple tool turns within the same user prompt.
			let queryVec: number[];
			if (userPrompt === round.lastContextUserPrompt) {
				queryVec = round.lastContextVec;
			} else {
				queryVec = normalize(await embedText(userPrompt, apiKey, embeddingClientDeps(ctx)));
				round.lastContextUserPrompt = userPrompt;
				round.lastContextVec = queryVec;
			}
			// Stash for agent_end to reuse (saves 1 embedding call per round)
			round.promptVec = queryVec;

			// Load and score the index
			const index = loadIndex();
			if (index.length === 0) {
				// Final response contract + current messages (no context lists)
				round.contextCache.envPreamble = envPreamble;
				round.contextCache.userPrompt = userPrompt;
				const emptyIdxMsgs = assembleContextPrefix({
					systemMsg,
					sessionArchitecture: buildSessionArchitecture(),
					workingMemory: buildWorkingMemorySection(session.miniMemStore),
					routingInstructions: ROUTING_INSTRUCTIONS,
					preamble: null,
					recencyList: null,
					relevanceList: null,
					followUpMsg: null,
					checkpointMsg: null,
					contractMsg: {
						role: "user" as const,
						content: [{ type: "text" as const, text: buildFinalResponseContract() }],
					},
				});
				contextCacheSnapshot(round.contextCache, emptyIdxMsgs);
				const finalMessages: unknown[] = applyContextSizeWarning(
					round.contextCache.messages!,
					currentMessages,
					systemMsg,
				);
				return { messages: finalMessages } as any;
			}

			const scoredRounds = collectSearchRoundScores(index, queryVec, readRoundFile);
			const bestScore = scoredRounds.length > 0 ? scoredRounds[0].bestScore : 0;

			// --- Dynamic budget ---
			const budgetTokens = computeContextBudget(
				bestScore,
				ctx.model?.contextWindow ?? 128_000,
				SEMBLR_CONFIG.minSimilarity,
			);

			const selectedRounds = selectContextRounds(scoredRounds, lastRoundFileName, readRoundFile, {
				minSimilarity: SEMBLR_CONFIG.minSimilarity,
				budgetTokens,
			});

			if (selectedRounds.length === 0) {
				ctx.ui.setStatus("semblr", `🧠 no relevant context (best: ${bestScore.toFixed(3)})`);
				// Cache the empty-context result so subsequent turns reuse it
				round.contextCache.envPreamble = envPreamble;
				round.contextCache.userPrompt = userPrompt;
				const zeroResultMsgs = assembleContextPrefix({
					systemMsg,
					sessionArchitecture: buildSessionArchitecture(),
					workingMemory: buildWorkingMemorySection(session.miniMemStore),
					routingInstructions: ROUTING_INSTRUCTIONS,
					preamble: null,
					recencyList: null,
					relevanceList: null,
					followUpMsg: null,
					checkpointMsg: null,
					contractMsg: {
						role: "user" as const,
						content: [{ type: "text" as const, text: buildFinalResponseContract() }],
					},
				});
				contextCacheSnapshot(round.contextCache, zeroResultMsgs);
				const finalMessages: unknown[] = applyContextSizeWarning(
					round.contextCache.messages!,
					currentMessages,
					systemMsg,
				);
				return { messages: finalMessages } as any;
			}

			// ── Build the three-section context block ──
			const dropRelevance = shouldDropRelevanceList(rawPromptWordCount);
			const relevanceList = dropRelevance
				? null
				: buildRelevanceList(
						selectedRounds.map((r) => ({ fileName: r.fileName, bestScore: r.bestScore, data: r.data })),
						getRoundSize,
					);
			const recencyList = buildGroupedRecencyList(session.roundGroups, session.causalChain, getRoundSize);
			const preamble = buildContextPreamble(!!recencyList, !!relevanceList);

			// ══ Stats: record all 5 positions presented ══
			{
				if (!round.presentedRecorded) {
					recordPresented(statsState, statsPresentedHashes, session.causalChain, TRACK_POSITIONS);
					round.presentedRecorded = true;
				}
				const uniqueRounds = new Set(index.map((e: { filePath: string }) => indexRoundFileFromPath(e.filePath)))
					.size;
				ctx.ui.setStatus(
					"semblr",
					`🧠 collapsed: ${selectedRounds.length} matched / ${uniqueRounds} total | ${formatGroupStats(session.roundGroups, SEMBLR_GROUP_THRESHOLD)}`,
				);
			}

			const { followUpMsg, checkpointMsg } = resolveCompoundInjections();

			const workingMem = buildWorkingMemorySection(session.miniMemStore);
			const sessionArchitecture = buildSessionArchitecture();

			const contractMsg = {
				role: "user" as const,
				content: [{ type: "text" as const, text: buildFinalResponseContract() }],
			};

			// Assemble context prefix via assembleContextPrefix — single call, consistent order
			const prefixMsgs = assembleContextPrefix({
				systemMsg,
				sessionArchitecture,
				workingMemory: workingMem,
				routingInstructions: ROUTING_INSTRUCTIONS,
				preamble,
				recencyList,
				relevanceList,
				followUpMsg,
				checkpointMsg,
				contractMsg,
			});

			// Cache the stable prefix (system + context sections + final response contract) for
			// subsequent context calls within this agent cycle. currentMessages is
			// appended fresh each time.
			contextCacheStore(round.contextCache, envPreamble, [...prefixMsgs], userPrompt);

			const resultMessages = applyContextSizeWarning(round.contextCache.messages!, currentMessages, systemMsg);

			return { messages: resultMessages } as any;
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
		// New agent cycle — fresh round state
		round = createRound();
	});

	pi.on("message_end", async (event, _ctx) => {
		const state: MessageEndProcessingState = {
			accumulatedText: round.accumulatedText,
			toolCallCount: round.toolCallCount,
			toolCallNames: round.toolCallNames,
			toolCalls: round.toolCalls,
			responseSegments: round.responseSegments,
		};
		applyMessageEndToState(event.message, state);
		round.accumulatedText = state.accumulatedText;
		round.toolCallCount = state.toolCallCount;
		round.toolCallNames = state.toolCallNames;
		round.toolCalls = state.toolCalls;
		round.responseSegments = state.responseSegments;
	});

	// ── Multi-model routing: apply pending model switch at turn end ──
	// This runs after each individual turn completes. If the LLM called
	// semblr_report_phase during the turn, the pending switch is applied
	// here so the next turn (or next round) starts on the correct model.
	pi.on("turn_end", async (_event, ctx) => {
		const routingActive = session.routingEnabled ?? SEMBLR_CONFIG.routing.enabled;
		if (!routingActive) return;
		if (!round.pendingModelSwitch) return;

		const targetModel = round.pendingModelSwitch;
		const phase = round.currentPhase ?? "unknown";
		const { provider, model: resolvedId } = resolveModelId(targetModel);

		// Skip if already on the target model
		if (ctx.model?.id !== resolvedId) {
			const modelObj = ctx.modelRegistry.find(provider, resolvedId);
			if (!modelObj) {
				ctx.ui.notify(
					`[Multi-model] Model not found in registry: ${targetModel} (resolved: ${provider}/${resolvedId})`,
					"error",
				);
			} else {
				const ok = await pi.setModel(modelObj);
				if (ok) {
					ctx.ui.notify(
						`[Multi-model] Switching to ${targetModel} for ${phase} phase (switch ${round.switchCounter}/${SEMBLR_CONFIG.routing.maxSwitchesPerCycle})`,
						"info",
					);
				} else {
					ctx.ui.notify(
						`[Multi-model] Failed to switch to ${targetModel} for ${phase} phase (no API key configured?)`,
						"error",
					);
				}
			}
		}

		// Reset pending switch regardless of success/failure
		round.pendingModelSwitch = null;
	});

	async function restoreOriginalModel(ctx: ExtensionContext): Promise<void> {
		const routingActive = session.routingEnabled ?? SEMBLR_CONFIG.routing.enabled;
		if (!routingActive || !round.originalModel) return;

		const original = round.originalModel;
		const originalDisplay = `${original.provider}/${original.modelId}`;
		const currentModelMatchesOriginal =
			ctx.model?.provider === original.provider && ctx.model?.id === original.modelId;
		let originalModelIsActive = currentModelMatchesOriginal;

		// Check if current model differs from original (i.e., a switch happened)
		if (!currentModelMatchesOriginal) {
			// Restore by full provider+model identity. Bare IDs are ambiguous (for example, gpt-5.5).
			let originalModel = ctx.modelRegistry.find(original.provider, original.modelId);

			// Fallback 1: match by modelId only (ignore provider).
			// The model may have been registered under a different provider alias.
			if (!originalModel) {
				const allModels = ctx.modelRegistry.getAll();
				originalModel = allModels.find((m) => m.id === original.modelId);
			}

			// Fallback 2: match by provider only (any model from the same provider).
			// Switching to a sibling model from the original provider is safer
			// than leaving a phase-specific model active across rounds.
			if (!originalModel) {
				const allModels = ctx.modelRegistry.getAll();
				originalModel = allModels.find((m) => m.provider === original.provider);
			}

			if (originalModel) {
				const ok = await pi.setModel(originalModel);
				if (ok) {
					originalModelIsActive = true;
					ctx.ui.notify(`[Multi-model] Restored original model: ${originalDisplay}`, "info");
				} else {
					ctx.ui.notify(`[Multi-model] Failed to restore original model: ${originalDisplay}`, "error");
				}
			} else {
				ctx.ui.notify(`[Multi-model] Original model not found in registry: ${originalDisplay}`, "error");
			}
		}

		if (
			originalModelIsActive &&
			original.thinkingLevel !== null &&
			pi.getThinkingLevel() !== original.thinkingLevel
		) {
			pi.setThinkingLevel(original.thinkingLevel);
		}
	}

	pi.on("agent_end", async (event, ctx) => {
		const { messages } = event;

		// Get user prompt -- prefer agent_start cached value, fall back to messages
		const userPrompt = extractAgentEndUserPrompt(round.userPrompt, messages);

		if (!userPrompt) {
			ctx.ui.setStatus("semblr", "\u{1f9e0} agent_end: no user prompt to save");
			await restoreOriginalModel(ctx);
			return;
		}

		// Build response text from accumulated assistant text across all tool iterations
		const rawResponseText = extractAgentEndResponseText(round.accumulatedText, messages);

		if (!rawResponseText) {
			ctx.ui.setStatus("semblr", "\u{1f9e0} agent_end: no response text");
			await restoreOriginalModel(ctx);
			return;
		}

		// Detect and strip the round_needs_followup marker, flagging for follow-up
		// injection on the next context assembly.
		const { cleanedText: responseText, needsFollowup } = extractAndStripFollowupMarker(rawResponseText);

		fs.mkdirSync(ROUNDS_DIR, { recursive: true });

		const roundFileName = createRoundFilePath(userPrompt, responseText, round.toolCalls);
		const roundPath = `${ROUNDS_DIR}/${roundFileName}`;

		// Push to causal chain — even on dedup, this ensures the in-memory buffer
		// tracks every round seen in this session.
		session.causalChain.push(
			buildAgentEndChainEntry(roundFileName, userPrompt, responseText, round.toolCalls.length, round.toolCallNames),
		);

		// Skip if already saved (deduplication by content hash)
		if (fs.existsSync(roundPath)) {
			// Even on dedup, run grouping if the round has a combined embedding
			// (or if this is a short-prompt round with embedding skipped, use null)
			try {
				const existing = JSON.parse(fs.readFileSync(roundPath, "utf-8"));
				if (existing.promptEmbedding) {
					const vec = existing.promptEmbedding || null;
					assignToGroup(
						session.roundGroups,
						session.causalChain[session.causalChain.length - 1],
						vec,
						SEMBLR_GROUP_THRESHOLD,
						needsFollowup ? session.lastFollowupGroupIdx : null,
					);
				}
			} catch {
				/* best-effort */
			}
			ctx.ui.setStatus("semblr", `\u{1f9e0} round already saved (${roundFileName})`);
			lastRoundFileName = roundFileName;
			round.accumulatedText = [];
			round.userPrompt = null;
			round.turnIndex = null;
			flushStatsFile(statsState, STATS_PATH, SEMBLR_DIR); // causal chain was pushed, so position scores may have changed
			await restoreOriginalModel(ctx);
			return;
		}

		// Compute parentId from session.causalChain
		const parentId = getAgentEndParentId(session.causalChain);

		// Write round file
		const roundData = buildAgentEndRoundData({
			userPrompt,
			responseText,
			turnIndex: round.turnIndex,
			toolCallCount: round.toolCallCount,
			toolCallNames: round.toolCallNames,
			toolCalls: round.toolCalls,
			responseSegments: round.responseSegments,
			parentId,
			needsFollowup,
			summary: round.lastCheckpointSummary ?? undefined,
		});

		try {
			fs.writeFileSync(roundPath, JSON.stringify(roundData, null, 2));
		} catch (err) {
			ctx.ui.setStatus("semblr", `\u{1f9e0} write error: ${(err as Error).message}`);
			round.accumulatedText = [];
			round.userPrompt = null;
			round.turnIndex = null;
			await restoreOriginalModel(ctx);
			return;
		}

		// Three embeddings for each round:
		//   1. prompt embedding → index.csv as :prompt
		//   2. response (truncated, REDACTED dropped) embedding → index.csv as :response
		//   3. concat(prompt, truncated+dropped response) embedding → round.json (for grouping)
		// Embedding providers may truncate long inputs, so we clip the response explicitly
		// using the configured embedding max-token budget to control what gets included.
		// Embedding a single concatenated text (rather than averaging separate prompt
		// and response vectors) preserves the semantic relationship between them.
		// See https://github.com/vhallac/semblr/issues/36
		const apiKey = await getApiKey(ctx, { config: SEMBLR_CONFIG });
		if (!apiKey) {
			ctx.ui.setStatus("semblr", "\u{1f9e0} saved but not embedded (no API key)");
			lastRoundFileName = roundFileName;
			round.accumulatedText = [];
			round.userPrompt = null;
			round.turnIndex = null;
			await restoreOriginalModel(ctx);
			return;
		}

		if (round.skipPromptEmbedding) {
			// Short prompt: skip prompt embedding (noisy), but still embed response.
			// Response is usually significant even when the prompt is short.
			// Per https://github.com/vhallac/semblr/issues/38#issuecomment-4629826478
			try {
				const { clippedResponse } = buildAgentEndEmbeddingTexts("", responseText, EMBEDDING_RESPONSE_MAX_BYTES);

				// Embed response only (no noisy prompt prefix)
				const responseVec = normalize(await embedText(clippedResponse, apiKey, embeddingClientDeps(ctx)));

				// Write :response row to index (skip :prompt -- prompt was noise)
				appendToIndex(`${roundFileName}:response`, responseVec, SEMBLR_CONFIG.embeddingModel);

				// Embed checkpoint summary if present
				if (round.lastCheckpointSummary) {
					const summaryText = buildCheckpointSummaryText(round.lastCheckpointSummary);
					const summaryVec = normalize(await embedText(summaryText, apiKey, embeddingClientDeps(ctx)));
					appendToIndex(`${roundFileName}:summary`, summaryVec, SEMBLR_CONFIG.embeddingModel);
				}

				ctx.ui.setStatus("semblr", `🧠 saved + response-embedded round (${roundFileName})`);

				finalizeRoundEmbedding({
					roundPath,
					embeddingVec: responseVec,
					needsFollowup,
				});
			} catch (err) {
				ctx.ui.setStatus("semblr", `🧠 response embedding error: ${(err as Error).message}`);
			}
		} else {
			try {
				// Strip context-injection REDACTED markers and clip to the configured embedding budget.
				const { clippedResponse, combinedText } = buildAgentEndEmbeddingTexts(
					userPrompt,
					responseText,
					EMBEDDING_RESPONSE_MAX_BYTES,
				);

				// Embedding #1: prompt (reuse cached round.promptVec if available)
				const promptVec =
					round.promptVec ?? normalize(await embedText(userPrompt, apiKey, embeddingClientDeps(ctx)));

				// Embedding #2 + #3 in parallel
				const [responseVec, combinedVec] = await Promise.all([
					embedText(clippedResponse, apiKey, embeddingClientDeps(ctx)),
					embedText(combinedText, apiKey, embeddingClientDeps(ctx)),
				]);

				// Save to index: :prompt and :response (normalized for cosine similarity)
				appendToIndex(`${roundFileName}:prompt`, normalize(promptVec), SEMBLR_CONFIG.embeddingModel);
				appendToIndex(`${roundFileName}:response`, normalize(responseVec), SEMBLR_CONFIG.embeddingModel);

				// Embed checkpoint summary if present
				if (round.lastCheckpointSummary) {
					const summaryText = buildCheckpointSummaryText(round.lastCheckpointSummary);
					const summaryVec = normalize(await embedText(summaryText, apiKey, embeddingClientDeps(ctx)));
					appendToIndex(`${roundFileName}:summary`, summaryVec, SEMBLR_CONFIG.embeddingModel);
				}

				ctx.ui.setStatus("semblr", `\u{1f9e0} saved + embedded round (${roundFileName})`);

				finalizeRoundEmbedding({
					roundPath,
					embeddingVec: combinedVec,
					needsFollowup,
				});
			} catch (err) {
				ctx.ui.setStatus("semblr", `\u{1f9e0} embedding error: ${(err as Error).message}`);
			}
		}

		lastRoundFileName = roundFileName;
		round.accumulatedText = [];
		round.userPrompt = null;
		round.turnIndex = null;

		// ◈ Flush stats to disk (atomically) and show in TUI
		flushStatsFile(statsState, STATS_PATH, SEMBLR_DIR);

		// Combine chain-read stats with total indexed rounds count
		const indexExists = fs.existsSync(INDEX_PATH);
		const idx = indexExists ? loadIndex() : [];
		const totalRounds = new Set(idx.map((e: { filePath: string }) => indexRoundFileFromPath(e.filePath))).size;
		ctx.ui.setStatus(
			"semblr",
			`🧠 ${totalRounds} total indexed | ${formatGroupStats(session.roundGroups, SEMBLR_GROUP_THRESHOLD)}`,
		);

		// ── Multi-model routing: restore original model at round end ──
		// The model switch was applied at turn_end. At agent_end we restore the
		// original model because pi's model setting is sticky — it survives past
		// the current round. Restoring ensures the next round starts on the
		// user's original model, not the phase-specific model from this round.
		await restoreOriginalModel(ctx);
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
		// Clear session scoped state — new session starts fresh
		session = createSession();

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

				const apiKey = await getApiKey(ctx2, { config: SEMBLR_CONFIG });
				if (!apiKey) {
					return {
						content: [{ type: "text", text: "No API key available for embedding. Skipping search." }],
						details: {},
					};
				}

				// Embed the query
				const queryVec = normalize(await embedText(query, apiKey, embeddingClientDeps(ctx2)));

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

		// Register semblr_checkpoint — progress checkpoint for context-size warnings
		// Always registered; the agent is instructed NOT to call it unless a context-size
		// warning prompt explicitly tells it to. Dynamic registration would wreck the cache.
		pi.registerTool({
			name: "semblr_checkpoint",
			label: "Semblr Checkpoint",
			description:
				"INTERNAL: Records a progress checkpoint. DO NOT call this tool unless explicitly instructed to do so by a context size warning in the conversation. Calling this tool without being instructed is a waste of tokens.",
			promptSnippet: "Records a progress checkpoint (internal use only)",
			parameters: Type.Object({
				currentTask: Type.String({
					description: "1-2 sentence summary of the current task you were working on",
				}),
				progressMade: Type.Array(Type.String(), {
					description: "Concrete items completed so far, one per entry",
				}),
				currentState: Type.Array(Type.String(), {
					description: "Current state: files modified, decisions made, tests passing/failing, known issues",
				}),
				nextSteps: Type.Array(Type.String(), {
					description: "Concrete next actions, ordered by priority",
				}),
				keyFindings: Type.Array(Type.String(), {
					description: "Important discoveries, design decisions, rationale",
				}),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const p = params as CheckpointSummary;
				// Only store if a context-size warning was actually issued this cycle.
				// This prevents the agent from calling it unsolicited and corrupting state.
				if (round.contextWarningIssued > 0) {
					round.lastCheckpointSummary = {
						currentTask: p.currentTask,
						progressMade: p.progressMade,
						currentState: p.currentState,
						nextSteps: p.nextSteps,
						keyFindings: p.keyFindings,
					};
					return {
						content: [
							{
								type: "text",
								text: "Checkpoint recorded. Your progress summary has been saved. You may now stop — do not start new work.",
							},
						],
						details: {},
					};
				}
				return {
					content: [
						{
							type: "text",
							text: "No context size warning is active. This tool should not be called without being instructed. No checkpoint was saved.",
						},
					],
					details: {},
				};
			},
		});

		// Register mini_mem__add — add a working memory slot
		pi.registerTool({
			name: "mini_mem__add",
			label: "Add Working Memory",
			description:
				'Store a note in working memory. Use this after making a plan, after an important decision, or when the user says "remember this." Slots are limited to ~7; when full, the oldest is silently evicted. Returns the assigned id and the updated list.',
			promptSnippet: "Add a note to working memory",
			parameters: Type.Object({
				summary: Type.String({ description: "Short label for the memory slot" }),
				content: Type.String({ description: "Full note text to store" }),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const { summary, content } = params as { summary: string; content: string };
				const id = addSlot(session.miniMemStore, summary, content, lastRoundFileName ?? undefined);
				const lines: string[] = [`Stored as memory slot [id: ${id}]. Current slots:`];
				for (const slot of session.miniMemStore.slots) {
					lines.push(`- [id: ${slot.id}] ${slot.summary}`);
				}
				return {
					content: [{ type: "text", text: lines.join("\n") }],
					details: {},
				};
			},
		});

		// Register mini_mem__get — retrieve a working memory slot without consuming it
		pi.registerTool({
			name: "mini_mem__get",
			label: "Get Working Memory",
			description:
				"Retrieve a working memory slot by its id. The slot stays in memory after retrieval. Use this to review a plan, decision, or note stored earlier.",
			promptSnippet: "Retrieve a working memory slot",
			parameters: Type.Object({
				id: Type.Number({ description: "The slot id to retrieve" }),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const { id } = params as { id: number };
				const slot = getSlot(session.miniMemStore, id);
				if (!slot) {
					return {
						content: [{ type: "text", text: `No memory slot found with id: ${id}.` }],
						details: {},
					};
				}
				return {
					content: [{ type: "text", text: formatMiniMemSlot(slot) }],
					details: {},
				};
			},
		});

		// Register mini_mem__update — overwrite a working memory slot
		pi.registerTool({
			name: "mini_mem__update",
			label: "Update Working Memory",
			description:
				"Overwrite an existing working memory slot's summary and content. Use for evolving plans, TODO lists, or updating decisions.",
			promptSnippet: "Update a working memory slot",
			parameters: Type.Object({
				id: Type.Number({ description: "The slot id to update" }),
				summary: Type.String({ description: "New short label" }),
				content: Type.String({ description: "New full content" }),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const { id, summary, content } = params as { id: number; summary: string; content: string };
				const slot = updateSlot(session.miniMemStore, id, summary, content, lastRoundFileName ?? undefined);
				if (!slot) {
					return {
						content: [{ type: "text", text: `No memory slot found with id: ${id}.` }],
						details: {},
					};
				}
				return {
					content: [{ type: "text", text: formatMiniMemSlot(slot) }],
					details: {},
				};
			},
		});

		// Register mini_mem__delete — remove a working memory slot
		pi.registerTool({
			name: "mini_mem__delete",
			label: "Delete Working Memory",
			description: "Delete a working memory slot by its id.",
			promptSnippet: "Delete a working memory slot",
			parameters: Type.Object({
				id: Type.Number({ description: "The slot id to delete" }),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const { id } = params as { id: number };
				const found = deleteSlot(session.miniMemStore, id);
				if (!found) {
					return {
						content: [{ type: "text", text: `No memory slot found with id: ${id}.` }],
						details: {},
					};
				}
				return {
					content: [{ type: "text", text: `Memory slot [id: ${id}] deleted.` }],
					details: {},
				};
			},
		});

		// Register mini_mem__get_and_delete — one-shot get + delete
		pi.registerTool({
			name: "mini_mem__get_and_delete",
			label: "Get and Delete Working Memory",
			description:
				'One-shot retrieval: get the full content of a working memory slot, then delete it. Use for truly disposable notes (e.g., "after this task remind me to X").',
			promptSnippet: "Retrieve and delete a working memory slot",
			parameters: Type.Object({
				id: Type.Number({ description: "The slot id to retrieve and delete" }),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, _ctx2) {
				const { id } = params as { id: number };
				const slot = getAndDeleteSlot(session.miniMemStore, id);
				if (!slot) {
					return {
						content: [{ type: "text", text: `No memory slot found with id: ${id}.` }],
						details: {},
					};
				}
				return {
					content: [{ type: "text", text: formatMiniMemSlot(slot) }],
					details: {},
				};
			},
		});

		// Register semblr_report_phase — multi-model routing phase reporter
		pi.registerTool({
			name: "semblr_report_phase",
			label: "Report Phase",
			description:
				"Report your current generation phase for multi-model routing. Call this BEFORE starting work in a new phase — not during, not after — to route the next agent turn to a model specialized for that phase. See [MULTI-MODEL ROUTING] in the context for details on when to report each phase.",
			promptSnippet: "Report your current generation phase for multi-model routing",
			parameters: Type.Object({
				phase: Type.Union(
					[
						Type.Literal("exploring"),
						Type.Literal("planning"),
						Type.Literal("executing"),
						Type.Literal("verifying"),
						Type.Literal("reporting"),
					],
					{
						description:
							"Your current generation phase. 'exploring': pulling in external data by reading, searching, exploring. 'planning': formulating a plan of response, structured thinking. 'executing': implementing a plan, writing code, making edits. 'verifying': execution done, validating output and created files. 'reporting': done with work, about to deliver final output or summary.",
					},
				),
				note: Type.Optional(
					Type.String({
						description:
							"EXPERIMENTAL: A note to the next model that handles this task. Share context, decisions made, open questions, or suggestions for how to proceed. Keep it concise.",
					}),
				),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx2) {
				const { phase, note } = params as { phase: PhaseName; note?: string };

				const routingActive = session.routingEnabled ?? SEMBLR_CONFIG.routing.enabled;
				if (!routingActive) {
					return {
						content: [{ type: "text", text: "Multi-model routing is not enabled. Phase report ignored." }],
						details: {},
					};
				}

				// Capture original model on first phase report (before any switch)
				if (round.originalModel === null && ctx2.model?.id) {
					round.originalModel = {
						provider: ctx2.model.provider,
						modelId: ctx2.model.id,
						thinkingLevel: pi.getThinkingLevel(),
					};
				}

				// Store the reported phase and optional note on round state
				round.currentPhase = phase;
				round.phaseNote = note ?? null;
				round.phaseHistory.push(phase);

				// Update status bar with routing indicator
				const modelShortName = ctx2.model?.id ?? "?";
				ctx2.ui.setStatus(
					"semblr-routing",
					`\u{1f500} ${phase} (${modelShortName}) | Sw: ${round.switchCounter}/${SEMBLR_CONFIG.routing.maxSwitchesPerCycle}`,
				);

				// Look up the target model from the phase→model map
				const targetModel = SEMBLR_CONFIG.routing.phaseModels[phase];

				// If target is null, no switch needed for this phase
				if (targetModel === null) {
					round.pendingModelSwitch = null;
					return {
						content: [{ type: "text", text: `Phase recorded: ${phase}.` }],
						details: {},
					};
				}

				// Idempotent: same pending switch already set, don't double-count
				if (round.pendingModelSwitch === targetModel) {
					return {
						content: [{ type: "text", text: `Phase recorded: ${phase}.` }],
						details: {},
					};
				}

				// Check if already on the target model
				const { model: resolvedId } = resolveModelId(targetModel);
				if (ctx2.model?.id === resolvedId) {
					round.pendingModelSwitch = null;
					return {
						content: [{ type: "text", text: `Phase recorded: ${phase}.` }],
						details: {},
					};
				}

				// Check switch limit — only count if we actually set a pending switch
				if (round.switchCounter >= SEMBLR_CONFIG.routing.maxSwitchesPerCycle) {
					round.pendingModelSwitch = null;
					round.switchLimitReached = true;
					return {
						content: [{ type: "text", text: `Phase recorded: ${phase}.` }],
						details: {},
					};
				}

				// Set the pending model switch
				round.pendingModelSwitch = targetModel;
				round.switchCounter++;

				return {
					content: [{ type: "text", text: `Phase recorded: ${phase}.` }],
					details: {},
				};
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

		// Register /semblr:routing — multi-model routing control
		pi.registerCommand("semblr:routing", {
			description: "Control multi-model routing: on, off, status",
			handler: async (args, commandCtx) => {
				const tokens = splitCommandArgs(args.trim());
				const subcommand = tokens[0]?.toLowerCase() ?? "";

				switch (subcommand) {
					case "on": {
						session.routingEnabled = true;
						commandCtx.ui.notify("\u{1f500} Multi-model routing: ON", "info");
						break;
					}
					case "off": {
						session.routingEnabled = false;
						round.pendingModelSwitch = null;
						commandCtx.ui.notify("\u{1f500} Multi-model routing: OFF", "info");
						break;
					}
					case "status": {
						const active = session.routingEnabled ?? SEMBLR_CONFIG.routing.enabled;
						const lines: string[] = [];
						lines.push(`\u{1f500} Semblr Routing — ${active ? "ACTIVE" : "INACTIVE"}`);

						const phases = Object.entries(SEMBLR_CONFIG.routing.phaseModels)
							.filter(([, model]) => model !== null)
							.map(([phase, model]) => `${phase}\u2192${model}`);
						if (phases.length > 0) {
							lines.push(`   Phases:   ${phases.join(", ")}`);
						} else {
							lines.push("   Phases:   (none configured — no switches will occur)");
						}

						lines.push(
							`   Switches: ${round.switchCounter}/${SEMBLR_CONFIG.routing.maxSwitchesPerCycle} this cycle`,
						);
						if (round.phaseHistory.length > 0) {
							lines.push(`   History:  ${round.phaseHistory.join(" \u2192 ")}`);
						}
						if (round.switchLimitReached) {
							lines.push("   Limit:    switch limit reached this cycle");
						}

						commandCtx.ui.notify(lines.join("\n"), "info");
						break;
					}
					default: {
						commandCtx.ui.notify("Usage: /semblr:routing [on|off|status]", "error");
						break;
					}
				}
			},
		});
	});
}
