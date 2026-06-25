import type { SemanticGroup } from "./grouping.ts";
import type { ChainEntry, CheckpointSummary, ResponseSegment, ToolCallDetail } from "./round-data.ts";
import type { PhaseName } from "./semblr-config.ts";
import { createMiniMemStore, type MiniMemStore } from "./working-memory.ts";

export type { ChainEntry, CheckpointSummary, ResponseSegment, ToolCallDetail } from "./round-data.ts";

export type RoundGroup = SemanticGroup<ChainEntry>;

// ─────────────────────────────────────────────
// ContextCache — triple that must stay in lockstep.
// Wrapping these in a single helper prevents the three variables drifting apart
// (partial reset, set-one-miss-two, etc.).  Use .valid(userPrompt) to check and
// .store(preamble, messages, userPrompt) to set all three at once.
// ─────────────────────────────────────────────

export interface ContextCache {
	envPreamble: string | null;
	messages: unknown[] | null;
	userPrompt: string | null;
}

export function createContextCache(): ContextCache {
	return { envPreamble: null, messages: null, userPrompt: null };
}

/** True when the cache is populated AND the current user prompt matches. */
export function contextCacheValid(cc: ContextCache, userPrompt: string): boolean {
	return !!(cc.messages && cc.envPreamble && cc.userPrompt === userPrompt);
}

/** Store all three at once — atomic assignment. */
export function contextCacheStore(
	cc: ContextCache,
	envPreamble: string,
	messages: unknown[],
	userPrompt: string,
): void {
	cc.envPreamble = envPreamble;
	cc.messages = messages;
	cc.userPrompt = userPrompt;
}

/** Snapshot messages into the cache (keeps envPreamble + userPrompt as-is). */
export function contextCacheSnapshot(cc: ContextCache, messages: unknown[]): void {
	cc.messages = messages;
}

// ─────────────────────────────────────────────
// Session — state that survives between rounds within the same session.
// Reset at session_start.
// ─────────────────────────────────────────────

export interface SessionState {
	miniMemStore: MiniMemStore;
	causalChain: ChainEntry[];
	roundGroups: RoundGroup[];
	lastFollowupGroupIdx: number | null;
	injectedFollowupRounds: Set<string>;
	injectedCheckpointRounds: Set<string>;
}

export function createSession(): SessionState {
	return {
		miniMemStore: createMiniMemStore(),
		causalChain: [],
		roundGroups: [],
		lastFollowupGroupIdx: null,
		injectedFollowupRounds: new Set(),
		injectedCheckpointRounds: new Set(),
	};
}

// ─────────────────────────────────────────────
// Round — state reset at each agent_start. Lives for one user prompt → full response.
// ─────────────────────────────────────────────

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

export interface OriginalModelState {
	provider: string;
	modelId: string;
	thinkingLevel: ThinkingLevel | null;
}

export interface RoundState {
	userPrompt: string | null;
	turnIndex: number | null;
	accumulatedText: string[];
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
	responseSegments: ResponseSegment[];
	// embedding cache
	lastContextUserPrompt: string | null;
	lastContextVec: number[];
	promptVec: number[] | null;
	skipPromptEmbedding: boolean;
	presentedRecorded: boolean;
	// full-message cache (triple wrapped in ContextCache — see helpers above)
	contextCache: ContextCache;
	// checkpoint
	lastCheckpointSummary: CheckpointSummary | null;
	contextWarningIssued: number;
	// multi-model routing
	/** Number of model switches that have occurred this agent cycle. Resets at agent_start. */
	switchCounter: number;
	/** Most recently reported phase for the current round. */
	currentPhase: PhaseName | null;
	/** Optional note from the LLM to pass to the next model on switch. */
	phaseNote: string | null;
	/** Model ID to switch to at turn_end, derived from the reported phase and the phase→model map. */
	pendingModelSwitch: string | null;
	/** Full model identity active when semblr_report_phase was first called (before any switch). */
	originalModel: OriginalModelState | null;
}

export function createRound(): RoundState {
	return {
		userPrompt: null,
		turnIndex: null,
		accumulatedText: [],
		toolCallCount: 0,
		toolCallNames: [],
		toolCalls: [],
		responseSegments: [],
		lastContextUserPrompt: null,
		lastContextVec: [],
		promptVec: null,
		skipPromptEmbedding: false,
		presentedRecorded: false,
		contextCache: createContextCache(),
		lastCheckpointSummary: null,
		contextWarningIssued: 0,
		switchCounter: 0,
		currentPhase: null,
		phaseNote: null,
		pendingModelSwitch: null,
		originalModel: null,
	};
}
