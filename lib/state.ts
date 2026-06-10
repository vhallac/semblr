import type { SemanticGroup } from "./grouping.ts";
import type { ChainEntry, CheckpointSummary, ResponseSegment, ToolCallDetail } from "./round-data.ts";
import { createMiniMemStore, type MiniMemStore } from "./working-memory.ts";

export type { ChainEntry, CheckpointSummary, ResponseSegment, ToolCallDetail } from "./round-data.ts";

export type RoundGroup = SemanticGroup<ChainEntry>;

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
	// full-message cache
	cachedEnvPreamble: string | null;
	cachedContextMessages: unknown[] | null;
	cachedUserPromptForContext: string | null;
	// checkpoint
	lastCheckpointSummary: CheckpointSummary | null;
	contextWarningIssued: number;
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
		cachedEnvPreamble: null,
		cachedContextMessages: null,
		cachedUserPromptForContext: null,
		lastCheckpointSummary: null,
		contextWarningIssued: 0,
	};
}
