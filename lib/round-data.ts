export interface ChainEntry {
	fileName: string;
	userPrompt: string;
	responseSequence: string;
	toolSummary: string;
}

export interface ResponseSegment {
	type: "text" | "toolCall";
	text?: string;
	toolCallIndex?: number;
}

export interface ToolCallDetail {
	index: number;
	id?: string;
	name: string;
	arguments: string;
	result_summary: string;
	result_full?: string;
	result_truncated?: boolean;
}

export interface RoundData {
	/** Content hash (MD5 of userPrompt + responseSequence) — matches the round filename stem. */
	id?: string;
	userPrompt: string;
	responseSequence: string;
	turnIndex: number;
	userTimestamp?: number;
	toolCallCount?: number;
	toolCallNames?: string[];
	toolCalls?: ToolCallDetail[];
	responseSegments?: ResponseSegment[];
	promptEmbedding?: number[];
	parentId?: string | null;
	relatedParentId?: string | null;
	needsFollowup?: boolean;
	/** Set on rounds recovered from previous session files (backfill), not live-saved rounds. */
	recovered?: boolean;
	summary?: CheckpointSummary;
}

export interface CheckpointSummary {
	currentTask: string;
	progressMade: string[];
	currentState: string[];
	nextSteps: string[];
	keyFindings: string[];
}

export interface ToolResult {
	content: Array<{ type: "text"; text: string }>;
	details: Record<string, unknown>;
}
