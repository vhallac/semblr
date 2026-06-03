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
	name: string;
	arguments: string;
	result_summary: string;
	result_full?: string;
	result_truncated?: boolean;
}

export interface RoundData {
	userPrompt: string;
	responseSequence: string;
	turnIndex: number;
	userTimestamp?: number;
	toolCallCount?: number;
	toolCallNames?: string[];
	toolCalls?: ToolCallDetail[];
	promptEmbedding?: number[];
	parentId?: string | null;
	relatedParentId?: string | null;
	needsFollowup?: boolean;
}

export interface ToolResult {
	content: Array<{ type: "text"; text: string }>;
	details: Record<string, unknown>;
}
