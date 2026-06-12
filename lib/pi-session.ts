import { createRoundFilePath } from "./hash.ts";

interface ParsedToolCallDetail {
	index: number;
	name: string;
	arguments: string;
	result_summary: string;
	result_full?: string;
	result_truncated?: boolean;
}

interface ParsedResponseSegment {
	type: "text" | "toolCall";
	text?: string;
	toolCallIndex?: number;
}

export interface ParsedPiRound {
	id: string;
	userPrompt: string;
	responseSequence: string;
	responseSegments: ParsedResponseSegment[];
	userTimestamp: number;
	responseEndTimestamp: number;
	turnIndex: number;
	sessionLabel?: string;
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ParsedToolCallDetail[];
}

export interface ParsePiSessionOptions {
	sessionLabel?: string;
	skipShortFinalResponse?: boolean;
	now?: () => number;
}

interface SessionEntry {
	type?: string;
	id?: string;
	message?: {
		role?: string;
		content?: Array<Record<string, unknown>>;
		timestamp?: number;
		toolName?: string;
	};
}

export interface ReconstructedPiRound {
	roundFile: string;
	round: ParsedPiRound;
}

export function parsePiTextContent(content?: Array<{ type?: string; text?: unknown }>): string {
	if (!content || !Array.isArray(content)) return "";
	return content
		.filter((c): c is { type: string; text: string } => c.type === "text" && typeof c.text === "string")
		.map((c) => c.text)
		.join(" ")
		.trim();
}

export function parsePiSessionJsonl(
	raw: string,
	options: ParsePiSessionOptions & { sessionLabel: string },
): Array<ParsedPiRound & { sessionLabel: string }>;
export function parsePiSessionJsonl(raw: string, options?: ParsePiSessionOptions): ParsedPiRound[];
export function parsePiSessionJsonl(raw: string, options: ParsePiSessionOptions = {}): ParsedPiRound[] {
	const now = options.now ?? Date.now;
	const entries: SessionEntry[] = raw
		.trim()
		.split("\n")
		.filter(Boolean)
		.flatMap((line) => {
			try {
				const parsed = JSON.parse(line) as SessionEntry;
				return parsed && typeof parsed === "object" ? [parsed] : [];
			} catch {
				return [];
			}
		});

	const rounds: ParsedPiRound[] = [];
	let currentUserMsg: SessionEntry | null = null;
	let responseParts: string[] = [];
	let responseSegments: ParsedResponseSegment[] = [];
	let toolNames: string[] = [];
	let toolCallCount = 0;
	let toolCalls: ParsedToolCallDetail[] = [];
	let roundIndex = 0;

	const flush = (responseEndTimestamp: number, isFinal: boolean) => {
		if (!currentUserMsg) return;
		const responseSequence = responseParts.join("\n\n").trim();
		if (isFinal && options.skipShortFinalResponse && responseSequence.length < 20 && roundIndex === 0) return;
		const round: ParsedPiRound = {
			id: currentUserMsg.id ?? "",
			userPrompt: parsePiTextContent(currentUserMsg.message?.content),
			responseSequence,
			responseSegments,
			userTimestamp: currentUserMsg.message?.timestamp ?? 0,
			responseEndTimestamp,
			turnIndex: roundIndex,
			toolCallCount,
			toolCallNames: [...new Set(toolNames)],
			toolCalls,
		};
		if (options.sessionLabel) round.sessionLabel = options.sessionLabel;
		rounds.push(round);
		roundIndex++;
	};

	for (const entry of entries) {
		if (entry.type !== "message" || !entry.message) continue;
		const role = entry.message.role;
		const content = entry.message.content;

		if (role === "user") {
			if (currentUserMsg) flush(entry.message.timestamp ?? now(), false);
			currentUserMsg = entry;
			responseParts = [];
			responseSegments = [];
			toolNames = [];
			toolCallCount = 0;
			toolCalls = [];
		} else if (role === "assistant" && currentUserMsg && content) {
			for (const block of content) {
				if (block.type === "text" && typeof block.text === "string" && block.text) {
					responseParts.push(block.text);
					responseSegments.push({ type: "text", text: block.text });
				} else if (block.type === "toolCall") {
					toolCallCount++;
					const name = typeof block.name === "string" ? block.name : undefined;
					if (name) toolNames.push(name);
					toolCalls.push({
						index: toolCalls.length,
						name: name ?? "unknown",
						arguments: JSON.stringify(block.arguments ?? {}),
						result_summary: "",
					});
					responseSegments.push({ type: "toolCall", toolCallIndex: toolCalls.length - 1 });
				}
			}
		} else if (role === "toolResult" && currentUserMsg) {
			const toolName = entry.message.toolName;
			if (toolName) toolNames.push(toolName);
			for (let i = toolCalls.length - 1; i >= 0; i--) {
				if (toolCalls[i].result_summary === "") {
					const resultText = parsePiTextContent(entry.message.content);
					toolCalls[i].result_summary = resultText.slice(0, 300);
					toolCalls[i].result_full = resultText;
					toolCalls[i].result_truncated = false;
					break;
				}
			}
		}
	}

	flush(now(), true);
	return rounds;
}

export function reconstructPiSessionRounds(rounds: readonly ParsedPiRound[]): ReconstructedPiRound[] {
	return rounds.map((round) => ({
		roundFile: createRoundFilePath(round.userPrompt, round.responseSequence, round.toolCalls),
		round,
	}));
}
