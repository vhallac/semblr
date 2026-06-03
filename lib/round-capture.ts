import * as fs from "node:fs";
import { computeContentHash } from "./hash.ts";
import { extractText } from "./message-content.ts";
import type { ChainEntry, ResponseSegment, RoundData, ToolCallDetail } from "./round-data.ts";

export function extractAgentEndUserPrompt(cachedPrompt: string | null, messages?: readonly unknown[]): string {
	let userPrompt = cachedPrompt ?? "";
	if (!userPrompt && messages) {
		const lastUser = [...messages].reverse().find((m) => (m as { role: string }).role === "user");
		if (lastUser) {
			const content = (lastUser as { content: unknown }).content;
			if (typeof content === "string") {
				userPrompt = content;
			} else if (Array.isArray(content)) {
				userPrompt = extractText(content as Array<{ type: string; text?: string }>);
			}
		}
	}
	return userPrompt;
}

export function extractAgentEndResponseText(accumulatedText: readonly string[], messages?: readonly unknown[]): string {
	let responseText = accumulatedText.join("\n\n").trim();
	if (!responseText) {
		const lastAssistant = messages
			? [...messages].reverse().find((m) => (m as { role: string }).role === "assistant")
			: null;
		if (lastAssistant) {
			const content = (lastAssistant as { content: unknown }).content;
			if (typeof content === "string") {
				responseText = content;
			} else if (Array.isArray(content)) {
				responseText = extractText(content as Array<{ type: string; text?: string }>);
			}
		}
	}
	return responseText;
}

export function buildAgentEndToolSummary(toolCallCount: number, toolCallNames: readonly string[]): string {
	return toolCallCount > 0 ? `${toolCallCount} tools (${toolCallNames.join(", ")})` : "0 tools (discussion)";
}

export function buildAgentEndChainEntry(
	fileName: string,
	userPrompt: string,
	responseText: string,
	toolCallCount: number,
	toolCallNames: readonly string[],
): ChainEntry {
	return {
		fileName,
		userPrompt,
		responseSequence: responseText,
		toolSummary: buildAgentEndToolSummary(toolCallCount, toolCallNames),
	};
}

export function getAgentEndParentId(chain: readonly { fileName: string }[]): string | null {
	return chain.length >= 2 ? chain[chain.length - 2].fileName : null;
}

export function buildAgentEndRoundData(args: {
	userPrompt: string;
	responseText: string;
	turnIndex: number | null;
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
	responseSegments: ResponseSegment[];
	parentId: string | null;
	userTimestamp?: number;
	needsFollowup?: boolean;
}): Record<string, unknown> {
	return {
		id: computeContentHash(args.userPrompt, args.responseText, args.toolCalls),
		userPrompt: args.userPrompt,
		responseSequence: args.responseText,
		turnIndex: args.turnIndex ?? 0,
		userTimestamp: args.userTimestamp ?? Date.now(),
		toolCallCount: args.toolCallCount,
		toolCallNames: args.toolCallNames,
		toolCalls: args.toolCalls,
		responseSegments: args.responseSegments,
		promptEmbedding: undefined,
		parentId: args.parentId,
		relatedParentId: null,
		needsFollowup: args.needsFollowup ?? false,
	};
}

export function buildAgentEndEmbeddingTexts(
	userPrompt: string,
	responseText: string,
	maxResponseBytes = 24000,
): { clippedResponse: string; combinedText: string } {
	const strippedResponse = responseText.replace(/\[(?:Tool call )?REDACTED[^\]]*\]\n?/g, "");
	const responseBuf = Buffer.from(strippedResponse, "utf-8");
	const clippedResponse =
		responseBuf.length > maxResponseBytes
			? responseBuf.slice(0, maxResponseBytes).toString("utf-8")
			: strippedResponse;
	return {
		clippedResponse,
		combinedText: `${userPrompt}\n\n${clippedResponse}`,
	};
}

/**
 * Detect and strip the round_needs_followup marker from the end of a response.
 * Returns the stripped text and whether the marker was present.
 */
export function extractAndStripFollowupMarker(responseText: string): {
	cleanedText: string;
	needsFollowup: boolean;
} {
	const followupMarker = "\nround_needs_followup";
	if (responseText.endsWith(followupMarker)) {
		return {
			cleanedText: responseText.slice(0, -followupMarker.length).trimEnd(),
			needsFollowup: true,
		};
	}
	return { cleanedText: responseText, needsFollowup: false };
}

/**
 * Read a round file and clear its needsFollowup flag atomically.
 * Returns the round data with needsFollowup=true, or null.
 */
export function readAndClearFollowupFlag(
	fullPath: string,
	fsImpl?: {
		existsSync: (p: string) => boolean;
		readFileSync: (p: string, encoding: "utf-8") => string;
		writeFileSync: (p: string, data: string) => void;
		renameSync: (oldP: string, newP: string) => void;
	},
): RoundData | null {
	const fs_ = fsImpl ?? fs;

	if (!fs_.existsSync(fullPath)) return null;

	let roundData: Record<string, unknown>;
	try {
		roundData = JSON.parse(fs_.readFileSync(fullPath, "utf-8"));
	} catch {
		return null;
	}

	if (!roundData.needsFollowup) return null;

	// Clear the flag atomically so it's only injected once
	try {
		const updated = { ...roundData, needsFollowup: false };
		fs_.writeFileSync(`${fullPath}.tmp.${process.pid}`, JSON.stringify(updated, null, 2));
		fs_.renameSync(`${fullPath}.tmp.${process.pid}`, fullPath);
	} catch {
		// best-effort — if we can't clear the flag, still return the data
	}

	return roundData as unknown as RoundData;
}

export function getRelatedParentIdFromGroup<T extends { fileName: string }>(
	group: { rounds: readonly T[] },
	roundEntry: T,
): string | null {
	if (group.rounds.length <= 1) return null;
	const groupRoundIdx = group.rounds.indexOf(roundEntry);
	return groupRoundIdx > 0 ? group.rounds[groupRoundIdx - 1].fileName : null;
}

export interface MessageEndProcessingState {
	accumulatedText: string[];
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
	responseSegments: ResponseSegment[];
}

export function applyMessageEndToState(message: unknown, state: MessageEndProcessingState): void {
	if (!message) return;

	const msg = message as { role?: string; content?: unknown; toolCallId?: string };
	if (msg.role === "user") {
		// User sent something -- don't reset the accumulator, this is a new agent
		// cycle (agent_start will reset it). Keep safe.
		return;
	}

	if (msg.role === "assistant") {
		// Extract text from this assistant message
		const content = msg.content as Array<{ type: string; text?: string }> | undefined;
		if (Array.isArray(content)) {
			for (const block of content) {
				if (block.type === "text" && block.text) {
					state.accumulatedText.push(block.text);
					state.responseSegments.push({ type: "text", text: block.text });
				} else if (block.type === "toolCall") {
					state.toolCallCount++;
					const blockRec = block as Record<string, unknown>;
					const name = blockRec.name as string | undefined;
					const id = blockRec.id as string | undefined;
					if (name && !state.toolCallNames.includes(name)) {
						state.toolCallNames.push(name);
					}
					if (id && name) {
						const detail: ToolCallDetail = {
							index: state.toolCalls.length,
							name,
							arguments: JSON.stringify(blockRec.arguments ?? {}),
							result_summary: "",
						};
						state.toolCalls.push(detail);
					}
					// Record this tool call's position in the response stream
					state.responseSegments.push({ type: "toolCall", toolCallIndex: state.toolCalls.length - 1 });
				}
			}
		}
		return;
	}

	if (msg.role === "toolResult") {
		// Pair tool results with their calls
		const toolCallId = msg.toolCallId;
		if (toolCallId) {
			// Find the matching ToolCallDetail by matching the last call without a result
			// (pi sessions don't expose the toolCallId -> toolCall mapping directly, so
			// we match sequentially — results arrive in order)
			for (let i = state.toolCalls.length - 1; i >= 0; i--) {
				if (state.toolCalls[i].result_summary === "") {
					const resultContent = msg.content as Array<{ type: string; text?: string }> | undefined;
					const resultText = resultContent ? extractText(resultContent) : "";
					state.toolCalls[i].result_summary = resultText.slice(0, 300);
					state.toolCalls[i].result_full = resultText;
					state.toolCalls[i].result_truncated = false;
					break;
				}
			}
		}
	}
}
