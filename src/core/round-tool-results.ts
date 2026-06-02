import * as fs from "node:fs";
import { formatFileSize } from "./context-format.ts";
import { selectRoundAssistantOutput, selectToolResultOutput } from "./detail-rendering.ts";
import type { RoundData, ToolResult } from "./round-data.ts";

export interface RoundDetailsParams {
	round: string;
	from_line?: number;
	line_count?: number;
	match?: string;
	max_matches?: number;
}

export type RoundDetailsToolResult = ToolResult;

export interface ToolDetailsParams {
	round: string;
	index: number;
	out__from_line?: number;
	out_line_count?: number;
	match?: string;
	max_matches?: number;
}

export type ToolDetailsToolResult = ToolResult;

export function buildRoundAssistantOutput(roundName: string, roundData: Record<string, unknown>): string {
	if (Array.isArray(roundData.responseSegments) && roundData.responseSegments.length > 0) {
		const parts: string[] = [];
		for (const seg of roundData.responseSegments as Array<{ type: string; text?: string; toolCallIndex?: number }>) {
			if (seg.type === "text" && seg.text) {
				parts.push(seg.text);
			} else if (seg.type === "toolCall" && seg.toolCallIndex != null) {
				parts.push(`[Tool call REDACTED: use get_tool_details("${roundName}", ${seg.toolCallIndex}) to expand]`);
			}
		}
		return parts.join("\n");
	}

	return (roundData.responseSequence as string) ?? "(empty)";
}

export function formatRoundToolMeta(roundData: Record<string, unknown>): string {
	if (roundData.toolCallCount != null && Number(roundData.toolCallCount) > 0) {
		const names = Array.isArray(roundData.toolCallNames) ? roundData.toolCallNames.join(", ") : "unknown";
		return `\n  Tools used: ${roundData.toolCallCount} (${names})`;
	}
	if (roundData.toolCallCount === 0) return "\n  Tools used: 0 (discussion only)";
	return "";
}

export function collapseRoundDetails(roundName: string, roundData: Record<string, unknown>): Record<string, unknown> {
	const collapsedDetails = { ...roundData };
	if (Array.isArray(collapsedDetails.toolCalls)) {
		collapsedDetails.toolCalls = collapsedDetails.toolCalls.map((tc: any) => {
			const sourceText = tc.result_full ?? tc.result_summary ?? "";
			const sizeLabel = formatFileSize(Buffer.byteLength(sourceText, "utf-8"));
			return {
				...tc,
				arguments: `[REDACTED — use get_tool_details("${roundName}", ${tc.index}) to expand]`,
				result_summary: `[REDACTED — size: ${sizeLabel}; use get_tool_details("${roundName}", ${tc.index}) to expand]`,
				result_full: undefined,
			};
		});
	}
	return collapsedDetails;
}

export function renderRoundDetailsToolResult(
	params: RoundDetailsParams,
	roundData: Record<string, unknown>,
): RoundDetailsToolResult {
	const useMatch = params.match !== undefined && params.match.length > 0;
	const useFromLine = params.from_line !== undefined;
	const roundSelection = selectRoundAssistantOutput({
		userPrompt: (roundData.userPrompt as string) ?? "",
		responseSequence: (roundData.responseSequence as string) ?? "",
		assistantOutput: buildRoundAssistantOutput(params.round, roundData),
		fromLine: params.from_line,
		lineCount: params.line_count,
		match: params.match,
		maxMatches: params.max_matches,
	});
	if (!roundSelection.ok) {
		return { content: [{ type: "text", text: roundSelection.error }], details: {} };
	}

	let headerSuffix = "";
	if (useFromLine) {
		headerSuffix = ` (lines ${params.from_line ?? 1}–${Math.min(
			(params.from_line ?? 1) - 1 + (params.line_count ?? 200),
			roundSelection.responseTotalLines,
		)} of ${roundSelection.responseTotalLines})`;
	} else if (useMatch) {
		headerSuffix = roundSelection.matchHeader;
	}

	let parentMeta = "";
	if (roundData.parentId != null) parentMeta += `\nParent round: ${roundData.parentId}`;
	if (roundData.relatedParentId != null)
		parentMeta += `\nRelated to:   ${roundData.relatedParentId} (same topic group)`;

	return {
		content: [
			{
				type: "text",
				text:
					`=== Round: ${params.round}${headerSuffix} ===\n` +
					`User: ${roundData.userPrompt ?? "(empty)"}\n` +
					`Assistant: ${roundSelection.assistantOutput}` +
					`${roundSelection.paginationMarker ? `\n${roundSelection.paginationMarker}` : ""}` +
					`${formatRoundToolMeta(roundData)}` +
					`${parentMeta}`,
			},
		],
		details: collapseRoundDetails(params.round, roundData),
	};
}

export function loadRoundDataForToolDetails(
	fullPath: string,
	roundName: string,
): { ok: true; roundData: RoundData } | { ok: false; result: ToolDetailsToolResult } {
	if (!fs.existsSync(fullPath)) {
		return {
			ok: false,
			result: {
				content: [{ type: "text", text: `Round file not found: ${roundName}` }],
				details: {},
			},
		};
	}

	try {
		return { ok: true, roundData: JSON.parse(fs.readFileSync(fullPath, "utf-8")) };
	} catch {
		return {
			ok: false,
			result: {
				content: [{ type: "text", text: `Failed to parse round file: ${roundName}` }],
				details: {},
			},
		};
	}
}

export function renderToolDetailsToolResult(params: ToolDetailsParams, roundData: RoundData): ToolDetailsToolResult {
	if (!roundData.toolCalls || roundData.toolCalls.length === 0) {
		return {
			content: [
				{
					type: "text",
					text: "This round has no tool calls stored. It may have been indexed before tool call metadata was added. Consider re-indexing.",
				},
			],
			details: {},
		};
	}

	if (params.index < 0 || params.index >= roundData.toolCalls.length) {
		return {
			content: [
				{
					type: "text",
					text: `Invalid index ${params.index}. This round has ${roundData.toolCalls.length} tool calls (indices 0–${roundData.toolCalls.length - 1}).`,
				},
			],
			details: {},
		};
	}

	const tc = roundData.toolCalls[params.index];

	let argsParsed: unknown = tc.arguments;
	try {
		argsParsed = JSON.parse(tc.arguments);
	} catch {
		/* keep as string */
	}

	const resultText = tc.result_full ?? tc.result_summary ?? "";
	const toolSelection = selectToolResultOutput({
		resultText,
		fromLine: params.out__from_line,
		lineCount: params.out_line_count,
		match: params.match,
		maxMatches: params.max_matches,
	});
	if (!toolSelection.ok) {
		return {
			content: [{ type: "text", text: toolSelection.error }],
			details: {},
		};
	}

	if (toolSelection.mode === "page") {
		return {
			content: [
				{
					type: "text",
					text:
						`[Showing lines ${toolSelection.fromLine}–${toolSelection.endLine} of ${toolSelection.totalLines} for tool call #${tc.index} (${tc.name})]\n` +
						`Tool name: ${tc.name}\n` +
						`Arguments: ${JSON.stringify(argsParsed, null, 2)}\n` +
						"Result:\n" +
						`  ${toolSelection.resultBlock}${toolSelection.footer}`,
				},
			],
			details: {
				name: tc.name,
				arguments: tc.arguments,
				lines_shown: {
					from: toolSelection.fromLine,
					to: toolSelection.endLine,
					of: toolSelection.totalLines,
				},
			},
		};
	}

	if (toolSelection.mode === "match") {
		return {
			content: [
				{
					type: "text",
					text:
						`[Match results for tool call #${tc.index} (${tc.name})${toolSelection.matchSummary}]\n` +
						`Tool name: ${tc.name}\n` +
						`Arguments: ${JSON.stringify(argsParsed, null, 2)}\n` +
						"Result:\n" +
						`  ${toolSelection.resultBlock}`,
				},
			],
			details: {
				name: tc.name,
				arguments: tc.arguments,
				matches: { shown: toolSelection.matchCount, total: toolSelection.totalMatches },
			},
		};
	}

	const resultBlock = resultText.length > 0 ? `  ${resultText}` : "  (empty)";
	const truncatedFlag = tc.result_truncated ? "\n\n[Output exceeds storage cap — showing entire stored result]" : "";

	return {
		content: [
			{
				type: "text",
				text:
					`Tool call #${tc.index} in round ${params.round}\n` +
					`  Name: ${tc.name}\n` +
					`  Arguments: ${JSON.stringify(argsParsed, null, 2)}\n` +
					`  Result:\n${resultBlock}${truncatedFlag}`,
			},
		],
		details: { name: tc.name, arguments: tc.arguments, result_full: tc.result_full ?? tc.result_summary },
	};
}
