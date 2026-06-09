import { formatFileSize } from "./context-format.ts";
import type { IndexEntry } from "./index-storage.ts";
import type { RoundData, ToolCallDetail, ToolResult } from "./round-data.ts";
import { estimateTokens } from "./tokens.ts";
import { cosineSimilarity } from "./vector.ts";

const DEFAULT_CONTEXT_BUDGET_RATIO = 0.5;

export interface SearchInteractionsParams {
	query?: string;
	minSimilarity?: number;
	rounds?: string[];
}

export interface NormalizedSearchInteractionsParams {
	query: string | null;
	threshold: number;
	scopeRounds: string[] | null;
}

export interface SearchRoundScore {
	fileName: string;
	data: RoundData;
	bestScore: number;
}

export type SearchInteractionsToolResult = ToolResult;

export function normalizeSearchInteractionsParams(
	params: SearchInteractionsParams,
): NormalizedSearchInteractionsParams {
	return {
		query: params.query || null,
		threshold: params.minSimilarity ?? 0.25,
		scopeRounds: params.rounds ?? null,
	};
}

export function filterSearchIndexByRounds(
	index: readonly IndexEntry[],
	scopeRounds: readonly string[] | null,
): IndexEntry[] {
	if (!scopeRounds || scopeRounds.length === 0) return [...index];
	const scopeSet = new Set(scopeRounds);
	return index.filter((entry) => {
		const roundFile = entry.filePath.replace(/(:prompt|:response|:round|:summary)$/, "");
		return scopeSet.has(roundFile);
	});
}

export function collectSearchRoundScores(
	index: readonly IndexEntry[],
	queryVec: number[],
	readRound: (filePath: string) => RoundData | null,
): SearchRoundScore[] {
	const scored = index
		.map((entry) => ({ ...entry, similarity: cosineSimilarity(queryVec, entry.vector) }))
		.sort((a, b) => b.similarity - a.similarity);

	const roundScores = new Map<string, SearchRoundScore>();
	for (const entry of scored) {
		const roundFile = entry.filePath.replace(/(:prompt|:response|:round|:summary)$/, "");
		if (!roundFile.endsWith(".json")) continue;
		if (roundScores.has(roundFile)) continue;
		const roundData = readRound(entry.filePath);
		if (!roundData) continue;
		roundScores.set(roundFile, { fileName: roundFile, data: roundData, bestScore: entry.similarity });
	}

	return Array.from(roundScores.values()).sort((a, b) => b.bestScore - a.bestScore);
}

export function computeContextBudget(
	bestScore: number,
	contextWindow = 128_000,
	minSimilarity = 0.3,
	minBudget = 2000,
	budgetRatio = DEFAULT_CONTEXT_BUDGET_RATIO,
): number {
	const maxBudget = Math.floor(budgetRatio * contextWindow);
	const t = Math.max(0, Math.min(1, (bestScore - minSimilarity) / (1 - minSimilarity)));
	return Math.floor(minBudget + t * (maxBudget - minBudget));
}

export function selectContextRounds(
	scoredRounds: readonly SearchRoundScore[],
	lastRoundFileName: string | null,
	readRound: (filePath: string) => RoundData | null,
	options: {
		minSimilarity?: number;
		budgetTokens: number;
		estimateTokensFn?: (text: string) => number;
	} = { budgetTokens: 2000 },
): SearchRoundScore[] {
	const minSimilarity = options.minSimilarity ?? 0.3;
	const estimateTokensFn = options.estimateTokensFn ?? estimateTokens;
	const selectedRounds: SearchRoundScore[] = [];
	let usedTokens = 0;

	for (const round of scoredRounds) {
		if (round.bestScore < minSimilarity) break;
		const roundTokens = estimateTokensFn(round.data.userPrompt + round.data.responseSequence);
		if (usedTokens + roundTokens > options.budgetTokens) break;
		selectedRounds.push(round);
		usedTokens += roundTokens;
	}

	if (lastRoundFileName) {
		const lastData = readRound(lastRoundFileName);
		if (lastData) selectedRounds.push({ data: lastData, fileName: lastRoundFileName, bestScore: 0 });
	}

	return selectedRounds;
}

export function renderSearchInteractionsToolResult(
	sorted: readonly SearchRoundScore[],
	threshold: number,
	getRoundSizeFn: (fileName: string) => string | null,
): SearchInteractionsToolResult {
	if (sorted.length === 0) {
		return {
			content: [{ type: "text", text: "No matching turns found in the index." }],
			details: {},
		};
	}

	const lines: string[] = [];
	let count = 0;
	for (const round of sorted) {
		if (round.bestScore < threshold) break;
		if (count >= 5) break;
		count++;
		const toolStr =
			round.data.toolCallCount != null && round.data.toolCallCount > 0
				? ` | ${round.data.toolCallCount} tools (${(round.data.toolCallNames ?? []).join(", ")})`
				: round.data.toolCallCount === 0
					? " | 0 tools (discussion only)"
					: "";

		const roundSizeStr = getRoundSizeFn(round.fileName);
		const sizeTag = roundSizeStr ? ` | ${roundSizeStr}` : "";
		lines.push(`--- Round ${round.fileName} (score: ${round.bestScore.toFixed(3)}${toolStr}${sizeTag}) ---`);
		lines.push(`User: ${round.data.userPrompt}`);

		if (
			round.data.toolCallCount != null &&
			round.data.toolCallCount > 0 &&
			round.data.toolCalls &&
			round.data.toolCalls.length > 0
		) {
			const turnLines = round.data.toolCalls.map((tc: ToolCallDetail) => {
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
			content: [
				{
					type: "text",
					text: `No relevant rounds found (best score: ${sorted[0].bestScore.toFixed(3)}).`,
				},
			],
			details: {},
		};
	}

	return {
		content: [{ type: "text", text: `Found ${count} relevant rounds:\n\n${lines.join("\n")}` }],
		details: { matched: count, topScore: sorted[0].bestScore },
	};
}
