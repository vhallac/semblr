import { formatFileSize } from "./context-format.ts";
import { indexRoundFileFromPath } from "./index-io.ts";
import type { IndexEntry } from "./index-storage.ts";
import type { RoundData, ToolCallDetail, ToolResult } from "./round-data.ts";
import { estimateTokens } from "./tokens.ts";
import { cosineSimilarity, normalize } from "./vector.ts";

const DEFAULT_CONTEXT_BUDGET_RATIO = 0.5;

export type SearchInteractionsMode = "similarity" | "text-match" | "hybrid" | "tool";

export interface SearchInteractionsParams {
	query?: string;
	minSimilarity?: number;
	rounds?: string[];
	mode?: SearchInteractionsMode;
	alpha?: number;
}

export interface NormalizedSearchInteractionsParams {
	query: string | null;
	threshold: number;
	scopeRounds: string[] | null;
	mode: SearchInteractionsMode;
	alpha: number;
}

export interface SearchRoundScore {
	fileName: string;
	data: RoundData;
	bestScore: number;
	semanticScore?: number;
	bm25Score?: number;
}

export type SearchInteractionsToolResult = ToolResult;

export function normalizeSearchInteractionsParams(
	params: SearchInteractionsParams,
): NormalizedSearchInteractionsParams {
	const validModes: readonly SearchInteractionsMode[] = ["similarity", "text-match", "hybrid", "tool"];
	return {
		query: params.query || null,
		threshold: params.minSimilarity ?? 0.25,
		scopeRounds: params.rounds ?? null,
		mode: validModes.includes(params.mode as SearchInteractionsMode)
			? (params.mode as SearchInteractionsMode)
			: "similarity",
		alpha: Math.max(0, Math.min(1, params.alpha ?? 0.7)),
	};
}

export function filterSearchIndexByRounds(
	index: readonly IndexEntry[],
	scopeRounds: readonly string[] | null,
): IndexEntry[] {
	if (!scopeRounds || scopeRounds.length === 0) return [...index];
	const scopeSet = new Set(scopeRounds);
	return index.filter((entry) => {
		const roundFile = indexRoundFileFromPath(entry.filePath);
		return scopeSet.has(roundFile);
	});
}

export function getIndexEmbeddingModels(index: readonly IndexEntry[], fallbackModel: string): string[] {
	return [...new Set(index.map((entry) => entry.model ?? fallbackModel))];
}

export interface PreparedMultiModelQueryVectors {
	scopedIndex: IndexEntry[];
	queryVectorsByModel: Map<string, number[]>;
	/** Models whose query embedding call rejected, with the reason — never swallowed. */
	failedModels: Array<{ model: string; reason: string }>;
}

export async function prepareMultiModelQueryVectors(
	index: readonly IndexEntry[],
	scopeRounds: readonly string[] | null,
	query: string,
	fallbackModel: string,
	embedQuery: (query: string, model: string) => Promise<number[]>,
): Promise<PreparedMultiModelQueryVectors> {
	const scopedIndex = filterSearchIndexByRounds(index, scopeRounds);
	const models = getIndexEmbeddingModels(scopedIndex, fallbackModel);
	const embeddingResults = await Promise.allSettled(
		models.map(async (model) => [model, normalize(await embedQuery(query, model))] as const),
	);
	const queryVectorsByModel = new Map<string, number[]>();
	const failedModels: Array<{ model: string; reason: string }> = [];
	for (let i = 0; i < models.length; i++) {
		const result = embeddingResults[i];
		if (result.status === "fulfilled") {
			queryVectorsByModel.set(result.value[0], result.value[1]);
		} else {
			failedModels.push({ model: models[i], reason: embeddingFailureReason(result.reason) });
		}
	}
	return { scopedIndex, queryVectorsByModel, failedModels };
}

function embeddingFailureReason(reason: unknown): string {
	return reason instanceof Error ? reason.message : String(reason);
}

function collectSearchRoundScoresWithVector(
	index: readonly IndexEntry[],
	queryVectorForEntry: (entry: IndexEntry) => number[] | null,
	readRound: (filePath: string) => RoundData | null,
	options: {
		bm25Scores?: ReadonlyMap<string, number>;
		semanticWeight?: number;
	} = {},
): SearchRoundScore[] {
	const semanticWeight =
		options.bm25Scores && options.bm25Scores.size > 0 ? Math.max(0, Math.min(1, options.semanticWeight ?? 0.7)) : 1;
	const scored = index
		.map((entry) => {
			const queryVector = queryVectorForEntry(entry);
			if (!queryVector) return null;
			const semanticScore = cosineSimilarity(queryVector, entry.vector);
			const bm25Score = options.bm25Scores?.get(indexRoundFileFromPath(entry.filePath)) ?? 0;
			const similarity = semanticWeight * semanticScore + (1 - semanticWeight) * bm25Score;
			return { ...entry, similarity, semanticScore, bm25Score };
		})
		.filter((scoredEntry): scoredEntry is NonNullable<typeof scoredEntry> => scoredEntry !== null)
		.sort((a, b) => b.similarity - a.similarity);

	const roundScores = new Map<string, SearchRoundScore>();
	for (const entry of scored) {
		const roundFile = indexRoundFileFromPath(entry.filePath);
		if (!roundFile.endsWith(".json")) continue;
		if (roundScores.has(roundFile)) continue;
		const roundData = readRound(entry.filePath);
		if (!roundData) continue;
		roundScores.set(roundFile, {
			fileName: roundFile,
			data: roundData,
			bestScore: entry.similarity,
			semanticScore: entry.semanticScore,
			bm25Score: entry.bm25Score,
		});
	}

	return Array.from(roundScores.values()).sort((a, b) => b.bestScore - a.bestScore);
}

export function collectSearchRoundScores(
	index: readonly IndexEntry[],
	queryVec: number[],
	readRound: (filePath: string) => RoundData | null,
	options: {
		bm25Scores?: ReadonlyMap<string, number>;
		semanticWeight?: number;
	} = {},
): SearchRoundScore[] {
	return collectSearchRoundScoresWithVector(index, () => queryVec, readRound, options);
}

export function collectMultiModelSearchRoundScores(
	index: readonly IndexEntry[],
	queryVectorsByModel: ReadonlyMap<string, number[]>,
	fallbackModel: string,
	readRound: (filePath: string) => RoundData | null,
	options: {
		bm25Scores?: ReadonlyMap<string, number>;
		semanticWeight?: number;
	} = {},
): SearchRoundScore[] {
	return collectSearchRoundScoresWithVector(
		index,
		(entry) => {
			const model = entry.model ?? fallbackModel;
			return queryVectorsByModel.get(model) ?? null;
		},
		readRound,
		options,
	);
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
	getAnnotationFn?: (fileName: string) => string | null,
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

		const annotation = getAnnotationFn?.(round.fileName);
		if (annotation) lines.push(annotation);

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
