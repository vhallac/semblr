import { indexRoundFileFromPath } from "./index-io.ts";
import type { IndexEntry } from "./index-storage.ts";
import type { RoundData } from "./round-data.ts";
import type { ChainReadStatsState } from "./stats.ts";

export interface AsOfFilterResult {
	entries: IndexEntry[];
	excludedMissingTimestamp: number;
}

export function filterCandidatesAsOf(
	entries: readonly IndexEntry[],
	rounds: ReadonlyMap<string, Pick<RoundData, "userTimestamp">>,
	queryRoundFile: string,
	queryTimestamp: number,
): AsOfFilterResult {
	const eligibleRoundFiles = new Map<string, boolean>();
	let excludedMissingTimestamp = 0;

	for (const entry of entries) {
		const roundFile = indexRoundFileFromPath(entry.filePath);
		if (eligibleRoundFiles.has(roundFile)) continue;
		if (roundFile === queryRoundFile) {
			eligibleRoundFiles.set(roundFile, false);
			continue;
		}
		const userTimestamp = rounds.get(roundFile)?.userTimestamp;
		if (typeof userTimestamp !== "number") {
			excludedMissingTimestamp++;
			eligibleRoundFiles.set(roundFile, false);
			continue;
		}
		eligibleRoundFiles.set(roundFile, userTimestamp < queryTimestamp);
	}

	return {
		entries: entries.filter((entry) => eligibleRoundFiles.get(indexRoundFileFromPath(entry.filePath)) === true),
		excludedMissingTimestamp,
	};
}

export interface QueryRanking {
	labels: string[];
	ranked: string[];
}

export interface QueryMetrics {
	hitAt5: number;
	recallAt5: number;
	mrr: number;
	firstHitRank: number | null;
}

export interface AggregateMetrics {
	queryCount: number;
	emptyLabelCount: number;
	hitAt5: number;
	recallAt5: number;
	mrr: number;
}

export function evaluateQueryMetrics({ labels, ranked }: QueryRanking): QueryMetrics {
	const top5 = ranked.slice(0, 5);
	const top5Set = new Set(top5);
	const firstHitIndex = ranked.findIndex((candidate) => labels.includes(candidate));
	const hits = labels.filter((label) => top5Set.has(label)).length;

	return {
		hitAt5: hits > 0 ? 1 : 0,
		recallAt5: labels.length > 0 ? hits / labels.length : 0,
		mrr: firstHitIndex >= 0 ? 1 / (firstHitIndex + 1) : 0,
		firstHitRank: firstHitIndex >= 0 ? firstHitIndex + 1 : null,
	};
}

export function aggregateQueryMetrics(queries: readonly QueryRanking[]): AggregateMetrics {
	const withLabels = queries.filter((query) => query.labels.length > 0);
	const sums = withLabels.reduce(
		(acc, query) => {
			const metrics = evaluateQueryMetrics(query);
			acc.hitAt5 += metrics.hitAt5;
			acc.recallAt5 += metrics.recallAt5;
			acc.mrr += metrics.mrr;
			return acc;
		},
		{ hitAt5: 0, recallAt5: 0, mrr: 0 },
	);

	return {
		queryCount: withLabels.length,
		emptyLabelCount: queries.length - withLabels.length,
		hitAt5: withLabels.length > 0 ? sums.hitAt5 / withLabels.length : 0,
		recallAt5: withLabels.length > 0 ? sums.recallAt5 / withLabels.length : 0,
		mrr: withLabels.length > 0 ? sums.mrr / withLabels.length : 0,
	};
}

export function noiseRate(stats: ChainReadStatsState): number | null {
	const presented = stats.positionScores.reduce((sum, position) => sum + position.presentedCount, 0);
	if (presented === 0) return null;
	const read = stats.positionScores.reduce((sum, position) => sum + position.readCount, 0);
	return 1 - read / presented;
}
