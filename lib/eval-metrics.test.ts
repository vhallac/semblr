import { describe, expect, it } from "vitest";
import { aggregateQueryMetrics, evaluateQueryMetrics, filterCandidatesAsOf, noiseRate } from "./eval-metrics.ts";
import type { IndexEntry } from "./index-storage.ts";
import { createDefaultStatsState } from "./stats.ts";

function entry(filePath: string): IndexEntry {
	return { filePath, vector: [1, 0] };
}

describe("eval metrics", () => {
	it("filters candidates strictly earlier than the query and excludes the query round itself", () => {
		const entries = [
			entry("older.json:prompt"),
			entry("older.json:response"),
			entry("same-time.json:prompt"),
			entry("query.json:prompt"),
			entry("missing.json:response"),
		];
		const rounds = new Map([
			["older.json", { userTimestamp: 10 }],
			["same-time.json", { userTimestamp: 20 }],
			["query.json", { userTimestamp: 20 }],
		]);

		expect(filterCandidatesAsOf(entries, rounds, "query.json", 20)).toEqual({
			entries: [entry("older.json:prompt"), entry("older.json:response")],
			excludedMissingTimestamp: 1,
		});
	});

	it("computes per-query hit@5, recall@5, and mrr across boundary ranks", () => {
		expect(
			evaluateQueryMetrics({
				labels: ["target-a", "target-b"],
				ranked: ["x", "target-a", "y", "z", "target-b", "late"],
			}),
		).toEqual({ hitAt5: 1, recallAt5: 1, mrr: 0.5, firstHitRank: 2 });

		expect(evaluateQueryMetrics({ labels: ["target"], ranked: ["a", "b", "c", "d", "e", "target"] })).toEqual({
			hitAt5: 0,
			recallAt5: 0,
			mrr: 1 / 6,
			firstHitRank: 6,
		});
	});

	it("aggregates means across non-empty-label queries and counts empty-label queries separately", () => {
		expect(
			aggregateQueryMetrics([
				{ labels: ["a"], ranked: ["a"] },
				{ labels: ["b", "c"], ranked: ["x", "b", "y", "z", "c"] },
				{ labels: [], ranked: ["ignored"] },
			]),
		).toEqual({
			queryCount: 2,
			emptyLabelCount: 1,
			hitAt5: 1,
			recallAt5: 1,
			mrr: 0.75,
		});
	});

	it("computes noise rate from chain-read stats and returns null when nothing was presented", () => {
		const stats = createDefaultStatsState("now", 3);
		stats.positionScores[0].presentedCount = 4;
		stats.positionScores[0].readCount = 1;
		stats.positionScores[1].presentedCount = 2;
		stats.positionScores[1].readCount = 1;
		stats.positionScores[2].presentedCount = 0;
		stats.positionScores[2].readCount = 0;

		expect(noiseRate(stats)).toBe(1 - 2 / 6);
		expect(noiseRate(createDefaultStatsState("now", 2))).toBeNull();
	});
});
