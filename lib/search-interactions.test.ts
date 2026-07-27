import { describe, expect, it } from "vitest";
import { indexRoundFileFromPath } from "./index-io.ts";
import type { IndexEntry } from "./index-storage.ts";
import type { RoundData } from "./round-data.ts";
import {
	collectSearchRoundScores,
	computeContextBudget,
	filterSearchIndexByRounds,
	normalizeSearchInteractionsParams,
	renderSearchInteractionsToolResult,
	type SearchRoundScore,
	selectContextRounds,
} from "./search-interactions.ts";

describe("normalizeSearchInteractionsParams", () => {
	it("uses defaults for missing optional params", () => {
		const result = normalizeSearchInteractionsParams({ query: "test" });
		expect(result.query).toBe("test");
		expect(result.threshold).toBe(0.25);
		expect(result.scopeRounds).toBeNull();
		expect(result.mode).toBe("similarity");
		expect(result.alpha).toBe(0.7);
	});

	it("preserves all provided params", () => {
		const result = normalizeSearchInteractionsParams({
			query: "something",
			minSimilarity: 0.5,
			rounds: ["abc.json", "def.json"],
			mode: "tool",
			alpha: 0.4,
		});
		expect(result.query).toBe("something");
		expect(result.threshold).toBe(0.5);
		expect(result.scopeRounds).toEqual(["abc.json", "def.json"]);
		expect(result.mode).toBe("tool");
		expect(result.alpha).toBe(0.4);
	});

	it("handles empty query", () => {
		const result = normalizeSearchInteractionsParams({ query: "" });
		expect(result.query).toBeNull();
	});

	it("handles missing query", () => {
		const result = normalizeSearchInteractionsParams({});
		expect(result.query).toBeNull();
	});

	it("handles empty rounds array", () => {
		const result = normalizeSearchInteractionsParams({ rounds: [] });
		expect(result.scopeRounds).toEqual([]);
	});

	it("falls back to similarity for an invalid runtime mode", () => {
		const result = normalizeSearchInteractionsParams({ query: "test", mode: "invalid" as never });
		expect(result.mode).toBe("similarity");
	});

	it.each(["text-match", "hybrid"] as const)("preserves the %s mode", (mode) => {
		const result = normalizeSearchInteractionsParams({ query: "test", mode });
		expect(result.mode).toBe(mode);
	});

	it.each([
		[-0.5, 0],
		[1.5, 1],
	])("clamps alpha %s to %s", (alpha, expected) => {
		expect(normalizeSearchInteractionsParams({ alpha }).alpha).toBe(expected);
	});
});

describe("filterSearchIndexByRounds", () => {
	const index: IndexEntry[] = [
		{ filePath: "rounds/abc.json:prompt", vector: [0.1] },
		{ filePath: "rounds/abc.json:response", vector: [0.2] },
		{ filePath: "rounds/def.json:round", vector: [0.3] },
		{ filePath: "rounds/ghi.json:prompt", vector: [0.4] },
	];

	it("returns full index when scopeRounds is null", () => {
		expect(filterSearchIndexByRounds(index, null)).toEqual(index);
	});

	it("returns full index when scopeRounds is empty", () => {
		expect(filterSearchIndexByRounds(index, [])).toEqual(index);
	});

	it("filters to only matching rounds", () => {
		const result = filterSearchIndexByRounds(index, ["rounds/abc.json"]);
		expect(result).toHaveLength(2);
		expect(result.every((e) => e.filePath.startsWith("rounds/abc.json"))).toBe(true);
	});

	it("handles multiple scope rounds", () => {
		const result = filterSearchIndexByRounds(index, ["rounds/abc.json", "rounds/def.json"]);
		expect(result).toHaveLength(3);
	});

	it("returns empty when scope rounds don't match", () => {
		const result = filterSearchIndexByRounds(index, ["rounds/zzz.json"]);
		expect(result).toHaveLength(0);
	});
});

describe("collectSearchRoundScores", () => {
	const index: IndexEntry[] = [
		{ filePath: "rounds/abc.json:prompt", vector: [1, 0, 0] },
		{ filePath: "rounds/abc.json:response", vector: [0.9, 0.1, 0] },
		{ filePath: "rounds/def.json:round", vector: [0, 1, 0] },
		{ filePath: "rounds/ghi.json:summary", vector: [0, 0, 1] },
	];

	const roundDataMap: Record<string, RoundData> = {
		"rounds/abc.json": {
			userPrompt: "test prompt",
			responseSequence: "test response",
			turnIndex: 0,
		},
		"rounds/def.json": {
			userPrompt: "another prompt",
			responseSequence: "another response",
			turnIndex: 0,
		},
		"rounds/ghi.json": {
			userPrompt: "checkpoint summary input",
			responseSequence: "checkpoint summary output",
			turnIndex: 1,
		},
	};

	function readRound(filePath: string): RoundData | null {
		const roundFile = indexRoundFileFromPath(filePath);
		return roundDataMap[roundFile] ?? null;
	}

	it("scores and returns rounds sorted by similarity", () => {
		const queryVec = [1, 0, 0]; // close to abc
		const results = collectSearchRoundScores(index, queryVec, readRound);
		expect(results).toHaveLength(3);
		// abc should have higher score than def
		expect(results[0].fileName).toBe("rounds/abc.json");
		expect(results[0].bestScore).toBeGreaterThan(results[1].bestScore);
	});

	it("returns empty for empty index", () => {
		const results = collectSearchRoundScores([], [1, 0, 0], readRound);
		expect(results).toHaveLength(0);
	});

	it("skips entries where readRound returns null", () => {
		const emptyRead = () => null as unknown as RoundData;
		const results = collectSearchRoundScores(index, [1, 0, 0], emptyRead);
		expect(results).toHaveLength(0);
	});

	it("takes the best score per round file", () => {
		// abc.json has two entries; prompt vector [1,0,0] matches query better than response [0.9,0.1,0]
		const queryVec = [1, 0, 0];
		const results = collectSearchRoundScores(index, queryVec, readRound);
		const abcEntry = results.find((r) => r.fileName === "rounds/abc.json");
		expect(abcEntry).toBeDefined();
		// Should be 1.0 (exact match with prompt vector)
		expect(abcEntry?.bestScore).toBeCloseTo(1.0, 5);
	});

	it("matches :summary entries", () => {
		// ghi.json:summary vector [0,0,1] should match query [0,0,1] exactly
		const queryVec = [0, 0, 1];
		const results = collectSearchRoundScores(index, queryVec, readRound);
		expect(results).toHaveLength(3);
		const ghiEntry = results.find((r) => r.fileName === "rounds/ghi.json");
		expect(ghiEntry).toBeDefined();
		expect(ghiEntry?.bestScore).toBeCloseTo(1.0, 5);
	});

	it("fuses semantic and BM25 scores so exact keyword matches can outrank vector-only matches", () => {
		const queryVec = [1, 0, 0];
		const localIndex: IndexEntry[] = [
			{ filePath: "rounds/semantic.json:prompt", vector: [1, 0, 0] },
			{ filePath: "rounds/exact.json:prompt", vector: [0.2, 0.8, 0] },
		];
		const localRounds: Record<string, RoundData> = {
			"rounds/semantic.json": {
				userPrompt: "conceptual memory",
				responseSequence: "architecture notes",
				turnIndex: 0,
			},
			"rounds/exact.json": {
				userPrompt: "search_interactions cannot find get_round_details",
				responseSequence: "Investigate native tool names.",
				turnIndex: 0,
			},
		};
		const results = collectSearchRoundScores(
			localIndex,
			queryVec,
			(filePath) => localRounds[indexRoundFileFromPath(filePath)] ?? null,
			{
				bm25Scores: new Map([
					["rounds/semantic.json", 0],
					["rounds/exact.json", 1],
				]),
				semanticWeight: 0.3,
			},
		);

		expect(results.map((result) => result.fileName)).toEqual(["rounds/exact.json", "rounds/semantic.json"]);
		expect(results[0].semanticScore).toBeLessThan(results[1].semanticScore ?? 0);
		expect(results[0].bm25Score).toBe(1);
	});

	it("uses semantic scores unchanged when BM25 has no matches", () => {
		const results = collectSearchRoundScores(index, [1, 0, 0], readRound, {
			bm25Scores: new Map(),
			semanticWeight: 0.3,
		});

		expect(results[0].bestScore).toBeCloseTo(1);
		expect(results[0].fileName).toBe("rounds/abc.json");
	});
});

describe("computeContextBudget", () => {
	it("returns minimum budget for low similarity", () => {
		const budget = computeContextBudget(0.3, 128000, 0.3, 2000);
		expect(budget).toBe(2000);
	});

	it("returns minimum budget for similarity below minSimilarity", () => {
		const budget = computeContextBudget(0.1, 128000, 0.3, 2000);
		expect(budget).toBe(2000);
	});

	it("returns maximum budget for perfect similarity", () => {
		const budget = computeContextBudget(1.0, 128000, 0.3, 2000);
		expect(budget).toBe(64000);
	});

	it("scales budget between min and max proportionally", () => {
		const budget = computeContextBudget(0.65, 128000, 0.3, 2000);
		// t = (0.65 - 0.3) / (1 - 0.3) = 0.5
		// budget = 2000 + 0.5 * (64000 - 2000) = 2000 + 31000 = 33000
		expect(budget).toBe(33000);
	});

	it("uses default parameters when not specified", () => {
		const budget = computeContextBudget(0.9);
		expect(budget).toBeGreaterThan(0);
	});
});

describe("selectContextRounds", () => {
	const makeRound = (name: string, prompt: string, responseLen: number): SearchRoundScore => ({
		fileName: name,
		data: {
			userPrompt: prompt,
			responseSequence: "x".repeat(responseLen),
			turnIndex: 0,
		},
		bestScore: 0.9,
	});

	it("selects rounds within budget", () => {
		const rounds = [makeRound("a.json", "hi", 20), makeRound("b.json", "yo", 30)];
		// With enough budget, both should be selected
		const result = selectContextRounds(rounds, null, () => null as unknown as RoundData, {
			budgetTokens: 10000,
			estimateTokensFn: (text: string) => text.length,
			minSimilarity: 0.3,
		});
		expect(result).toHaveLength(2);
	});

	it("stops when budget is exhausted", () => {
		const rounds = [makeRound("a.json", "hi", 5000), makeRound("b.json", "yo", 5000)];
		const result = selectContextRounds(rounds, null, () => null as unknown as RoundData, {
			budgetTokens: 6000,
			estimateTokensFn: (text: string) => text.length,
			minSimilarity: 0.3,
		});
		expect(result).toHaveLength(1);
	});

	it("skips rounds below minSimilarity", () => {
		const rounds = [
			{ ...makeRound("a.json", "hi", 20), bestScore: 0.5 },
			{ ...makeRound("b.json", "yo", 20), bestScore: 0.2 },
		];
		const result = selectContextRounds(rounds, null, () => null as unknown as RoundData, {
			budgetTokens: 10000,
			estimateTokensFn: (text: string) => text.length,
			minSimilarity: 0.3,
		});
		expect(result).toHaveLength(1);
	});

	it("includes lastRoundFileName at the end with score 0", () => {
		const rounds = [makeRound("a.json", "hi", 20)];
		const lastData: RoundData = {
			userPrompt: "last",
			responseSequence: "last response",
			turnIndex: 0,
		};
		const readRound = (fp: string) => (fp === "last.json" ? lastData : null);
		const result = selectContextRounds(rounds, "last.json", readRound, {
			budgetTokens: 10000,
			estimateTokensFn: (text: string) => text.length,
			minSimilarity: 0.3,
		});
		expect(result[result.length - 1].fileName).toBe("last.json");
		expect(result[result.length - 1].bestScore).toBe(0);
	});

	it("handles null lastRoundFileName", () => {
		const rounds = [makeRound("a.json", "hi", 20)];
		const result = selectContextRounds(rounds, null, () => null as unknown as RoundData, {
			budgetTokens: 10000,
			estimateTokensFn: (text: string) => text.length,
			minSimilarity: 0.3,
		});
		expect(result).toHaveLength(1);
	});

	it("handles empty scored rounds", () => {
		const result = selectContextRounds([], null, () => null as unknown as RoundData, {
			budgetTokens: 10000,
			estimateTokensFn: (text: string) => text.length,
			minSimilarity: 0.3,
		});
		expect(result).toHaveLength(0);
	});
});

describe("renderSearchInteractionsToolResult", () => {
	const makeRoundScore = (overrides: Partial<SearchRoundScore> = {}): SearchRoundScore => ({
		fileName: "rounds/test.json",
		data: {
			userPrompt: "What is the answer?",
			responseSequence: "The answer is 42.",
			turnIndex: 0,
		},
		bestScore: 0.85,
		...overrides,
	});

	it("returns no results message for empty array", () => {
		const result = renderSearchInteractionsToolResult([], 0.3, () => null);
		expect(result.content[0].text).toContain("No matching turns found");
	});

	it("renders round with no tool calls", () => {
		const round = makeRoundScore({
			data: {
				userPrompt: "Hello?",
				responseSequence: "Hi!",
				turnIndex: 0,
				toolCallCount: 0,
			},
		});
		const result = renderSearchInteractionsToolResult([round], 0.3, () => null);
		expect(result.content[0].text).toContain("Found 1 relevant rounds");
		expect(result.content[0].text).toContain("0 tools (discussion only)");
		expect(result.details).toEqual({ matched: 1, topScore: 0.85 });
	});

	it("renders round with tool calls", () => {
		const round = makeRoundScore({
			data: {
				userPrompt: "Run tests",
				responseSequence: "Done",
				turnIndex: 0,
				toolCallCount: 2,
				toolCallNames: ["bash", "edit"],
				toolCalls: [
					{ index: 0, name: "bash", arguments: "{}", result_summary: "ok" },
					{ index: 1, name: "edit", arguments: "{}", result_summary: "ok" },
				],
			},
		});
		const result = renderSearchInteractionsToolResult([round], 0.3, () => "1.2KB");
		expect(result.content[0].text).toContain("2 tools (bash, edit)");
		expect(result.content[0].text).toContain("get_tool_details");
		expect(result.content[0].text).toContain("1.2KB");
	});

	it("renders round with tool calls but no toolCallNames", () => {
		const round = makeRoundScore({
			data: {
				userPrompt: "Run tests",
				responseSequence: "Done",
				turnIndex: 0,
				toolCallCount: 1,
				toolCalls: [{ index: 0, name: "bash", arguments: "{}", result_summary: "ok" }],
			},
		});
		const result = renderSearchInteractionsToolResult([round], 0.3, () => null);
		// When toolCallNames is undefined, (undefined ?? []).join(", ") = ""
		expect(result.content[0].text).toContain("1 tools ()");
	});

	it("limits to 5 rounds", () => {
		const rounds = Array.from({ length: 7 }, (_, i) =>
			makeRoundScore({
				fileName: `rounds/r${i}.json`,
				data: { userPrompt: `Q${i}`, responseSequence: `A${i}`, turnIndex: 0 },
				bestScore: 0.9 - i * 0.05,
			}),
		);
		const result = renderSearchInteractionsToolResult(rounds, 0.3, () => null);
		expect(result.details.matched).toBe(5);
	});

	it("skips rounds below threshold", () => {
		const round = makeRoundScore({ bestScore: 0.2 });
		const result = renderSearchInteractionsToolResult([round], 0.3, () => null);
		// Since the round is below threshold, count should be 0
		expect(result.content[0].text).toContain("No relevant rounds found");
	});

	it("shows no result message when all rounds below threshold", () => {
		const rounds = [makeRoundScore({ bestScore: 0.2 })];
		const result = renderSearchInteractionsToolResult(rounds, 0.3, () => null);
		expect(result.content[0].text).toContain("No relevant rounds found");
	});

	it("handles round with null toolCallCount", () => {
		const round = makeRoundScore({
			data: {
				userPrompt: "Hello?",
				responseSequence: "Hi!",
				turnIndex: 0,
				toolCallCount: undefined,
			},
		});
		const result = renderSearchInteractionsToolResult([round], 0.3, () => null);
		expect(result.content[0].text).not.toContain("tools");
	});

	it("inserts the annotation line when getAnnotationFn returns text", () => {
		const round = makeRoundScore();
		const result = renderSearchInteractionsToolResult(
			[round],
			0.3,
			() => null,
			() => "Matched tool: bash(index 0)",
		);
		expect(result.content[0].text).toContain("Matched tool: bash(index 0)");
	});

	it("omits the annotation line when getAnnotationFn returns null", () => {
		const round = makeRoundScore();
		const result = renderSearchInteractionsToolResult(
			[round],
			0.3,
			() => null,
			() => null,
		);
		expect(result.content[0].text).not.toContain("Matched tool:");
	});

	it("works without a getAnnotationFn (backward compatible)", () => {
		const round = makeRoundScore();
		const result = renderSearchInteractionsToolResult([round], 0.3, () => null);
		expect(result.content[0].text).not.toContain("Matched tool:");
	});
});
