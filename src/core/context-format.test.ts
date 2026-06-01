import { describe, expect, it } from "vitest";
import {
	buildContextPreamble,
	buildGroupedRecencyList,
	buildRelevanceList,
	formatFileSize,
	formatGroupedRoundEntry,
	formatRoundEntry,
	splitCommandArgs,
} from "./context-format.ts";

describe("context formatting", () => {
	it("formats relevance and grouped entries with multiline prompts and optional size", () => {
		expect(formatRoundEntry(2, "abc.json", "0.91", "1 tools", "first\nsecond", "2KB")).toEqual([
			"2. abc.json [0.91 | 1 tools | 2KB]:",
			"  user: first",
			"  second",
			"  ---",
		]);
		expect(formatGroupedRoundEntry(1, "abc.json", "0 tools", "prompt")).toEqual([
			"- [index: 1] abc.json [n/a | 0 tools]:",
			"  user: prompt",
			"  ---",
		]);
	});

	it("builds grouped recency lists newest topic first with stable global indices", () => {
		const older = { fileName: "older.json", userPrompt: "older", responseSequence: "", toolSummary: "0 tools" };
		const newerA = { fileName: "new-a.json", userPrompt: "newer A", responseSequence: "", toolSummary: "1 tools" };
		const newerB = { fileName: "new-b.json", userPrompt: "newer B", responseSequence: "", toolSummary: "2 tools" };
		const causalChain = [older, newerA, newerB];
		const list = buildGroupedRecencyList([{ rounds: [older] }, { rounds: [newerA, newerB] }], causalChain, (file) =>
			file === "new-b.json" ? "4KB" : null,
		);

		expect(list).toContain("--- RECENCY LIST (current session, by topic) ---");
		expect(list).toContain("**Group 1**\n\n- [index: 1] new-b.json [n/a | 2 tools | 4KB]:");
		expect(list).toContain("- [index: 2] new-a.json [n/a | 1 tools]:");
		expect(list).toContain("**Group 2**\n\n- [index: 3] older.json [n/a | 0 tools]:");
		expect(buildGroupedRecencyList([], [])).toBeNull();
	});

	it("builds relevance lists with score, size, and per-tool result sizes", () => {
		const list = buildRelevanceList(
			[
				{
					fileName: "round.json",
					bestScore: 0.876,
					data: {
						userPrompt: "Prompt",
						toolCallCount: 2,
						toolCalls: [
							{ name: "read", result_full: "x".repeat(1024) },
							{ name: "grep", result_summary: "" },
						],
					},
				},
			],
			() => "10KB",
		);

		expect(list).toContain("--- RELEVANCE LIST (all sessions, by similarity) ---");
		expect(list).toContain("1. round.json [0.88 | 2 tools (read (1KB), grep) | 10KB]:");
		expect(buildRelevanceList([])).toBeNull();
	});

	it("builds context preamble only when at least one list exists", () => {
		expect(buildContextPreamble(false, false)).toBeNull();
		expect(buildContextPreamble(true, false)).toContain("[CONTEXT BUILDING REFERENCES]");
		expect(buildContextPreamble(false, true)).toContain("get_round_details");
	});

	it("formats byte counts", () => {
		expect(formatFileSize(512)).toBe("0.5KB");
		expect(formatFileSize(5 * 1024)).toBe("5KB");
		expect(formatFileSize(54 * 1024)).toBe("54KB");
		expect(formatFileSize(1_153_434)).toBe("1.1MB");
	});

	it("splits command arguments with quotes and escapes", () => {
		expect(splitCommandArgs("--dry-run --limit 10")).toEqual(["--dry-run", "--limit", "10"]);
		expect(splitCommandArgs("--name 'two words' \"more words\"")).toEqual(["--name", "two words", "more words"]);
		expect(splitCommandArgs("escaped\\ space trailing\\")).toEqual(["escaped space", "trailing\\"]);
		expect(splitCommandArgs("   ")).toEqual([]);
	});
});
