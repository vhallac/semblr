import { describe, expect, it } from "vitest";
import { hasLineSelectionConflict, selectRoundAssistantOutput, selectToolResultOutput } from "./detail-rendering.ts";

describe("detail rendering", () => {
	it("detects conflicting pagination and match selectors", () => {
		expect(hasLineSelectionConflict("needle", 1)).toBe(true);
		expect(hasLineSelectionConflict("", 1)).toBe(false);
		expect(selectRoundAssistantOutput(baseRound({ fromLine: 1, match: "needle" }))).toEqual({
			ok: false,
			error: "Error: match and from_line are mutually exclusive. Use one or the other, not both.",
		});
		expect(selectToolResultOutput({ resultText: "x", fromLine: 1, match: "x" })).toEqual({
			ok: false,
			error: "Error: match and out__from_line are mutually exclusive. Use one or the other, not both.",
		});
	});

	it("paginates round assistant output by 1-based lines", () => {
		expect(selectRoundAssistantOutput(baseRound({ fromLine: 2, lineCount: 2 }))).toEqual({
			ok: true,
			assistantOutput: "beta\ngamma",
			responseTotalLines: 4,
			paginationMarker: "[Truncated — use from_line=4, line_count=2 to continue]",
			matchHeader: "",
		});
	});

	it("matches round assistant output with context and max matches", () => {
		const result = selectRoundAssistantOutput(baseRound({ match: "a$", lineCount: 1, maxMatches: 2 }));

		expect(result).toEqual({
			ok: true,
			assistantOutput:
				"[M 1/2 at assistant line 1 (1 lines of context)]\nalpha\nbeta\n\n" +
				"[M 2/2 at assistant line 2 (1 lines of context)]\nbeta\ngamma",
			responseTotalLines: 4,
			paginationMarker: "",
			matchHeader: " (2 matches shown of 4 total)",
		});
	});

	it("clips round search to one user-prompt match when the user prompt matches", () => {
		const result = selectRoundAssistantOutput(
			baseRound({
				userPrompt: "needle first\nneedle second",
				responseSequence: "needle assistant",
				match: "needle",
				maxMatches: 3,
			}),
		);

		expect(result).toEqual({
			ok: true,
			assistantOutput: "[M 1/1 in user prompt] needle first",
			responseTotalLines: 1,
			paginationMarker: "",
			matchHeader: " (1 match shown of 3 total)",
		});
	});

	it("reports no round matches and invalid regexp patterns", () => {
		expect(selectRoundAssistantOutput(baseRound({ match: "missing" }))).toEqual({
			ok: true,
			assistantOutput: "(no matches)",
			responseTotalLines: 4,
			paginationMarker: "",
			matchHeader: "",
		});
		expect(selectRoundAssistantOutput(baseRound({ match: "[" }))).toMatchObject({
			ok: false,
			error: expect.stringContaining("Invalid regexp pattern:"),
		});
	});

	it("paginates tool result output with continuation footer", () => {
		expect(selectToolResultOutput({ resultText: "one\ntwo\nthree", fromLine: 2, lineCount: 1 })).toEqual({
			ok: true,
			mode: "page",
			resultBlock: "two",
			footer: "\n\n[Truncated — lines remaining: 1. Use out__from_line=3 and out_line_count=1 to continue.]",
			fromLine: 2,
			endLine: 2,
			totalLines: 3,
		});
	});

	it("matches tool results with context, summaries, and no-match output", () => {
		expect(
			selectToolResultOutput({ resultText: "apple\nbanana\napricot", match: "^a", lineCount: 1, maxMatches: 1 }),
		).toEqual({
			ok: true,
			mode: "match",
			resultBlock: "[M 1/1 at line 1 (1 lines of context)]\napple\nbanana",
			matchSummary: " (1 match shown of 2 total)",
			matchCount: 1,
			totalMatches: 2,
		});
		expect(selectToolResultOutput({ resultText: "apple", match: "missing" })).toEqual({
			ok: true,
			mode: "match",
			resultBlock: "(no matches)",
			matchSummary: "",
			matchCount: 0,
			totalMatches: 0,
		});
		expect(selectToolResultOutput({ resultText: "apple", match: "[" })).toMatchObject({
			ok: false,
			error: expect.stringContaining("Invalid regexp pattern:"),
		});
	});

	it("returns full mode when no selector is provided", () => {
		expect(selectToolResultOutput({ resultText: "all" })).toEqual({ ok: true, mode: "full" });
	});
});

function baseRound(overrides: Partial<Parameters<typeof selectRoundAssistantOutput>[0]> = {}) {
	return {
		userPrompt: "user prompt",
		responseSequence: "alpha\nbeta\ngamma\ndelta",
		assistantOutput: "interleaved assistant",
		...overrides,
	};
}
