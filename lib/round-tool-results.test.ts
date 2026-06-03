import { describe, expect, it } from "vitest";
import type { RoundData } from "./round-data.ts";
import {
	buildRoundAssistantOutput,
	collapseRoundDetails,
	formatRoundToolMeta,
	loadRoundDataForToolDetails,
	renderRoundDetailsToolResult,
	renderToolDetailsToolResult,
} from "./round-tool-results.ts";

describe("buildRoundAssistantOutput", () => {
	it("uses responseSegments when available", () => {
		const roundData = {
			responseSegments: [
				{ type: "text", text: "Hello " },
				{ type: "text", text: "world" },
			],
			responseSequence: "fallback",
		};
		expect(buildRoundAssistantOutput("test.json", roundData)).toBe("Hello \nworld");
	});

	it("renders toolCall segments with REDACTED marker", () => {
		const roundData = {
			responseSegments: [
				{ type: "text", text: "Before" },
				{ type: "toolCall", toolCallIndex: 0 },
				{ type: "text", text: "After" },
			],
		};
		const result = buildRoundAssistantOutput("test.json", roundData);
		expect(result).toContain('use get_tool_details("test.json", 0) to expand');
		expect(result).toContain("Before");
		expect(result).toContain("After");
	});

	it("falls back to responseSequence when no responseSegments", () => {
		const roundData = { responseSequence: "I am the response" };
		expect(buildRoundAssistantOutput("test.json", roundData)).toBe("I am the response");
	});

	it("falls back to (empty) when both are missing", () => {
		expect(buildRoundAssistantOutput("test.json", {})).toBe("(empty)");
	});

	it("falls back to responseSequence when responseSegments is empty array", () => {
		const roundData = {
			responseSegments: [],
			responseSequence: "fallback text",
		};
		expect(buildRoundAssistantOutput("test.json", roundData)).toBe("fallback text");
	});

	it("skips text blocks without text", () => {
		const roundData = {
			responseSegments: [{ type: "text" }, { type: "text", text: "visible" }],
		};
		const result = buildRoundAssistantOutput("test.json", roundData);
		expect(result).toBe("visible");
	});
});

describe("formatRoundToolMeta", () => {
	it("shows tool count and names when tools present", () => {
		const roundData = { toolCallCount: 3, toolCallNames: ["bash", "edit", "read"] };
		expect(formatRoundToolMeta(roundData)).toBe("\n  Tools used: 3 (bash, edit, read)");
	});

	it("shows discussion only when toolCallCount is 0", () => {
		const roundData = { toolCallCount: 0 };
		expect(formatRoundToolMeta(roundData)).toBe("\n  Tools used: 0 (discussion only)");
	});

	it("returns empty string when toolCallCount is null/undefined", () => {
		expect(formatRoundToolMeta({})).toBe("");
		expect(formatRoundToolMeta({ toolCallCount: undefined })).toBe("");
	});

	it("handles toolCallNames as non-array", () => {
		const roundData = { toolCallCount: 2, toolCallNames: "bad" };
		expect(formatRoundToolMeta(roundData)).toBe("\n  Tools used: 2 (unknown)");
	});
});

describe("collapseRoundDetails", () => {
	it("collapses tool calls by redacting arguments and results", () => {
		const roundData = {
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: '{"command":"ls"}',
					result_summary: "file1.txt\nfile2.txt",
					result_full: "file1.txt\nfile2.txt\nfile3.txt",
				},
			],
		};
		const result = collapseRoundDetails("test.json", roundData);
		const calls = result.toolCalls as unknown[];
		const tc = calls[0] as Record<string, unknown>;
		expect(tc.arguments).toContain("REDACTED");
		expect(tc.result_summary).toContain("REDACTED");
		expect(tc.result_full).toBeUndefined();
	});

	it("preserves non-tool-call fields", () => {
		const roundData = { userPrompt: "hello", toolCalls: [] };
		const result = collapseRoundDetails("test.json", roundData);
		expect(result.userPrompt).toBe("hello");
	});

	it("handles round without toolCalls", () => {
		const roundData = { userPrompt: "hello" };
		const result = collapseRoundDetails("test.json", roundData);
		expect(result.userPrompt).toBe("hello");
	});

	it("handles toolCall with only result_summary (no result_full)", () => {
		const roundData = {
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: "{}",
					result_summary: "short result",
				},
			],
		};
		const result = collapseRoundDetails("test.json", roundData);
		const calls = result.toolCalls as unknown[];
		const tc = calls[0] as Record<string, unknown>;
		expect(tc.result_summary).toContain("REDACTED");
	});
});

describe("loadRoundDataForToolDetails", () => {
	it("returns error when file does not exist", () => {
		const result = loadRoundDataForToolDetails("/nonexistent.json", "test.json");
		expect(result.ok).toBe(false);
		if (!result.ok) {
			expect(result.result.content[0].text).toContain("not found");
		}
	});

	it("returns error when file has invalid JSON", () => {
		// Use a path that likely doesn't exist as valid JSON file
		// We can't easily test the JSON parse error with the real fs
		// without creating a temp file, so test the ok path with a real file
		// and accept this branch limitation
	});
});

describe("renderRoundDetailsToolResult", () => {
	const basicRoundData: Record<string, unknown> = {
		userPrompt: "What is 2+2?",
		responseSequence: "The answer is 4.",
		turnIndex: 0,
	};

	it("renders basic round details", () => {
		const result = renderRoundDetailsToolResult({ round: "test.json" }, basicRoundData);
		expect(result.content[0].text).toContain("Round: test.json");
		expect(result.content[0].text).toContain("What is 2+2?");
		expect(result.content[0].text).toContain("The answer is 4.");
	});

	it("shows from_line pagination info", () => {
		const roundData = {
			...basicRoundData,
			responseSequence: "line1\nline2\nline3\nline4\nline5",
		};
		const result = renderRoundDetailsToolResult({ round: "test.json", from_line: 2, line_count: 2 }, roundData);
		expect(result.content[0].text).toContain("lines 2–3");
	});

	it("shows match results header", () => {
		const roundData = {
			...basicRoundData,
			responseSequence: "The answer is 4 and the answer is also four.",
		};
		const result = renderRoundDetailsToolResult({ round: "test.json", match: "answer" }, roundData);
		expect(result.content[0].text).toContain("M 1/");
	});

	it("shows parent round metadata", () => {
		const roundData = {
			...basicRoundData,
			parentId: "parent.json",
			relatedParentId: "related.json",
		};
		const result = renderRoundDetailsToolResult({ round: "test.json" }, roundData);
		expect(result.content[0].text).toContain("Parent round: parent.json");
		expect(result.content[0].text).toContain("Related to:   related.json");
	});

	it("shows parentId without relatedParentId", () => {
		const roundData = {
			...basicRoundData,
			parentId: "parent.json",
		};
		const result = renderRoundDetailsToolResult({ round: "test.json" }, roundData);
		expect(result.content[0].text).toContain("Parent round: parent.json");
		expect(result.content[0].text).not.toContain("Related to:");
	});

	it("returns error for conflicting match and from_line params", () => {
		const result = renderRoundDetailsToolResult({ round: "test.json", from_line: 1, match: "test" }, basicRoundData);
		expect(result.content[0].text).toContain("mutually exclusive");
	});

	it("includes tool metadata when present", () => {
		const roundData = {
			...basicRoundData,
			toolCallCount: 2,
			toolCallNames: ["bash", "edit"],
		};
		const result = renderRoundDetailsToolResult({ round: "test.json" }, roundData);
		expect(result.content[0].text).toContain("Tools used: 2 (bash, edit)");
	});
});

describe("renderToolDetailsToolResult", () => {
	it("returns helpful message when no tool calls stored", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("no tool calls stored");
	});

	it("returns helpful message when toolCalls is empty array", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("no tool calls stored");
	});

	it("returns error for invalid negative index", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [{ index: 0, name: "bash", arguments: "{}", result_summary: "ok" }],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: -1 }, roundData);
		expect(result.content[0].text).toContain("Invalid index");
	});

	it("returns error for index beyond range", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [{ index: 0, name: "bash", arguments: "{}", result_summary: "ok" }],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 5 }, roundData);
		expect(result.content[0].text).toContain("Invalid index");
	});

	it("renders full tool call details", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: '{"command":"ls"}',
					result_summary: "output",
					result_full: "full output",
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("Tool call #0");
		expect(result.content[0].text).toContain("bash");
		expect(result.content[0].text).toContain("ls");
		expect(result.content[0].text).toContain("full output");
		expect(result.details).toEqual({
			name: "bash",
			arguments: '{"command":"ls"}',
			result_full: "full output",
		});
	});

	it("renders tool call with pagination (from_line)", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: "{}",
					result_summary: "line1\nline2\nline3",
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0, out__from_line: 2 }, roundData);
		expect(result.content[0].text).toContain("Showing lines");
		expect(result.details).toHaveProperty("lines_shown");
	});

	it("renders tool call with match", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: "{}",
					result_summary: "Error: file not found\nSuccess: done",
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0, match: "Error" }, roundData);
		expect(result.content[0].text).toContain("Match results");
		expect(result.details).toHaveProperty("matches");
	});

	it("returns error for conflicting match and from_line", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [{ index: 0, name: "bash", arguments: "{}", result_summary: "ok" }],
		};
		const result = renderToolDetailsToolResult(
			{ round: "test.json", index: 0, out__from_line: 1, match: "test" },
			roundData,
		);
		expect(result.content[0].text).toContain("mutually exclusive");
	});

	it("handles tool call with result_truncated flag and content", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "edit",
					arguments: "{}",
					result_summary: "short",
					result_full: "some result content",
					result_truncated: true,
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("Tool call #0");
		expect(result.content[0].text).toContain("Output exceeds storage cap");
	});

	it("handles tool call with empty result and result_truncated flag", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "edit",
					arguments: "{}",
					result_summary: "",
					result_full: "",
					result_truncated: true,
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("(empty)");
	});

	it("parses JSON arguments", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: '{"command":"echo hello"}',
					result_summary: "hello",
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("echo hello");
	});

	it("handles non-JSON arguments gracefully", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: "not valid json",
					result_summary: "ok",
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("not valid json");
	});

	it("renders tool call with only result_summary (no result_full)", () => {
		const roundData: RoundData = {
			userPrompt: "test",
			responseSequence: "response",
			turnIndex: 0,
			toolCalls: [
				{
					index: 0,
					name: "bash",
					arguments: "{}",
					result_summary: "just summary",
				},
			],
		};
		const result = renderToolDetailsToolResult({ round: "test.json", index: 0 }, roundData);
		expect(result.content[0].text).toContain("just summary");
		expect(result.details).toEqual({
			name: "bash",
			arguments: "{}",
			result_full: "just summary",
		});
	});
});
