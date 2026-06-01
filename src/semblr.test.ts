import { describe, expect, it } from "vitest";
import {
	buildContextMessagePrefix,
	buildRoundAssistantOutput,
	collapseRoundDetails,
	countWordsInMessageContent,
	extractContextPrompt,
	formatRoundToolMeta,
	prepareContextMessages,
	renderRoundDetailsToolResult,
	shouldDropRelevanceList,
	startsWithEnvironmentPreamble,
} from "./semblr.ts";

const env = "[ENVIRONMENT]\nHost: test\nCWD: /repo\nCurrent date/time: 20260101T000000Z";

describe("semblr context hook join points", () => {
	it("prepares current-round messages from the last user prompt and preserves the system message", () => {
		const system = { role: "system", content: "rules" };
		const prepared = prepareContextMessages(
			[
				system,
				{ role: "user", content: "old prompt" },
				{ role: "assistant", content: "old answer" },
				{ role: "user", content: "current prompt" },
				{ role: "assistant", content: "partial answer" },
			],
			env,
		);

		expect(prepared.systemMsg).toBe(system);
		expect(prepared.hasUserMessage).toBe(true);
		expect(prepared.rawPromptWordCount).toBe(2);
		expect(prepared.currentMessages).toHaveLength(2);
		expect((prepared.currentMessages[0] as { content: string }).content).toBe(`${env}\n\ncurrent prompt`);
		expect(prepared.userPrompt).toBe(`${env}\n\ncurrent prompt`);
	});

	it("does not prepend duplicate environment blocks to replayed string prompts", () => {
		const content = `  ${env}\n\nalready augmented`;
		const prepared = prepareContextMessages([{ role: "user", content }], env);

		expect((prepared.augmentedMessages[0] as { content: string }).content).toBe(content);
		expect(prepared.userPrompt).toBe(content);
		expect(startsWithEnvironmentPreamble(content)).toBe(true);
	});

	it("prepends environment blocks to text-array prompts and extracts text-only prompt content", () => {
		const prepared = prepareContextMessages(
			[
				{
					role: "user",
					content: [
						{ type: "text", text: "look here" },
						{ type: "image", image_url: "ignored" },
						{ type: "text", text: "and there" },
					],
				},
			],
			env,
		);

		const content = (prepared.augmentedMessages[0] as { content: Array<{ text?: string }> }).content;
		expect(content[0].text).toBe(`${env}\n\nlook here`);
		expect(prepared.rawPromptWordCount).toBe(4);
		expect(prepared.userPrompt).toBe(`${env}\n\nlook here and there`);
	});

	it("adds an environment text block before non-text array prompts", () => {
		const prepared = prepareContextMessages(
			[
				{
					role: "user",
					content: [
						{ type: "image", image_url: "ignored" },
						{ type: "text", text: "describe this" },
					],
				},
			],
			env,
		);

		const content = (prepared.augmentedMessages[0] as { content: Array<{ type: string; text?: string }> }).content;
		expect(content[0]).toEqual({ type: "text", text: `${env}\n\n` });
		expect(prepared.userPrompt).toBe(`${env}\n\n describe this`);
	});

	it("distinguishes no-user messages from unsupported user content", () => {
		const noUser = prepareContextMessages([{ role: "assistant", content: "hello" }], env);
		expect(noUser.hasUserMessage).toBe(false);
		expect(noUser.userPrompt).toBeNull();

		const unsupported = prepareContextMessages([{ role: "user", content: 42 }], env);
		expect(unsupported.hasUserMessage).toBe(true);
		expect(unsupported.userPrompt).toBeNull();
		expect(unsupported.rawPromptWordCount).toBe(0);
	});

	it("counts words and clips string prompts to 200 space-separated tokens", () => {
		expect(countWordsInMessageContent(" one\n two   three ")).toBe(3);
		expect(
			countWordsInMessageContent([
				{ type: "text", text: "one two" },
				{ type: "text", text: "three" },
			]),
		).toBe(3);
		expect(countWordsInMessageContent([{ type: "image" }])).toBe(0);

		const longPrompt = Array.from({ length: 205 }, (_, i) => `w${i}`).join(" ");
		expect(extractContextPrompt(longPrompt)?.split(" ")).toHaveLength(200);
		expect(extractContextPrompt([{ type: "text", text: "array prompt" }])).toBe("array prompt");
		expect(extractContextPrompt({ nope: true })).toBeNull();
	});

	it("applies relevance-list suppression gates", () => {
		expect(shouldDropRelevanceList(100, { DROP_RELEVANCE_LIST: "1" })).toBe(true);
		expect(shouldDropRelevanceList(100, { DROP_RELEVANCE_LIST: "true" })).toBe(true);
		expect(shouldDropRelevanceList(19, { RELEVANCE_LIST_MIN_WORDS: "20" })).toBe(true);
		expect(shouldDropRelevanceList(20, { RELEVANCE_LIST_MIN_WORDS: "20" })).toBe(false);
		expect(
			shouldDropRelevanceList(20, { DROP_RELEVANCE_LIST: "false", RELEVANCE_LIST_MIN_WORDS: "not-a-number" }),
		).toBe(false);
	});

	it("builds the stable context message prefix in display order", () => {
		const system = { role: "developer", content: "dev rules" };
		const prefix = buildContextMessagePrefix(system, "preamble", "recency", "relevance");

		expect(prefix).toHaveLength(4);
		expect(prefix[0]).toBe(system);
		expect((prefix[1] as { content: Array<{ text: string }> }).content[0].text).toBe("preamble");
		expect((prefix[2] as { content: Array<{ text: string }> }).content[0].text).toBe("recency");
		expect((prefix[3] as { content: Array<{ text: string }> }).content[0].text).toBe("relevance");
		expect(buildContextMessagePrefix(null, null, null, null)).toEqual([]);
	});
});

describe("get_round_details join points", () => {
	const roundData = {
		userPrompt: "user prompt",
		responseSequence: "line one\nline two needle\nline three\nline four",
		toolCallCount: 1,
		toolCallNames: ["read"],
		responseSegments: [
			{ type: "text", text: "before tool" },
			{ type: "toolCall", toolCallIndex: 0 },
			{ type: "ignored", text: "not rendered" },
			{ type: "text", text: "after tool" },
		],
		toolCalls: [
			{
				index: 0,
				name: "read",
				arguments: '{"path":"secret"}',
				result_summary: "short result",
				result_full: "full result text",
			},
		],
		parentId: "parent.json",
		relatedParentId: "related.json",
	};

	it("renders interleaved assistant output and redacts tool details", () => {
		const result = renderRoundDetailsToolResult({ round: "round.json" }, roundData);

		expect(result.content[0].text).toContain("=== Round: round.json ===");
		expect(result.content[0].text).toContain("User: user prompt");
		expect(result.content[0].text).toContain("before tool\n[Tool call REDACTED");
		expect(result.content[0].text).toContain('use get_tool_details("round.json", 0)');
		expect(result.content[0].text).toContain("Tools used: 1 (read)");
		expect(result.content[0].text).toContain("Parent round: parent.json");
		expect(result.content[0].text).toContain("Related to:   related.json (same topic group)");

		const toolCall = (result.details.toolCalls as Array<Record<string, unknown>>)[0];
		expect(toolCall.arguments).toBe('[REDACTED — use get_tool_details("round.json", 0) to expand]');
		expect(toolCall.result_summary).toContain("[REDACTED — size:");
		expect(toolCall.result_full).toBeUndefined();
	});

	it("falls back to responseSequence and discussion-only tool metadata", () => {
		const result = renderRoundDetailsToolResult(
			{ round: "old.json" },
			{ responseSequence: "legacy", toolCallCount: 0 },
		);

		expect(result.content[0].text).toContain("User: (empty)");
		expect(result.content[0].text).toContain("Assistant: legacy");
		expect(result.content[0].text).toContain("Tools used: 0 (discussion only)");
		expect(buildRoundAssistantOutput("empty.json", {})).toBe("(empty)");
	});

	it("pages assistant lines and emits a continuation marker", () => {
		const result = renderRoundDetailsToolResult({ round: "round.json", from_line: 2, line_count: 2 }, roundData);

		expect(result.content[0].text).toContain("=== Round: round.json (lines 2–3 of 4) ===");
		expect(result.content[0].text).toContain("Assistant: line two needle\nline three");
		expect(result.content[0].text).toContain("[Truncated — use from_line=4, line_count=2 to continue]");
	});

	it("renders regexp matches from assistant and user text", () => {
		const assistantMatch = renderRoundDetailsToolResult(
			{ round: "round.json", match: "needle", line_count: 1 },
			roundData,
		);
		expect(assistantMatch.content[0].text).toContain("=== Round: round.json (1 match) ===");
		expect(assistantMatch.content[0].text).toContain("[M 1/1 at assistant line 2 (1 lines of context)]");
		expect(assistantMatch.content[0].text).toContain("line two needle\nline three");

		const userMatch = renderRoundDetailsToolResult({ round: "round.json", match: "user", max_matches: 2 }, roundData);
		expect(userMatch.content[0].text).toContain("[M 1/1 in user prompt] user prompt");
	});

	it("returns selection errors without expanded details", () => {
		const conflict = renderRoundDetailsToolResult({ round: "round.json", from_line: 1, match: "x" }, roundData);
		expect(conflict.content[0].text).toBe(
			"Error: match and from_line are mutually exclusive. Use one or the other, not both.",
		);
		expect(conflict.details).toEqual({});

		const invalidRegex = renderRoundDetailsToolResult({ round: "round.json", match: "(" }, roundData);
		expect(invalidRegex.content[0].text).toContain("Invalid regexp pattern:");
		expect(invalidRegex.details).toEqual({});
	});

	it("covers standalone metadata and collapse branches", () => {
		expect(formatRoundToolMeta({ toolCallCount: 2 })).toBe("\n  Tools used: 2 (unknown)");
		expect(formatRoundToolMeta({})).toBe("");
		expect(
			collapseRoundDetails("round.json", { toolCalls: [{ index: 1, result_summary: "summary" }] }).toolCalls,
		).toEqual([
			expect.objectContaining({
				arguments: '[REDACTED — use get_tool_details("round.json", 1) to expand]',
				result_full: undefined,
			}),
		]);
	});
});
