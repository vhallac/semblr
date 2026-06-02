import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it, vi } from "vitest";
import registerSemblr, {
	applyMessageEndToState,
	buildAgentEndChainEntry,
	buildAgentEndEmbeddingTexts,
	buildAgentEndRoundData,
	buildAgentEndToolSummary,
	buildContextMessagePrefix,
	buildRoundAssistantOutput,
	collapseRoundDetails,
	collectSearchRoundScores,
	countWordsInMessageContent,
	embedText,
	extractAgentEndResponseText,
	extractAgentEndUserPrompt,
	extractContextPrompt,
	filterSearchIndexByRounds,
	formatRoundToolMeta,
	getAgentEndParentId,
	getRelatedParentIdFromGroup,
	loadRoundDataForToolDetails,
	type MessageEndProcessingState,
	normalizeSearchInteractionsParams,
	prepareContextMessages,
	type RoundData,
	renderRoundDetailsToolResult,
	renderSearchInteractionsToolResult,
	renderToolDetailsToolResult,
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

describe("agent_end handler join points", () => {
	it("extracts the prompt from cached state, string messages, and text-array messages", () => {
		expect(extractAgentEndUserPrompt("cached prompt", [{ role: "user", content: "message prompt" }])).toBe(
			"cached prompt",
		);
		expect(
			extractAgentEndUserPrompt(null, [
				{ role: "user", content: "first prompt" },
				{ role: "assistant", content: "answer" },
				{ role: "user", content: "last prompt" },
			]),
		).toBe("last prompt");
		expect(
			extractAgentEndUserPrompt("", [
				{
					role: "user",
					content: [{ type: "text", text: "array" }, { type: "image" }, { type: "text", text: "prompt" }],
				},
			]),
		).toBe("array prompt");
		expect(extractAgentEndUserPrompt(null, [{ role: "assistant", content: "answer" }])).toBe("");
		expect(extractAgentEndUserPrompt(null, [{ role: "user", content: 42 }])).toBe("");
	});

	it("prefers accumulated assistant text and falls back to the last assistant message", () => {
		expect(extractAgentEndResponseText([" first ", "second"])).toBe("first \n\nsecond");
		expect(
			extractAgentEndResponseText(
				[],
				[
					{ role: "assistant", content: "old answer" },
					{ role: "user", content: "prompt" },
					{ role: "assistant", content: "new answer" },
				],
			),
		).toBe("new answer");
		expect(
			extractAgentEndResponseText(
				[],
				[
					{
						role: "assistant",
						content: [
							{ type: "text", text: "array" },
							{ type: "toolCall", name: "read" },
							{ type: "text", text: "answer" },
						],
					},
				],
			),
		).toBe("array answer");
		expect(extractAgentEndResponseText([], [{ role: "user", content: "prompt" }])).toBe("");
		expect(extractAgentEndResponseText([], [{ role: "assistant", content: 42 }])).toBe("");
	});

	it("builds causal chain entries and parent metadata from existing handler state", () => {
		expect(buildAgentEndToolSummary(0, [])).toBe("0 tools (discussion)");
		expect(buildAgentEndToolSummary(2, ["read", "bash"])).toBe("2 tools (read, bash)");

		const entry = buildAgentEndChainEntry("round.json", "prompt", "answer", 1, ["read"]);
		expect(entry).toEqual({
			fileName: "round.json",
			userPrompt: "prompt",
			responseSequence: "answer",
			toolSummary: "1 tools (read)",
		});
		expect(getAgentEndParentId([])).toBeNull();
		expect(getAgentEndParentId([{ fileName: "one.json" }])).toBeNull();
		expect(getAgentEndParentId([{ fileName: "one.json" }, { fileName: "two.json" }])).toBe("one.json");
	});

	it("builds the persisted round payload with default and explicit values", () => {
		const toolCalls = [
			{
				index: 0,
				name: "read",
				arguments: "{}",
				result_summary: "summary",
				result_full: "full",
				result_truncated: false,
			},
		];
		const roundData = buildAgentEndRoundData({
			userPrompt: "prompt",
			responseText: "answer",
			turnIndex: null,
			toolCallCount: 1,
			toolCallNames: ["read"],
			toolCalls,
			responseSegments: [{ type: "toolCall", toolCallIndex: 0 }],
			parentId: "parent.json",
			userTimestamp: 123,
		});

		expect(roundData).toMatchObject({
			id: expect.any(String),
			userPrompt: "prompt",
			responseSequence: "answer",
			turnIndex: 0,
			userTimestamp: 123,
			toolCallCount: 1,
			toolCallNames: ["read"],
			toolCalls,
			responseSegments: [{ type: "toolCall", toolCallIndex: 0 }],
			parentId: "parent.json",
			relatedParentId: null,
		});
		expect(Object.hasOwn(roundData, "promptEmbedding")).toBe(true);
		expect(roundData.promptEmbedding).toBeUndefined();

		const explicitTurn = buildAgentEndRoundData({
			userPrompt: "prompt",
			responseText: "answer",
			turnIndex: 7,
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
			parentId: null,
			userTimestamp: 456,
		});
		expect(explicitTurn.turnIndex).toBe(7);
	});

	it("strips redacted markers and clips embedding response input by byte length", () => {
		expect(buildAgentEndEmbeddingTexts("prompt", "keep\n[REDACTED secret]\nafter")).toEqual({
			clippedResponse: "keep\nafter",
			combinedText: "prompt\n\nkeep\nafter",
		});
		expect(buildAgentEndEmbeddingTexts("prompt", "before\n[Tool call REDACTED: hidden]\nafter")).toEqual({
			clippedResponse: "before\nafter",
			combinedText: "prompt\n\nbefore\nafter",
		});

		const clipped = buildAgentEndEmbeddingTexts("p", "abcdef", 3);
		expect(clipped.clippedResponse).toBe("abc");
		expect(clipped.combinedText).toBe("p\n\nabc");
	});

	it("finds the related parent inside a semantic group", () => {
		const first = { fileName: "first.json" };
		const second = { fileName: "second.json" };
		const third = { fileName: "third.json" };

		expect(getRelatedParentIdFromGroup({ rounds: [] }, first)).toBeNull();
		expect(getRelatedParentIdFromGroup({ rounds: [first] }, first)).toBeNull();
		expect(getRelatedParentIdFromGroup({ rounds: [first, second, third] }, third)).toBe("second.json");
		expect(getRelatedParentIdFromGroup({ rounds: [first, second] }, { fileName: "missing.json" })).toBeNull();
	});

	it("agent_end hook reports early skip statuses for missing prompt or response", async () => {
		const handlers = new Map<string, (event: unknown, ctx: unknown) => Promise<unknown>>();
		const pi = {
			on: vi.fn((name: string, handler: (event: unknown, ctx: unknown) => Promise<unknown>) => {
				handlers.set(name, handler);
			}),
			registerCommand: vi.fn(),
			registerTool: vi.fn(),
		};
		const statuses: string[] = [];
		const ctx = {
			ui: {
				setStatus: vi.fn((_key: string, value: string) => statuses.push(value)),
				notify: vi.fn(),
			},
		};

		registerSemblr(pi as never);
		await handlers.get("agent_start")?.({}, ctx);
		await handlers.get("agent_end")?.({ messages: [{ role: "assistant", content: "answer" }] }, ctx);
		await handlers.get("agent_start")?.({}, ctx);
		await handlers.get("agent_end")?.({ messages: [{ role: "user", content: "prompt" }] }, ctx);

		expect(statuses).toContain("🧠 agent_end: no user prompt to save");
		expect(statuses).toContain("🧠 agent_end: no response text");
	});
});

describe("embedText join points", () => {
	const response = (extra: Partial<{ ok: boolean; status: number; text: string; embedding: number[] }> = {}) => ({
		ok: extra.ok ?? true,
		status: extra.status ?? 200,
		text: async () => extra.text ?? "",
		json: async () => ({ data: [{ embedding: extra.embedding ?? [1, 2, 3] }] }),
	});

	it("sends the OpenRouter embedding request and returns the response vector", async () => {
		const fetchImpl = vi.fn(async (_url: string, _init: RequestInit) => response({ embedding: [0.1, 0.2] }));

		const embedding = await embedText("hello world", "api-key", { fetchImpl });

		expect(embedding).toEqual([0.1, 0.2]);
		expect(fetchImpl).toHaveBeenCalledTimes(1);
		const [url, init] = fetchImpl.mock.calls[0];
		expect(url).toBe("https://openrouter.ai/api/v1/embeddings");
		expect(init.method).toBe("POST");
		expect(init.headers).toEqual({
			"Content-Type": "application/json",
			Authorization: "Bearer api-key",
		});
		expect(JSON.parse(init.body as string)).toEqual({
			model: "openai/text-embedding-3-small",
			input: "hello world",
		});
		expect(init.signal).toBeInstanceOf(AbortSignal);
	});

	it("retries rate-limit and server errors with exponential backoff", async () => {
		const fetchImpl = vi
			.fn()
			.mockResolvedValueOnce(response({ ok: false, status: 429, text: "rate limited" }))
			.mockResolvedValueOnce(response({ ok: false, status: 500, text: "server exploded" }))
			.mockResolvedValueOnce(response({ embedding: [9, 8, 7] }));
		const sleep = vi.fn(async (_ms: number) => {});

		const embedding = await embedText("retry me", "api-key", { fetchImpl, sleep, backoffMs: 25 });

		expect(embedding).toEqual([9, 8, 7]);
		expect(fetchImpl).toHaveBeenCalledTimes(3);
		expect(sleep).toHaveBeenNthCalledWith(1, 25);
		expect(sleep).toHaveBeenNthCalledWith(2, 50);
	});

	it("does not retry fatal non-rate-limit client errors", async () => {
		const fetchImpl = vi.fn(async (_url: string, _init: RequestInit) =>
			response({ ok: false, status: 401, text: "bad key" }),
		);
		const sleep = vi.fn(async (_ms: number) => {});

		await expect(embedText("fatal", "api-key", { fetchImpl, sleep })).rejects.toThrow(
			"Embedding API error 401: bad key",
		);
		expect(fetchImpl).toHaveBeenCalledTimes(1);
		expect(sleep).not.toHaveBeenCalled();
	});

	it("reports timeout and final retry errors", async () => {
		const timeoutFetch = vi.fn(async () => {
			throw new DOMException("aborted", "AbortError");
		});
		await expect(
			embedText("slow", "api-key", { fetchImpl: timeoutFetch, maxRetries: 1, timeoutMs: 5 }),
		).rejects.toThrow("Embedding API timeout after 5ms");

		const networkFetch = vi.fn(async () => {
			throw new Error("network down");
		});
		const sleep = vi.fn(async (_ms: number) => {});
		await expect(embedText("offline", "api-key", { fetchImpl: networkFetch, sleep, maxRetries: 2 })).rejects.toThrow(
			"network down",
		);
		expect(networkFetch).toHaveBeenCalledTimes(2);
		expect(sleep).toHaveBeenCalledTimes(1);
	});
});

describe("message_end handler join points", () => {
	const emptyState = (): MessageEndProcessingState => ({
		accumulatedText: [],
		toolCallCount: 0,
		toolCallNames: [],
		toolCalls: [],
		responseSegments: [],
	});

	it("ignores missing messages, user messages, unknown roles, and non-array assistant content", () => {
		const state = emptyState();

		applyMessageEndToState(undefined, state);
		applyMessageEndToState({ role: "user", content: "prompt" }, state);
		applyMessageEndToState({ role: "assistant", content: "plain answer" }, state);
		applyMessageEndToState({ role: "other", content: [{ type: "text", text: "ignored" }] }, state);

		expect(state).toEqual(emptyState());
	});

	it("captures assistant text, tool-call details, unique names, and response segment order", () => {
		const state = emptyState();

		applyMessageEndToState(
			{
				role: "assistant",
				content: [
					{ type: "text", text: "before" },
					{ type: "text", text: "" },
					{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "a.txt" } },
					{ type: "toolCall", id: "call-2", name: "read", arguments: { path: "b.txt" } },
					{ type: "toolCall", name: "bash", arguments: { command: "pwd" } },
					{ type: "toolCall", id: "call-4", arguments: { ignored: true } },
					{ type: "text", text: "after" },
				],
			},
			state,
		);

		expect(state.accumulatedText).toEqual(["before", "after"]);
		expect(state.toolCallCount).toBe(4);
		expect(state.toolCallNames).toEqual(["read", "bash"]);
		expect(state.toolCalls).toEqual([
			{ index: 0, name: "read", arguments: '{"path":"a.txt"}', result_summary: "" },
			{ index: 1, name: "read", arguments: '{"path":"b.txt"}', result_summary: "" },
		]);
		expect(state.responseSegments).toEqual([
			{ type: "text", text: "before" },
			{ type: "toolCall", toolCallIndex: 0 },
			{ type: "toolCall", toolCallIndex: 1 },
			{ type: "toolCall", toolCallIndex: 1 },
			{ type: "toolCall", toolCallIndex: 1 },
			{ type: "text", text: "after" },
		]);
	});

	it("attaches tool results to the latest unresolved call in result order", () => {
		const state = emptyState();
		state.toolCalls.push(
			{ index: 0, name: "read", arguments: "{}", result_summary: "" },
			{ index: 1, name: "bash", arguments: "{}", result_summary: "" },
		);

		applyMessageEndToState({ role: "toolResult", content: [{ type: "text", text: "ignored" }] }, state);
		expect(state.toolCalls[1].result_summary).toBe("");

		const longResult = "x".repeat(350);
		applyMessageEndToState(
			{
				role: "toolResult",
				toolCallId: "call-2",
				content: [{ type: "text", text: longResult }, { type: "image" }, { type: "text", text: "tail" }],
			},
			state,
		);
		expect(state.toolCalls[1].result_summary).toBe(longResult.slice(0, 300));
		expect(state.toolCalls[1].result_full).toBe(`${longResult} tail`);
		expect(state.toolCalls[1].result_truncated).toBe(false);

		applyMessageEndToState(
			{ role: "toolResult", toolCallId: "call-1", content: [{ type: "text", text: "first result" }] },
			state,
		);
		expect(state.toolCalls[0].result_summary).toBe("first result");
		expect(state.toolCalls[0].result_full).toBe("first result");
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

describe("get_tool_details join points", () => {
	const roundData = (extra: Partial<RoundData> = {}): RoundData => ({
		userPrompt: "prompt",
		responseSequence: "response",
		turnIndex: 0,
		toolCalls: [
			{
				index: 0,
				name: "bash",
				arguments: '{"command":"printf"}',
				result_summary: "summary",
				result_full: "line one\nneedle line\nline three\nneedle again",
			},
		],
		...extra,
	});

	it("loads missing, malformed, and valid round files", () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-tool-details-"));
		try {
			const missing = loadRoundDataForToolDetails(path.join(dir, "missing.json"), "missing.json");
			expect(missing.ok).toBe(false);
			if (!missing.ok) expect(missing.result.content[0].text).toBe("Round file not found: missing.json");

			const malformedPath = path.join(dir, "bad.json");
			fs.writeFileSync(malformedPath, "{not json", "utf-8");
			const malformed = loadRoundDataForToolDetails(malformedPath, "bad.json");
			expect(malformed.ok).toBe(false);
			if (!malformed.ok) expect(malformed.result.content[0].text).toBe("Failed to parse round file: bad.json");

			const validPath = path.join(dir, "good.json");
			fs.writeFileSync(validPath, JSON.stringify(roundData()), "utf-8");
			const valid = loadRoundDataForToolDetails(validPath, "good.json");
			expect(valid.ok).toBe(true);
			if (valid.ok) expect(valid.roundData.toolCalls?.[0].name).toBe("bash");
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	it("renders no-tool and invalid-index errors", () => {
		const noToolCalls = renderToolDetailsToolResult(
			{ round: "empty.json", index: 0 },
			roundData({ toolCalls: undefined }),
		);
		expect(noToolCalls.content[0].text).toContain("This round has no tool calls stored");
		expect(noToolCalls.details).toEqual({});

		const emptyToolCalls = renderToolDetailsToolResult(
			{ round: "empty.json", index: 0 },
			roundData({ toolCalls: [] }),
		);
		expect(emptyToolCalls.content[0].text).toContain("This round has no tool calls stored");

		const negative = renderToolDetailsToolResult({ round: "round.json", index: -1 }, roundData());
		expect(negative.content[0].text).toBe("Invalid index -1. This round has 1 tool calls (indices 0–0).");

		const tooHigh = renderToolDetailsToolResult({ round: "round.json", index: 1 }, roundData());
		expect(tooHigh.content[0].text).toBe("Invalid index 1. This round has 1 tool calls (indices 0–0).");
	});

	it("renders full output, parsed arguments, empty output, and truncation metadata", () => {
		const full = renderToolDetailsToolResult(
			{ round: "round.json", index: 0 },
			roundData({ toolCalls: [{ ...roundData().toolCalls![0], result_truncated: true }] }),
		);
		expect(full.content[0].text).toContain("Tool call #0 in round round.json");
		expect(full.content[0].text).toContain('  Arguments: {\n  "command": "printf"\n}');
		expect(full.content[0].text).toContain("[Output exceeds storage cap — showing entire stored result]");
		expect(full.details).toEqual({
			name: "bash",
			arguments: '{"command":"printf"}',
			result_full: "line one\nneedle line\nline three\nneedle again",
		});

		const empty = renderToolDetailsToolResult(
			{ round: "round.json", index: 0 },
			roundData({ toolCalls: [{ index: 0, name: "read", arguments: "{}", result_summary: "" }] }),
		);
		expect(empty.content[0].text).toContain("  Result:\n  (empty)");
	});

	it("renders paginated output and selection errors", () => {
		const page = renderToolDetailsToolResult(
			{ round: "round.json", index: 0, out__from_line: 2, out_line_count: 2 },
			roundData(),
		);
		expect(page.content[0].text).toContain("[Showing lines 2–3 of 4 for tool call #0 (bash)]");
		expect(page.content[0].text).toContain("needle line\nline three");
		expect(page.content[0].text).toContain("Use out__from_line=4 and out_line_count=2 to continue");
		expect(page.details.lines_shown).toEqual({ from: 2, to: 3, of: 4 });

		const conflict = renderToolDetailsToolResult(
			{ round: "round.json", index: 0, out__from_line: 1, match: "needle" },
			roundData(),
		);
		expect(conflict.content[0].text).toBe(
			"Error: match and out__from_line are mutually exclusive. Use one or the other, not both.",
		);

		const invalidRegex = renderToolDetailsToolResult({ round: "round.json", index: 0, match: "(" }, roundData());
		expect(invalidRegex.content[0].text).toContain("Invalid regexp pattern:");
	});

	it("renders regexp matches and falls back to summary with non-json arguments", () => {
		const matches = renderToolDetailsToolResult(
			{ round: "round.json", index: 0, match: "needle", out_line_count: 1, max_matches: 1 },
			roundData({
				toolCalls: [
					{
						index: 0,
						name: "read",
						arguments: "raw args",
						result_summary: "first\nneedle summary\nafter",
					},
				],
			}),
		);

		expect(matches.content[0].text).toContain("[Match results for tool call #0 (read) (1 match)]");
		expect(matches.content[0].text).toContain('Arguments: "raw args"');
		expect(matches.content[0].text).toContain("[M 1/1 at line 2 (1 lines of context)]\nneedle summary\nafter");
		expect(matches.details).toEqual({ name: "read", arguments: "raw args", matches: { shown: 1, total: 1 } });
	});
});

describe("search_interactions join points", () => {
	const round = (userPrompt: string, responseSequence: string, extra = {}) => ({
		userPrompt,
		responseSequence,
		turnIndex: 0,
		...extra,
	});

	it("normalizes parameters and scopes index entries by round file", () => {
		expect(normalizeSearchInteractionsParams({ query: "topic" })).toEqual({
			query: "topic",
			threshold: 0.25,
			scopeRounds: null,
		});
		expect(normalizeSearchInteractionsParams({ query: "", minSimilarity: 0.7, rounds: ["a.json"] })).toEqual({
			query: null,
			threshold: 0.7,
			scopeRounds: ["a.json"],
		});

		const index = [
			{ filePath: "a.json:prompt", vector: [1, 0] },
			{ filePath: "b.json:response", vector: [0, 1] },
			{ filePath: "c.json:round", vector: [1, 1] },
			{ filePath: "weird.json:prompt-extra", vector: [1, 0] },
		];
		expect(filterSearchIndexByRounds(index, null)).toEqual(index);
		expect(filterSearchIndexByRounds(index, [])).toEqual(index);
		expect(
			filterSearchIndexByRounds(index, ["b.json", "c.json", "weird.json-extra"]).map((entry) => entry.filePath),
		).toEqual(["b.json:response", "c.json:round"]);
	});

	it("collects best search scores per readable json round", () => {
		const rounds = new Map([
			["a.json", round("a user", "a answer")],
			["b.json", round("b user", "b answer")],
		]);
		const scores = collectSearchRoundScores(
			[
				{ filePath: "a.json:prompt", vector: [0, 1] },
				{ filePath: "a.json:response", vector: [1, 0] },
				{ filePath: "missing.json:prompt", vector: [0.9, 0.1] },
				{ filePath: "notes.txt:prompt", vector: [1, 0] },
				{ filePath: "b.json:round", vector: [0, 1] },
			],
			[1, 0],
			(filePath) => rounds.get(filePath.replace(/(:prompt|:response|:round)$/, "")) ?? null,
		);

		expect(scores).toHaveLength(2);
		expect(scores[0].fileName).toBe("a.json");
		expect(scores[0].bestScore).toBeCloseTo(1);
		expect(scores[1].fileName).toBe("b.json");
		expect(scores[1].bestScore).toBeCloseTo(0);
	});

	it("renders capped search results with tool-call redaction and discussion-only turns", () => {
		const result = renderSearchInteractionsToolResult(
			[
				{
					fileName: "tool.json",
					bestScore: 0.9,
					data: round("tool user", "tool answer", {
						toolCallCount: 1,
						toolCallNames: ["read"],
						toolCalls: [
							{
								index: 0,
								name: "read",
								arguments: "{}",
								result_summary: "short",
								result_full: "x".repeat(2048),
							},
						],
					}),
				},
				{
					fileName: "discussion.json",
					bestScore: 0.8,
					data: round("discussion user", "discussion answer", { toolCallCount: 0 }),
				},
				{ fileName: "plain.json", bestScore: 0.7, data: round("plain user", "plain answer") },
				{ fileName: "four.json", bestScore: 0.6, data: round("four user", "four answer") },
				{ fileName: "threshold.json", bestScore: 0.25, data: round("threshold user", "threshold answer") },
				{ fileName: "six.json", bestScore: 0.4, data: round("six user", "six answer") },
			],
			0.25,
			(fileName) => (fileName === "tool.json" ? "99B" : null),
		);

		expect(result.details).toEqual({ matched: 5, topScore: 0.9 });
		expect(result.content[0].type).toBe("text");
		expect(result.content[0].text).toContain("Found 5 relevant rounds");
		expect(result.content[0].text).toContain("--- Round tool.json (score: 0.900 | 1 tools (read) | 99B) ---");
		expect(result.content[0].text).toContain(
			"\n--- Round discussion.json (score: 0.800 | 0 tools (discussion only)) ---",
		);
		expect(result.content[0].text).toContain("\n--- Round plain.json (score: 0.700) ---");
		expect(result.content[0].text).toContain(
			'  Turn 0: read (2KB) — [REDACTED: use get_tool_details("tool.json", 0) to expand.]',
		);
		expect(result.content[0].text).toContain(
			"--- Agent turns (all tool calls redacted — use get_tool_details to expand) ---",
		);
		expect(result.content[0].text).toContain("--- Agent turns ---\n  (no tool calls — discussion only)");
		expect(result.content[0].text).toContain("Assistant: plain answer");
		expect(result.content[0].text).toContain("threshold.json");
		expect(result.content[0].text).not.toContain("six.json");
		expect(result.content[0].text).not.toContain("Stryker was here");
	});

	it("renders no-match and below-threshold search outcomes", () => {
		const noMatches = renderSearchInteractionsToolResult([], 0.25, () => null);
		expect(noMatches.content[0]).toEqual({ type: "text", text: "No matching turns found in the index." });
		expect(noMatches.details).toEqual({});

		const belowThreshold = renderSearchInteractionsToolResult(
			[{ fileName: "weak.json", bestScore: 0.1, data: round("weak user", "weak answer") }],
			0.25,
			() => null,
		);
		expect(belowThreshold.content[0].text).toBe("No relevant rounds found (best score: 0.100).");
		expect(belowThreshold.details).toEqual({});
	});
});
