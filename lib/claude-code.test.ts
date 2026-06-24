import { describe, expect, it } from "vitest";
import {
	claudeRoundFileName,
	isRealClaudeUserPrompt,
	parseClaudeCodeJsonl,
	textFromClaudeContent,
} from "./claude-code.ts";

const line = (value: unknown) => JSON.stringify(value);

const options = {
	filePath: "/home/me/.claude/projects/proj/session.jsonl",
	projectsDir: "/home/me/.claude/projects",
	now: () => 123,
};

describe("Claude Code parsing", () => {
	it("extracts text and optional tool result content", () => {
		const content = [
			{ type: "text", text: "hello" },
			{ type: "tool_result", content: [{ type: "text", text: "tool output" }] },
		];

		expect(textFromClaudeContent(content)).toBe("hello");
		expect(textFromClaudeContent(content, { includeToolResults: true })).toBe("hello\ntool output");
		expect(textFromClaudeContent(" plain ")).toBe("plain");
		expect(textFromClaudeContent({})).toBe("");
	});

	it("identifies real user prompts and excludes tool-result-only messages", () => {
		expect(isRealClaudeUserPrompt({ type: "user", message: { content: "question" } })).toBe(true);
		expect(isRealClaudeUserPrompt({ type: "user", message: { content: [{ type: "text", text: "question" }] } })).toBe(
			true,
		);
		expect(
			isRealClaudeUserPrompt({ type: "user", message: { content: [{ type: "tool_result", content: "result" }] } }),
		).toBe(false);
		expect(isRealClaudeUserPrompt({ type: "assistant", message: { content: "question" } })).toBe(false);
	});

	it("parses Claude rounds with assistant text, tool calls, and tool results", () => {
		const raw = [
			line({
				type: "user",
				timestamp: "2026-01-01T00:00:00.000Z",
				sessionId: "s1",
				cwd: "/repo",
				gitBranch: "main",
				message: { content: [{ type: "text", text: "Prompt" }] },
			}),
			line({
				type: "assistant",
				timestamp: "2026-01-01T00:00:01.000Z",
				message: {
					content: [
						{ type: "text", text: "Long enough assistant text" },
						{ type: "tool_use", id: "tool-1", name: "Read", input: { path: "a.ts" } },
					],
				},
			}),
			line({
				type: "user",
				message: { content: [{ type: "tool_result", tool_use_id: "tool-1", content: "file contents" }] },
			}),
		].join("\n");

		const [round] = parseClaudeCodeJsonl(raw, options);

		expect(round).toMatchObject({
			source: "claude-code",
			userPrompt: "Prompt",
			responseSequence: "Long enough assistant text",
			responseSegments: [
				{ type: "text", text: "Long enough assistant text" },
				{ type: "toolCall", toolCallIndex: 0 },
			],
			userTimestamp: Date.parse("2026-01-01T00:00:00.000Z"),
			responseEndTimestamp: Date.parse("2026-01-01T00:00:01.000Z"),
			turnIndex: 0,
			sessionLabel: "proj/session.jsonl",
			claudeSessionId: "s1",
			cwd: "/repo",
			gitBranch: "main",
			toolCallCount: 1,
			toolCallNames: ["Read"],
			toolCalls: [{ index: 0, name: "Read", arguments: '{"path":"a.ts"}', result_summary: "file contents" }],
		});
		expect(round.id).toHaveLength(32);
		expect(claudeRoundFileName(round)).toBe(`${round.id}.json`);
	});

	it("skips sidechains by default and can include them", () => {
		const raw = [
			line({ isSidechain: true, type: "user", message: { content: "Prompt" } }),
			line({
				isSidechain: true,
				type: "assistant",
				message: { content: [{ type: "text", text: "Long enough side response" }] },
			}),
		].join("\n");

		expect(parseClaudeCodeJsonl(raw, options)).toEqual([]);
		expect(parseClaudeCodeJsonl(raw, { ...options, includeSidechains: true })).toHaveLength(1);
	});

	it("skips short responses and reports invalid JSON with file and line", () => {
		const short = [
			line({ type: "user", message: { content: "Prompt" } }),
			line({ type: "assistant", message: { content: [{ type: "text", text: "short" }] } }),
		].join("\n");

		expect(parseClaudeCodeJsonl(short, options)).toEqual([]);
		expect(() => parseClaudeCodeJsonl("{}\nnot-json", options)).toThrow("session.jsonl:2: invalid JSON");
	});

	it("sanitizes user prompt containing a JSON-stringified prompt envelope", () => {
		const envelope = JSON.stringify([
			{ role: "system", content: "You are a helpful assistant." },
			{ role: "user", content: "What is the capital of France?" },
		]);

		const raw = [
			line({
				type: "user",
				timestamp: "2026-01-01T00:00:00.000Z",
				sessionId: "s1",
				message: { content: [{ type: "text", text: envelope }] },
			}),
			line({
				type: "assistant",
				timestamp: "2026-01-01T00:00:01.000Z",
				message: { content: [{ type: "text", text: "Long enough assistant text here" }] },
			}),
		].join("\n");

		const [round] = parseClaudeCodeJsonl(raw, options);

		expect(round.userPrompt).toBe("What is the capital of France?");
		expect(round.source).toBe("claude-code");
	});

	it("pairs anonymous tool results with the latest pending tool call", () => {
		const raw = [
			line({ type: "user", timestamp: "bad-date", message: { content: "Prompt" } }),
			line({
				type: "assistant",
				message: {
					content: [
						{ type: "tool_use", name: "A", input: {} },
						{ type: "tool_use", input: { b: true } },
						{ type: "text", text: "Long enough response text" },
					],
				},
			}),
			line({ type: "user", message: { content: [{ type: "tool_result", content: "fallback result" }] } }),
		].join("\n");

		const [round] = parseClaudeCodeJsonl(raw, options);

		expect(round.userTimestamp).toBe(0);
		expect(round.responseEndTimestamp).toBe(123);
		expect(round.toolCallNames).toEqual(["A", "unknown"]);
		expect(round.toolCalls[1]).toMatchObject({ name: "unknown", result_summary: "fallback result" });
	});
});
