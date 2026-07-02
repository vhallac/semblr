import { describe, expect, it } from "vitest";
import { parsePiSessionJsonl, parsePiTextContent } from "./pi-session.ts";

const line = (value: unknown) => JSON.stringify(value);

describe("pi session parsing", () => {
	it("extracts text content and ignores non-text blocks", () => {
		expect(
			parsePiTextContent([
				{ type: "text", text: "hello" },
				{ type: "toolCall", text: "ignored" },
				{ type: "text", text: "world" },
			]),
		).toBe("hello world");
		expect(parsePiTextContent(undefined)).toBe("");
	});

	it("parses user and assistant messages into rounds", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "Prompt" }] },
			}),
			line({
				type: "message",
				id: "a1",
				message: {
					role: "assistant",
					timestamp: 20,
					content: [{ type: "text", text: "Response with enough text" }],
				},
			}),
		].join("\n");

		const rounds = parsePiSessionJsonl(raw, { now: () => 30 });

		expect(rounds).toMatchObject([
			{
				id: "u1",
				userPrompt: "Prompt",
				responseSequence: "Response with enough text",
				responseSegments: [{ type: "text", text: "Response with enough text" }],
				userTimestamp: 10,
				responseEndTimestamp: 30,
				turnIndex: 0,
				toolCallCount: 0,
				toolCallNames: [],
				toolCalls: [],
			},
		]);
	});

	it("pairs tool results with the most recent pending tool call", () => {
		const resultText = "x".repeat(350);
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "Use tool" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "toolCall", name: "first", arguments: { a: 1 } },
						{ type: "toolCall", arguments: { b: 2 } },
					],
				},
			}),
			line({
				type: "message",
				message: { role: "toolResult", toolName: "fallback", content: [{ type: "text", text: resultText }] },
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 99 });

		expect(round.toolCallCount).toBe(2);
		expect(round.toolCallNames).toEqual(["first", "fallback"]);
		expect(round.responseSegments).toEqual([
			{ type: "toolCall", toolCallIndex: 0 },
			{ type: "toolCall", toolCallIndex: 1 },
		]);
		expect(round.toolCalls[0]).toMatchObject({ name: "first", arguments: '{"a":1}', result_summary: "" });
		expect(round.toolCalls[1]).toMatchObject({
			name: "unknown",
			arguments: '{"b":2}',
			result_summary: "x".repeat(300),
			result_full: resultText,
			result_truncated: false,
		});
	});

	it("uses the next user timestamp as the previous response end", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "First" }] },
			}),
			line({ type: "message", message: { role: "assistant", content: [{ type: "text", text: "First response" }] } }),
			line({
				type: "message",
				id: "u2",
				message: { role: "user", timestamp: 25, content: [{ type: "text", text: "Second" }] },
			}),
			line({
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: "Second response" }] },
			}),
		].join("\n");

		const rounds = parsePiSessionJsonl(raw, { now: () => 40 });

		expect(rounds.map((round) => round.responseEndTimestamp)).toEqual([25, 40]);
		expect(rounds.map((round) => round.turnIndex)).toEqual([0, 1]);
	});

	it("matches tool results by toolCallId when IDs are present (parallel calls)", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "Use tools" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "toolCall", id: "call_a", name: "alpha", arguments: { x: 1 } },
						{ type: "toolCall", id: "call_b", name: "beta", arguments: { y: 2 } },
					],
				},
			}),
			// Results arrive in REVERSED order (beta before alpha — parallel race)
			line({
				type: "message",
				message: {
					role: "toolResult",
					toolCallId: "call_b",
					toolName: "beta",
					content: [{ type: "text", text: "result B" }],
				},
			}),
			line({
				type: "message",
				message: {
					role: "toolResult",
					toolCallId: "call_a",
					toolName: "alpha",
					content: [{ type: "text", text: "result A" }],
				},
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 99 });

		expect(round.toolCalls).toHaveLength(2);
		expect(round.toolCalls[0]).toMatchObject({
			id: "call_a",
			name: "alpha",
			result_summary: "result A",
		});
		expect(round.toolCalls[1]).toMatchObject({
			id: "call_b",
			name: "beta",
			result_summary: "result B",
		});
	});

	it("falls back to sequential matching when toolCallId is absent", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "Use tools" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "toolCall", name: "first", arguments: { a: 1 } },
						{ type: "toolCall", name: "second", arguments: { b: 2 } },
					],
				},
			}),
			line({
				type: "message",
				message: { role: "toolResult", toolName: "second", content: [{ type: "text", text: "result 2" }] },
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 99 });

		expect(round.toolCalls).toHaveLength(2);
		expect(round.toolCalls[1].result_summary).toBe("result 2");
		expect(round.toolCalls[0].result_summary).toBe("");
	});

	it("can skip a short final response for bulk digestion", () => {
		const raw = [
			line({ type: "message", id: "u1", message: { role: "user", content: [{ type: "text", text: "Prompt" }] } }),
			line({ type: "message", message: { role: "assistant", content: [{ type: "text", text: "too short" }] } }),
		].join("\n");

		expect(parsePiSessionJsonl(raw, { skipShortFinalResponse: true, now: () => 1 })).toEqual([]);
		expect(parsePiSessionJsonl(raw, { now: () => 1 })).toHaveLength(1);
	});

	it("handles thinking blocks in assistant responses", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "Think about it" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [{ type: "thinking", text: "Hmm, let me think" }],
				},
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 99 });

		expect(round.responseSequence).toBe("[thinking] Hmm, let me think [/thinking]");
		expect(round.responseSegments).toEqual([{ type: "text", text: "[thinking] Hmm, let me think [/thinking]" }]);
	});

	it("handles mixed text and thinking blocks", () => {
		const raw = [
			line({ type: "message", id: "u1", message: { role: "user", content: [{ type: "text", text: "Go" }] } }),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "thinking", text: "Analyzing..." },
						{ type: "text", text: "Here's the answer" },
						{ type: "thinking", text: "Double-checking..." },
					],
				},
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 99 });

		expect(round.responseSequence).toBe(
			"[thinking] Analyzing... [/thinking]\n\nHere's the answer\n\n[thinking] Double-checking... [/thinking]",
		);
		expect(round.responseSegments).toEqual([
			{ type: "text", text: "[thinking] Analyzing... [/thinking]" },
			{ type: "text", text: "Here's the answer" },
			{ type: "text", text: "[thinking] Double-checking... [/thinking]" },
		]);
	});

	it("adds a session label when supplied", () => {
		const raw = [
			line({ type: "message", id: "u1", message: { role: "user", content: [{ type: "text", text: "Prompt" }] } }),
			line({
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: "Long enough response" }] },
			}),
		].join("\n");

		expect(parsePiSessionJsonl(raw, { sessionLabel: "session-a", now: () => 1 })[0].sessionLabel).toBe("session-a");
	});
});
