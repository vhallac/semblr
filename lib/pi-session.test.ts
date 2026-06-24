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

	it("can skip a short final response for bulk digestion", () => {
		const raw = [
			line({ type: "message", id: "u1", message: { role: "user", content: [{ type: "text", text: "Prompt" }] } }),
			line({ type: "message", message: { role: "assistant", content: [{ type: "text", text: "too short" }] } }),
		].join("\n");

		expect(parsePiSessionJsonl(raw, { skipShortFinalResponse: true, now: () => 1 })).toEqual([]);
		expect(parsePiSessionJsonl(raw, { now: () => 1 })).toHaveLength(1);
	});

	it("sanitizes user prompt containing a JSON-stringified prompt envelope", () => {
		// Pi user message content containing the full prompt envelope as a text block
		const envelope = JSON.stringify([
			{ role: "system", content: "You are a helpful assistant." },
			{ role: "user", content: "What is the capital of France?" },
		]);
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: {
					role: "user",
					timestamp: 10,
					content: [{ type: "text", text: envelope }],
				},
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Long enough answer" }],
				},
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 30 });

		// Expected: the envelope is distilled to just the last user message's text
		expect(round.userPrompt).toBe("What is the capital of France?");
		expect(round.responseSequence).toBe("Long enough answer");
		expect(round.toolCalls).toEqual([]);
	});

	it("sanitizes user prompt where envelope is the sole content block with extra whitespace", () => {
		// Single text block that is just a JSON envelope — should be distilled
		const envelope = JSON.stringify({ role: "user", content: "What is the weather?" });
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: {
					role: "user",
					timestamp: 10,
					content: [{ type: "text", text: `  ${envelope}  ` }],
				},
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Long enough answer" }],
				},
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 30 });

		// The envelope is distilled to just the user message's content
		expect(round.userPrompt).toBe("What is the weather?");
	});

	it("preserves plain user prompt when no envelope is present", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: {
					role: "user",
					timestamp: 10,
					content: [{ type: "text", text: "Just a normal question" }],
				},
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Long enough answer" }],
				},
			}),
		].join("\n");

		const [round] = parsePiSessionJsonl(raw, { now: () => 30 });
		expect(round.userPrompt).toBe("Just a normal question");
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
