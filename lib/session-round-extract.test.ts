import { describe, expect, it } from "vitest";
import {
	extractSection,
	extractUserPromptText,
	hasPromptSectionMarkers,
	joinTextBlocks,
	looksLikePromptEnvelope,
} from "./session-round-extract.ts";

// ─────────────────────────────────────────────
// joinTextBlocks
// ─────────────────────────────────────────────

describe("joinTextBlocks", () => {
	it("joins text blocks and filters non-text", () => {
		expect(
			joinTextBlocks([{ type: "text", text: "hello" }, { type: "toolCall" }, { type: "text", text: "world" }]),
		).toBe("hello world");
	});

	it("returns empty for undefined/null/non-array", () => {
		expect(joinTextBlocks(undefined as unknown as Array<{ type?: string; text?: unknown }>)).toBe("");
	});

	it("ignores blocks with non-string text", () => {
		expect(
			joinTextBlocks([
				{ type: "text", text: 42 as unknown as string },
				{ type: "text", text: "valid" },
			]),
		).toBe("valid");
	});
});

// ─────────────────────────────────────────────
// looksLikePromptEnvelope
// ─────────────────────────────────────────────

describe("looksLikePromptEnvelope", () => {
	it("returns false for plain text", () => {
		expect(looksLikePromptEnvelope("What is the weather?")).toBe(false);
		expect(looksLikePromptEnvelope("  plain text  ")).toBe(false);
	});

	it("returns true for a JSON array of messages", () => {
		const input = JSON.stringify([
			{ role: "system", content: "You are a helpful assistant." },
			{ role: "user", content: "Hello" },
		]);
		expect(looksLikePromptEnvelope(input)).toBe(true);
	});

	it("returns true for a single message object", () => {
		expect(looksLikePromptEnvelope(JSON.stringify({ role: "user", content: "Hi" }))).toBe(true);
	});

	it("returns true for a messages wrapper", () => {
		expect(
			looksLikePromptEnvelope(
				JSON.stringify({
					messages: [
						{ role: "system", content: "sys" },
						{ role: "user", content: "Hello" },
					],
				}),
			),
		).toBe(true);
	});

	it("returns true for a content wrapper object", () => {
		expect(looksLikePromptEnvelope(JSON.stringify({ content: "some text" }))).toBe(true);
		expect(looksLikePromptEnvelope(JSON.stringify({ content: [{ type: "text", text: "hello" }] }))).toBe(true);
	});

	it("returns false for invalid JSON", () => {
		expect(looksLikePromptEnvelope("{invalid}")).toBe(false);
	});

	it("returns false for JSON that doesn't look like a message", () => {
		expect(looksLikePromptEnvelope('{"key": "value", "age": 42}')).toBe(false);
	});
});

// ─────────────────────────────────────────────
// hasPromptSectionMarkers / extractSection
// ─────────────────────────────────────────────

describe("prompt section markers", () => {
	it("detects section markers in text", () => {
		expect(hasPromptSectionMarkers("some text [ACTIONABLE PROMPT] more")).toBe(true);
		expect(hasPromptSectionMarkers("no markers here")).toBe(false);
	});

	it("extracts a named section", () => {
		const text = [
			"[SESSION ARCHITECTURE]",
			"Some architecture info",
			"[ACTIONABLE PROMPT]",
			"The actual prompt",
			"[ENVIRONMENT]",
			"Some env info",
		].join("\n");

		expect(extractSection(text, "[ACTIONABLE PROMPT]")).toBe("The actual prompt");
	});

	it("extracts a section that goes to end of string", () => {
		const text = "[ACTIONABLE PROMPT]\nJust this one section";
		expect(extractSection(text, "[ACTIONABLE PROMPT]")).toBe("Just this one section");
	});

	it("returns null for missing section", () => {
		expect(extractSection("plain text", "[ACTIONABLE PROMPT]")).toBe(null);
	});

	it("handles section headers without brackets", () => {
		const text = "[ACTIONABLE PROMPT]\nHello";
		expect(extractSection(text, "ACTIONABLE PROMPT")).toBe("Hello");
	});
});

// ─────────────────────────────────────────────
// extractUserPromptText — main sanitizer
// ─────────────────────────────────────────────

describe("extractUserPromptText", () => {
	it("returns plain text unchanged", () => {
		expect(extractUserPromptText("What is the weather?")).toBe("What is the weather?");
	});

	it("returns empty string unchanged", () => {
		expect(extractUserPromptText("")).toBe("");
		expect(extractUserPromptText("  ")).toBe("");
	});

	it("extracts last user message from array envelope", () => {
		const input = JSON.stringify([
			{ role: "system", content: "You are a helpful assistant." },
			{ role: "user", content: "First question" },
			{ role: "assistant", content: "First answer" },
			{ role: "user", content: "Second question" },
		]);
		expect(extractUserPromptText(input)).toBe("Second question");
	});

	it("extracts user message from single message object", () => {
		expect(extractUserPromptText(JSON.stringify({ role: "user", content: "Hello" }))).toBe("Hello");
	});

	it("extracts from messages wrapper", () => {
		const input = JSON.stringify({
			messages: [
				{ role: "system", content: "sys" },
				{ role: "user", content: "Actual prompt" },
			],
		});
		expect(extractUserPromptText(input)).toBe("Actual prompt");
	});

	it("extracts from content wrapper", () => {
		expect(extractUserPromptText(JSON.stringify({ content: "Hello" }))).toBe("Hello");
	});

	it("extracts from content wrapper with block array", () => {
		const input = JSON.stringify({ content: [{ type: "text", text: "Hello from blocks" }] });
		expect(extractUserPromptText(input)).toBe("Hello from blocks");
	});

	it("extracts [ACTIONABLE PROMPT] section from section-marked text", () => {
		const text = [
			"[SESSION ARCHITECTURE]",
			"Each round starts fresh by default.",
			"[SYSTEM]",
			"You are a coding assistant.",
			"[ACTIONABLE PROMPT]",
			"Please fix the bug in the parser.",
			"[ENVIRONMENT]",
			"CWD: /home/user/project",
		].join("\n");

		expect(extractUserPromptText(text)).toBe("Please fix the bug in the parser.");
	});

	it("returns original text if section markers present but no actionable section", () => {
		const text = "[SESSION ARCHITECTURE]\nSome architecture info";
		expect(extractUserPromptText(text)).toBe(text);
	});

	it("returns original text for single envelpe with no recognizable structure", () => {
		// Object with "age" but no role/content/messages — not a message-like object
		expect(extractUserPromptText('{"age": 42}')).toBe('{"age": 42}');
	});

	it("returns original text for non-envelope, non-section text", () => {
		expect(extractUserPromptText("Just a normal question")).toBe("Just a normal question");
	});

	it("handles messages with array content blocks", () => {
		const input = JSON.stringify([
			{ role: "system", content: [{ type: "text", text: "sys msg" }] },
			{
				role: "user",
				content: [
					{ type: "text", text: "Hello" },
					{ type: "text", text: "World" },
				],
			},
		]);
		expect(extractUserPromptText(input)).toBe("Hello World");
	});

	it("handles array-of-messages envelope with no user message", () => {
		const input = JSON.stringify([
			{ role: "system", content: "sys" },
			{ role: "assistant", content: "assistant" },
		]);
		// Falls back to original text when no user message found
		expect(extractUserPromptText(input)).toBe(input);
	});

	it("returns original messages wrapper without user messages", () => {
		const input = JSON.stringify({
			messages: [
				{ role: "system", content: "sys" },
				{ role: "assistant", content: "ans" },
			],
		});
		expect(extractUserPromptText(input)).toBe(input);
	});

	it("handles JSON with content as undefined-like value", () => {
		// content wrapper where content is not a string or array
		expect(extractUserPromptText('{"content": 42}')).toBe('{"content": 42}');
	});
});
