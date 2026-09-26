import { describe, expect, it } from "vitest";
import { splitAndExtractPrompt } from "./envelope-extract.ts";

const messages = [
	{ role: "system", content: "You are an assistant." },
	{ role: "user", content: [{ type: "text", text: "[ENVIRONMENT]\nreal user text here" }] },
];

describe("splitAndExtractPrompt", () => {
	it("passes clean text through untouched (idempotent no-op)", () => {
		const text = "Let's start working on #83";
		const result = splitAndExtractPrompt(text);
		expect(result.userText).toBe(text);
		expect(result.wasEnvelope).toBe(false);
	});

	it("extracts the real user text from a bare JSON-stringified message array", () => {
		const text = JSON.stringify(messages);
		const result = splitAndExtractPrompt(text);
		expect(result.wasEnvelope).toBe(true);
		expect(result.userText).toBe("[ENVIRONMENT]\nreal user text here");
	});

	it("extracts user content from a bare JSON-stringified single user message", () => {
		const text = JSON.stringify({ role: "user", content: "hello there" });
		const result = splitAndExtractPrompt(text);
		expect(result.wasEnvelope).toBe(true);
		expect(result.userText).toBe("hello there");
	});

	it("collapses a fenced envelope to a placeholder, keeping trailing user text", () => {
		const fence = `\`\`\`\n${JSON.stringify(messages, null, 2)}\n\`\`\`\nThis is only the input part`;
		const result = splitAndExtractPrompt(fence);
		expect(result.wasEnvelope).toBe(true);
		expect(result.userText).toContain("This is only the input part");
		expect(result.userText).not.toContain("You are an assistant");
		expect(result.userText).toMatch(/\[MESSAGE_ENVELOPE: ~\d+ chars\]/);
	});

	it("keeps leading prose around a fenced envelope", () => {
		const fence = `But I had this capture a second ago:\n\n\`\`\`\n${JSON.stringify(messages, null, 2)}\n\`\`\``;
		const result = splitAndExtractPrompt(fence);
		expect(result.wasEnvelope).toBe(true);
		expect(result.userText).toContain("But I had this capture a second ago:");
		expect(result.userText).not.toContain('"role": "system"');
	});

	it("does not collapse a fence whose body is not a message envelope", () => {
		const text = "```\nsome ordinary code\nconst x = 1;\n```\nafter";
		const result = splitAndExtractPrompt(text);
		expect(result.wasEnvelope).toBe(false);
		expect(result.userText).toBe(text);
	});

	it("does not treat non-message JSON as an envelope", () => {
		const text = JSON.stringify({ foo: "bar", baz: [1, 2, 3] });
		const result = splitAndExtractPrompt(text);
		expect(result.wasEnvelope).toBe(false);
		expect(result.userText).toBe(text);
	});

	it("falls back to a placeholder when an envelope array has no user message", () => {
		const text = JSON.stringify([{ role: "system", content: "sys prompt" }]);
		const result = splitAndExtractPrompt(text);
		expect(result.wasEnvelope).toBe(true);
		expect(result.userText).toMatch(/^\[MESSAGE_ENVELOPE: ~\d+ chars\]$/);
	});

	it("handles an object with a messages array", () => {
		const text = JSON.stringify({ messages });
		const result = splitAndExtractPrompt(text);
		expect(result.wasEnvelope).toBe(true);
		expect(result.userText).toBe("[ENVIRONMENT]\nreal user text here");
	});
});
