import { describe, expect, it } from "vitest";
import { extractText } from "./message-content.ts";

describe("extractText", () => {
	it("extracts text from a single text content block", () => {
		const result = extractText([{ type: "text", text: "Hello world" }]);
		expect(result).toBe("Hello world");
	});

	it("joins multiple text blocks with spaces", () => {
		const result = extractText([
			{ type: "text", text: "Hello" },
			{ type: "text", text: "world" },
		]);
		expect(result).toBe("Hello world");
	});

	it("skips blocks without text", () => {
		const result = extractText([{ type: "text" }]);
		expect(result).toBe("");
	});

	it("skips non-text blocks", () => {
		const result = extractText([
			{ type: "image", text: "should not appear" },
			{ type: "text", text: "visible" },
		]);
		expect(result).toBe("visible");
	});

	it("returns empty string for empty array", () => {
		expect(extractText([])).toBe("");
	});

	// Bug #94: thinking-only blocks are silently dropped
	it("extracts text from thinking blocks with [thinking] marker", () => {
		const result = extractText([{ type: "thinking", text: "Let me reason about this..." }]);
		expect(result).toBe("[thinking] Let me reason about this... [/thinking]");
	});

	it("extracts from mixed text and thinking blocks", () => {
		const result = extractText([
			{ type: "thinking", text: "I need to check something..." },
			{ type: "text", text: "The answer is 42." },
			{ type: "thinking", text: "Actually, double-checking..." },
		]);
		expect(result).toBe(
			"[thinking] I need to check something... [/thinking] The answer is 42. [thinking] Actually, double-checking... [/thinking]",
		);
	});

	it("skips thinking blocks without text", () => {
		const result = extractText([{ type: "thinking" }]);
		expect(result).toBe("");
	});
});
