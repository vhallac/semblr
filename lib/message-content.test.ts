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
});
