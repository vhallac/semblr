import { describe, expect, it } from "vitest";
import { estimateMessagesTokens, estimateTokens } from "./tokens.ts";

describe("token estimation", () => {
	it("uses one token per four characters rounded up", () => {
		expect(estimateTokens("")).toBe(0);
		expect(estimateTokens("abcd")).toBe(1);
		expect(estimateTokens("abcde")).toBe(2);
	});

	describe("estimateMessagesTokens", () => {
		it("returns 0 for empty array", () => {
			expect(estimateMessagesTokens([])).toBe(0);
		});

		it("counts tokens from string content", () => {
			expect(
				estimateMessagesTokens([
					{ role: "user", content: "abcdefgh" }, // 2 tokens (8 chars / 4)
					{ role: "assistant", content: "hello world" }, // 3 tokens (11 / 4 ceiling)
				]),
			).toBe(5);
		});

		it("counts tokens from content block arrays", () => {
			expect(
				estimateMessagesTokens([
					{
						role: "user",
						content: [
							{ type: "text", text: "abcdefgh" },
							{ type: "text", text: "1234" },
						],
					},
				]),
			).toBe(3); // 2 + 1 = 3
		});

		it("skips null/undefined content", () => {
			expect(
				estimateMessagesTokens([
					{ role: "user", content: null },
					{ role: "assistant" },
				]),
			).toBe(0);
		});

		it("handles mixed string and block content", () => {
			expect(
				estimateMessagesTokens([
					{ role: "system", content: "abcd" }, // 1
					{ role: "user", content: [{ type: "text", text: "12345678" }] }, // 2
					{ role: "assistant", content: "1234" }, // 1
				]),
			).toBe(4);
		});
	});
});
