import { describe, expect, it } from "vitest";
import { estimateTokens } from "./tokens.ts";

describe("token estimation", () => {
	it("uses one token per four characters rounded up", () => {
		expect(estimateTokens("")).toBe(0);
		expect(estimateTokens("abcd")).toBe(1);
		expect(estimateTokens("abcde")).toBe(2);
	});
});
