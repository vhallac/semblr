import { describe, expect, it } from "vitest";
import { cosineSimilarity, normalize } from "./vector.ts";

describe("vector helpers", () => {
	it("normalizes non-zero vectors", () => {
		expect(normalize([3, 4])).toEqual([0.6, 0.8]);
	});

	it("returns zero vectors unchanged", () => {
		const zero = [0, 0];
		expect(normalize(zero)).toBe(zero);
	});

	it("computes cosine similarity for normalized vectors", () => {
		expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
		expect(cosineSimilarity([0.6, 0.8], [0.6, 0.8])).toBeCloseTo(1);
	});
});
