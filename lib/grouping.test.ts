import { describe, expect, it } from "vitest";
import { assignToGroup, formatGroupStats, parseGroupThreshold } from "./grouping.ts";

describe("semantic grouping", () => {
	it("parses group threshold with fallback", () => {
		expect(parseGroupThreshold("0.42")).toBe(0.42);
		expect(parseGroupThreshold("0")).toBe(0);
		expect(parseGroupThreshold("1")).toBe(1);
		expect(parseGroupThreshold(undefined)).toBe(0.77);
		expect(parseGroupThreshold("bad", 0.5)).toBe(0.5);
		expect(parseGroupThreshold("-1", 0.5)).toBe(0.5);
		expect(parseGroupThreshold("2", 0.5)).toBe(0.5);
	});

	it("formats group stats", () => {
		expect(
			formatGroupStats(
				[
					{ centroid: [1, 0], rounds: ["a"] },
					{ centroid: [0, 1], rounds: ["b", "c"] },
				],
				0.77,
			),
		).toBe("THR: 77%; #groups: 2, #topics: {1,2}");
	});

	it("creates a new group when no centroid matches", () => {
		const groups: Array<{ centroid: number[]; rounds: string[] }> = [];

		expect(assignToGroup(groups, "a", [1, 0], 0.8)).toBe(0);
		expect(groups).toEqual([{ centroid: [1, 0], rounds: ["a"] }]);
		expect(assignToGroup(groups, "b", [0, 1], 0.8)).toBe(1);
		expect(groups[1]).toEqual({ centroid: [0, 1], rounds: ["b"] });
	});

	it("adds to the best matching group and updates the centroid", () => {
		const groups = [
			{ centroid: [1, 0], rounds: ["a"] },
			{ centroid: [0.6, 0.8], rounds: ["b"] },
		];

		expect(assignToGroup(groups, "c", [0.6, 0.8], 0.7)).toBe(1);
		expect(groups[1].rounds).toEqual(["b", "c"]);
		expect(groups[1].centroid).toEqual([0.6, 0.8]);
	});

	it("creates a new singleton group when vec is null", () => {
		const groups: Array<{ centroid: number[]; rounds: string[] }> = [{ centroid: [1, 0], rounds: ["a"] }];

		expect(assignToGroup(groups, "b", null, 0.8)).toBe(1);
		expect(groups).toHaveLength(2);
		expect(groups[1].centroid).toEqual([]);
		expect(groups[1].rounds).toEqual(["b"]);
	});

	it("uses forceGroupIdx with null vec (does not update centroid)", () => {
		const groups: Array<{ centroid: number[]; rounds: string[] }> = [{ centroid: [1, 0, 0], rounds: ["a"] }];

		expect(assignToGroup(groups, "b", null, 0.8, 0)).toBe(0);
		expect(groups[0].rounds).toEqual(["a", "b"]);
		expect(groups[0].centroid).toEqual([1, 0, 0]); // unchanged
	});
});
