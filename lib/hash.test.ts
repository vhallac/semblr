import { describe, expect, it } from "vitest";
import { computeContentHash, createRoundFilePath } from "./hash.ts";

describe("content hashing", () => {
	it("hashes prompt and response deterministically", () => {
		expect(computeContentHash("prompt", "response")).toBe("e8066ff7f4947a11817f1feeb4d14e32");
		expect(createRoundFilePath("prompt", "response")).toBe("e8066ff7f4947a11817f1feeb4d14e32.json");
	});

	it("includes tool arguments and full results when present", () => {
		const withSummary = computeContentHash("p", "r", [{ arguments: "{}", result_summary: "summary" }]);
		const withFull = computeContentHash("p", "r", [
			{ arguments: "{}", result_summary: "summary", result_full: "full" },
		]);

		expect(withSummary).not.toBe(computeContentHash("p", "r"));
		expect(withFull).not.toBe(withSummary);
		expect(withFull).toBe(computeContentHash("p", "r", [{ arguments: "{}", result_full: "full" }]));
	});
});
