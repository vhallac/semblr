import { describe, expect, it } from "vitest";
import { canonicalDuplicateKey, type RoundRecord, selectPrunes } from "./prune-recovery-duplicates.ts";

function rec(filename: string, key: string, recovered: boolean, mtimeMs: number): RoundRecord {
	return { filename, key, recovered, mtimeMs };
}

describe("canonicalDuplicateKey", () => {
	it("treats trailing-whitespace-only differences as duplicates (the backfill trim bug)", () => {
		const live = {
			userPrompt: "q",
			responseSequence: "a",
			toolCalls: [{ name: "bash", arguments: "{}", result_full: "out\n" }],
		};
		const recovered = {
			userPrompt: "q",
			responseSequence: "a",
			toolCalls: [{ name: "bash", arguments: "{}", result_full: "out" }],
		};
		expect(canonicalDuplicateKey(recovered)).toBe(canonicalDuplicateKey(live));
	});

	it("distinguishes genuinely different rounds", () => {
		expect(canonicalDuplicateKey({ userPrompt: "a" })).not.toBe(canonicalDuplicateKey({ userPrompt: "b" }));
	});
});

describe("selectPrunes", () => {
	const KEY = "k1";

	it("keeps the live round and prunes recovered twins within the cutoff", () => {
		const plans = selectPrunes(
			[rec("live.json", KEY, false, 1000), rec("dup1.json", KEY, true, 2000), rec("dup2.json", KEY, true, 3000)],
			0,
		);
		expect(plans).toEqual([{ keeper: "live.json", prunes: ["dup1.json", "dup2.json"] }]);
	});

	it("falls back to the earliest recovered round when no live twin exists", () => {
		const plans = selectPrunes([rec("old.json", KEY, true, 1000), rec("new.json", KEY, true, 2000)], 0);
		expect(plans).toEqual([{ keeper: "old.json", prunes: ["new.json"] }]);
	});

	it("never prunes rounds older than the cutoff", () => {
		const plans = selectPrunes(
			[
				rec("live.json", KEY, false, 1000),
				rec("ancient-dup.json", KEY, true, 1500),
				rec("fresh-dup.json", KEY, true, 5000),
			],
			2000,
		);
		expect(plans).toEqual([{ keeper: "live.json", prunes: ["fresh-dup.json"] }]);
	});

	it("never prunes a live round even when it is newer than the keeper", () => {
		const plans = selectPrunes([rec("old-live.json", KEY, false, 1000), rec("new-live.json", KEY, false, 9000)], 0);
		expect(plans).toEqual([]);
	});

	it("ignores singletons", () => {
		expect(selectPrunes([rec("solo.json", KEY, true, 1000)], 0)).toEqual([]);
	});
});
