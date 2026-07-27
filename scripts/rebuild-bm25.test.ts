import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { bm25IndexPathForRoundsDir, loadBm25Index, scoreBm25Query } from "../lib/bm25-index.ts";
import { runRebuildBm25 } from "./rebuild-bm25.ts";

function tmpDir() {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-rebuild-bm25-"));
}

describe("rebuild-bm25 script", () => {
	it("rebuilds index.bm25.json from existing round files without embedding", async () => {
		const roundsDir = tmpDir();
		fs.writeFileSync(
			path.join(roundsDir, "exact.json"),
			JSON.stringify({
				userPrompt: "Why does search_interactions miss get_round_details?",
				responseSequence: "The fix is a BM25 sidecar index.",
				turnIndex: 0,
			}),
		);
		fs.writeFileSync(path.join(roundsDir, "index.csv"), "vector,exact.json:prompt\n");
		const stdout: string[] = [];

		await expect(
			runRebuildBm25({
				roundsDir,
				stdout: { log: (line: string) => stdout.push(line) },
			}),
		).resolves.toBe(0);

		const bm25IndexPath = bm25IndexPathForRoundsDir(roundsDir);
		const index = loadBm25Index(bm25IndexPath);
		expect(stdout).toEqual([`Rebuilt BM25 index: 1 rounds at ${bm25IndexPath}`]);
		expect(scoreBm25Query(index, "get_round_details").get("exact.json")).toBeGreaterThan(0);
	});

	it("reports a missing rounds directory", async () => {
		const roundsDir = path.join(tmpDir(), "missing");
		const stderr: string[] = [];

		await expect(runRebuildBm25({ roundsDir, stderr: { error: (line: string) => stderr.push(line) } })).resolves.toBe(
			1,
		);

		expect(stderr).toEqual([`Rounds directory does not exist: ${roundsDir}`]);
	});
});
