import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { LEGACY_BM25_FILENAME, loadBm25Index, ownBm25IndexPath, scoreBm25Query } from "../lib/bm25-index.ts";
import { runRebuildBm25 } from "./rebuild-bm25.ts";

function tmpDir() {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-rebuild-bm25-"));
	dirs.push(dir);
	return dir;
}
const dirs: string[] = [];
afterEach(() => {
	for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
});

describe("rebuild-bm25 script", () => {
	it("rebuilds the host's own BM25 artifact from existing round files without embedding", async () => {
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
				hostname: "host-a",
				stdout: { log: (line: string) => stdout.push(line) },
			}),
		).resolves.toBe(0);

		const bm25IndexPath = ownBm25IndexPath(roundsDir, "host-a");
		const index = loadBm25Index(bm25IndexPath);
		expect(stdout).toEqual([`Rebuilt BM25 index: 1 rounds at ${bm25IndexPath}`]);
		expect(scoreBm25Query(index, "get_round_details").get("exact.json")).toBeGreaterThan(0);
	});

	it("adopts a legacy index.bm25.json into the host's own artifact by rename (decision 024)", async () => {
		const roundsDir = tmpDir();
		fs.writeFileSync(
			path.join(roundsDir, "exact.json"),
			JSON.stringify({
				userPrompt: "Why does search_interactions miss get_round_details?",
				responseSequence: "The fix is a BM25 sidecar index.",
				turnIndex: 0,
			}),
		);
		fs.writeFileSync(path.join(roundsDir, LEGACY_BM25_FILENAME), JSON.stringify({ documents: {} }));

		await expect(runRebuildBm25({ roundsDir, hostname: "host-a" })).resolves.toBe(0);

		const ownPath = ownBm25IndexPath(roundsDir, "host-a");
		expect(fs.existsSync(path.join(roundsDir, LEGACY_BM25_FILENAME))).toBe(false);
		expect(fs.existsSync(ownPath)).toBe(true);
		expect(scoreBm25Query(loadBm25Index(ownPath), "get_round_details").get("exact.json")).toBeGreaterThan(0);
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
