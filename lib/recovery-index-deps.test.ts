import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { bm25IndexPathForRoundsDir, loadOrRebuildBm25Index } from "./bm25-index.ts";
import { createRecoveryIndexDeps } from "./recovery-index-deps.ts";
import { loadToolIndex, toolIndexPathForRoundsDir } from "./search-tools.ts";
import { indexRecoveredRounds } from "./session-backfill.ts";

type RecoveryFs = Pick<typeof fs, "existsSync" | "mkdirSync" | "readFileSync" | "readdirSync" | "writeFileSync">;

function tmpDir() {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-recovery-deps-"));
}

function writeRound(roundsDir: string, fileName: string, data: object): void {
	fs.writeFileSync(path.join(roundsDir, fileName), JSON.stringify(data));
}

function writeSpy(written: string[]): RecoveryFs {
	return {
		existsSync: fs.existsSync,
		mkdirSync: fs.mkdirSync,
		readFileSync: fs.readFileSync,
		readdirSync: fs.readdirSync,
		writeFileSync: (p, data) => {
			written.push(String(p));
			fs.writeFileSync(p, data);
		},
	};
}

describe("createRecoveryIndexDeps (PR !138 F3: production wiring)", () => {
	it("persists the bm25 index exactly once per batch and appends tool rows with the real fs", () => {
		const roundsDir = tmpDir();
		const indexPath = bm25IndexPathForRoundsDir(roundsDir);
		const toolPath = toolIndexPathForRoundsDir(roundsDir);
		writeRound(roundsDir, "a.json", { userPrompt: "alpha keyword_one", responseSequence: "answer one" });
		writeRound(roundsDir, "b.json", {
			userPrompt: "beta keyword_two",
			responseSequence: "answer two",
			toolCalls: [{ name: "read", index: 0, arguments: "{}", result_summary: "file content" }],
		});

		const written: string[] = [];
		const fsImpl = writeSpy(written);
		let cached: ReturnType<typeof loadOrRebuildBm25Index> | null = null;
		const deps = createRecoveryIndexDeps({
			roundsDir,
			bm25IndexPath: indexPath,
			toolIndexPath: toolPath,
			loadBm25Index: () => (cached ??= loadOrRebuildBm25Index(indexPath, roundsDir, fsImpl)),
			fsImpl,
		});

		const report = indexRecoveredRounds(["a.json", "b.json", "missing.json"], deps);

		// Exactly-once contract (issue #137): the batch committed the sidecar
		// with a single real file write; unreadable rounds error out and never
		// trigger a write.
		expect(written).toEqual([indexPath]);
		expect(report.bm25Indexed).toBe(2);
		expect(report.toolIndexed).toBe(1);
		expect(report.errors).toEqual(["missing.json: unreadable round file"]);

		const persisted = JSON.parse(fs.readFileSync(indexPath, "utf-8"));
		expect(Object.keys(persisted.documents).sort()).toEqual(["a.json", "b.json"]);
		expect(loadToolIndex(toolPath).map((row) => row.toolName)).toEqual(["read"]);
	});

	it("writes nothing when no round was bm25-indexed", () => {
		const roundsDir = tmpDir();
		const indexPath = bm25IndexPathForRoundsDir(roundsDir);
		const written: string[] = [];
		const fsImpl = writeSpy(written);
		let cached: ReturnType<typeof loadOrRebuildBm25Index> | null = null;
		const deps = createRecoveryIndexDeps({
			roundsDir,
			bm25IndexPath: indexPath,
			toolIndexPath: toolIndexPathForRoundsDir(roundsDir),
			loadBm25Index: () => (cached ??= loadOrRebuildBm25Index(indexPath, roundsDir, fsImpl)),
			fsImpl,
		});

		const report = indexRecoveredRounds(["missing.json"], deps);

		expect(report.bm25Indexed).toBe(0);
		expect(written).toEqual([]);
		expect(fs.existsSync(indexPath)).toBe(false);
	});
});
