import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { encodeVectorIndexLine } from "../lib/index-io.ts";
import type { SemblrConfig } from "../lib/semblr-config.ts";
import {
	canonicalDuplicateKey,
	pruneRecoveryDuplicates,
	type RoundRecord,
	selectPrunes,
} from "./prune-recovery-duplicates.ts";

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

describe("pruneRecoveryDuplicates scan filter", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
	});

	it("skips index files and keeps empty rounds as the live keeper", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "prune-"));
		dirs.push(dir);
		// An index-like file whose canonical key would be empty content.
		fs.writeFileSync(path.join(dir, "index.bm25.json"), JSON.stringify({ documents: {} }));
		// A genuinely empty round that would otherwise group with the index file.
		fs.writeFileSync(
			path.join(dir, "empty-round.json"),
			JSON.stringify({ userPrompt: "", responseSequence: "", recovered: true }),
		);
		const lines: string[] = [];
		await pruneRecoveryDuplicates({
			roundsDir: dir,
			indexPath: path.join(dir, "index.bm25.json"),
			cutoffMs: 0,
			args: [],
			out: (line) => lines.push(line),
		});
		const joined = lines.join("\n");
		// Only the round was scanned; the index file is excluded.
		expect(joined).toContain("Rounds scanned: 1");
		// Pre-fix, index.bm25.json (non-recovered, empty-content key) would group
		// with the empty round and become the live keeper. Post-fix: singleton, no plan.
		expect(joined).toContain("Duplicate groups with prunable recovered twins: 0");
		// The index file survived even with --apply semantics (cutoff 0).
		expect(fs.existsSync(path.join(dir, "index.bm25.json"))).toBe(true);
	});
});

describe("pruneRecoveryDuplicates shard awareness (#149)", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
	});

	function testConfig(roundsDir: string): SemblrConfig {
		const agentDir = path.join(roundsDir, "agent");
		return {
			agentDir,
			embeddingProvider: "openrouter",
			embeddingModel: "openai/text-embedding-3-small",
			embeddingMaxTokens: 8000,
			roundsDir,
			indexPath: path.join(roundsDir, "index.csv"),
			hostId: "test-host",
			groupThreshold: 0.77,
			minSimilarity: 0.3,
			embedTimeoutMs: 15_000,
			embedMaxRetries: 3,
			embedBackoffMs: 1000,
			hybridSemanticWeight: 0.7,
			summaryThresholdExtra: 40_000,
			contextPromptHeadChars: 260,
			contextPromptTailChars: 140,
			contextBudgetRatio: 0.08,
			contextRelevanceMaxEntries: 20,
			contextRecencyMaxEntries: 20,
			promptNoiseFenceMaxChars: 600,
			promptNoiseJsonMaxChars: 600,
			promptNoiseRepeatMaxChars: 200,
		};
	}

	const row = (file: string): string => encodeVectorIndexLine([0.1, 0.2], `${file}:prompt`);

	it("rewrites the tool index atomically (tmp+rename, decision 015)", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "prune-tools-"));
		dirs.push(dir);
		fs.writeFileSync(path.join(dir, "live.json"), JSON.stringify({ userPrompt: "q", responseSequence: "a" }));
		fs.writeFileSync(
			path.join(dir, "dup.json"),
			JSON.stringify({ userPrompt: "q", responseSequence: "a", recovered: true }),
		);
		const ownShard = path.join(dir, "index.host-a");
		fs.writeFileSync(ownShard, `${[row("live.json"), row("dup.json")].join("\n")}\n`);
		const toolIndex = path.join(dir, "index-tools.fulltext.csv");
		fs.writeFileSync(toolIndex, "live.json,abc,0,bash\ndup.json,def,0,bash\n");
		const inodeBefore = fs.statSync(toolIndex).ino;

		await pruneRecoveryDuplicates({
			config: testConfig(dir),
			roundsDir: dir,
			cutoffMs: 0,
			args: ["--apply"],
			hostname: "host-a",
			out: () => {},
		});

		const toolIndexAfter = fs.readFileSync(toolIndex, "utf-8");
		expect(toolIndexAfter).toContain("live.json");
		expect(toolIndexAfter).not.toContain("dup.json");
		// A published rename installs a new file (new inode); an in-place write
		// would keep the old inode.
		expect(fs.statSync(toolIndex).ino).not.toBe(inodeBefore);
		// No temp artifact is left behind.
		expect(fs.readdirSync(dir).filter((name) => name.includes(".tmp."))).toEqual([]);
	});

	it("prunes only the duplicates this host owns and cleans its own shard", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "prune-shards-"));
		dirs.push(dir);
		// Duplicate pair #1 — a foreign shard still references the recovered twin.
		fs.writeFileSync(path.join(dir, "live.json"), JSON.stringify({ userPrompt: "q", responseSequence: "a" }));
		fs.writeFileSync(
			path.join(dir, "dup.json"),
			JSON.stringify({ userPrompt: "q", responseSequence: "a", recovered: true }),
		);
		// Duplicate pair #2 — only this host's shard references the twin.
		fs.writeFileSync(path.join(dir, "live2.json"), JSON.stringify({ userPrompt: "q2", responseSequence: "b" }));
		fs.writeFileSync(
			path.join(dir, "dup2.json"),
			JSON.stringify({ userPrompt: "q2", responseSequence: "b", recovered: true }),
		);

		const ownShard = path.join(dir, "index.host-a");
		const foreignShard = path.join(dir, "index.host-b");
		fs.writeFileSync(
			ownShard,
			`${[row("live.json"), row("dup.json"), row("live2.json"), row("dup2.json")].join("\n")}\n`,
		);
		fs.writeFileSync(foreignShard, `${row("dup.json")}\n`);
		const foreignBefore = fs.readFileSync(foreignShard, "utf-8");

		const lines: string[] = [];
		await pruneRecoveryDuplicates({
			config: testConfig(dir),
			roundsDir: dir,
			cutoffMs: 0,
			args: ["--apply"],
			hostname: "host-a",
			out: (line) => lines.push(line),
		});

		// The twin a foreign shard references is kept: its row cannot be rewritten here.
		expect(fs.existsSync(path.join(dir, "dup.json"))).toBe(true);
		// The twin only this host's shard references is pruned, and its row is dropped.
		expect(fs.existsSync(path.join(dir, "dup2.json"))).toBe(false);
		const ownAfter = fs.readFileSync(ownShard, "utf-8");
		expect(ownAfter).toContain("dup.json");
		expect(ownAfter).toContain("live2.json");
		expect(ownAfter).not.toContain("dup2.json");
		// The foreign shard is byte-identical.
		expect(fs.readFileSync(foreignShard, "utf-8")).toBe(foreignBefore);

		const joined = lines.join("\n");
		expect(joined).toContain("kept (referenced by a foreign shard): dup.json");
		expect(joined).toContain("Foreign shards left untouched: index.host-b (1 rows)");
	});
});
