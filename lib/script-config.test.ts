import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readForeignBacklog } from "./foreign-backlog.ts";
import { encodeVectorIndexLine } from "./index-io.ts";
import { appendLineWithLock } from "./index-storage.ts";
import { reportForeignShards, resolveScriptIndexContext } from "./script-config.ts";
import { loadSemblrConfig, type SemblrConfig } from "./semblr-config.ts";

describe("resolveScriptIndexContext", () => {
	let root = "";
	let roundsDir = "";
	let config: SemblrConfig;

	beforeEach(() => {
		root = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-script-index-"));
		roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		config = {
			...loadSemblrConfig({ cwd: root, agentDir: path.join(root, "agent"), env: {} }),
			roundsDir,
			indexPath: path.join(roundsDir, "index.csv"),
		};
	});

	afterEach(() => {
		fs.rmSync(root, { recursive: true, force: true });
	});

	it("resolves the shard root from the rounds dir by default", () => {
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });
		expect(index.shardRoot).toBe(roundsDir);
		expect(index.ownShardPath).toBe(path.join(roundsDir, "index.host-a"));
		expect(index.indexPath).toBe(path.join(roundsDir, "index.csv"));
	});

	it("names the own shard after the configured host id when no override is given", () => {
		const index = resolveScriptIndexContext({ ...config, hostId: "workstation-1" }, roundsDir);
		expect(index.ownShardPath).toBe(path.join(roundsDir, "index.workstation-1"));
	});

	it("honors a custom rounds dir (issue #149 ext 4.a)", () => {
		const otherDir = path.join(root, "other-rounds");
		fs.mkdirSync(otherDir, { recursive: true });
		const index = resolveScriptIndexContext(config, otherDir, undefined, { hostname: "host-a" });
		expect(index.shardRoot).toBe(otherDir);
		expect(index.ownShardPath).toBe(path.join(otherDir, "index.host-a"));
	});

	it("honors an explicit index path (issue #149 ext 4.a)", () => {
		const indexDir = path.join(root, "index-dir");
		fs.mkdirSync(indexDir, { recursive: true });
		const index = resolveScriptIndexContext(config, roundsDir, path.join(indexDir, "custom.csv"), {
			hostname: "host-a",
		});
		expect(index.shardRoot).toBe(indexDir);
		expect(index.ownShardPath).toBe(path.join(indexDir, "index.host-a"));
	});

	it("reads the merged view across every shard", () => {
		fs.writeFileSync(path.join(roundsDir, "index.host-a"), `${encodeVectorIndexLine([1], "a.json:prompt")}\n`);
		fs.writeFileSync(path.join(roundsDir, "index.host-b"), `${encodeVectorIndexLine([2], "b.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		expect(index.shards.map((shard) => path.basename(shard))).toEqual(["index.host-a", "index.host-b"]);
		expect(index.readMergedLines()).toHaveLength(2);
		expect(index.readMergedEntries().map((entry) => entry.filePath)).toEqual(["a.json:prompt", "b.json:prompt"]);
		// The own shard alone holds only this host's rows.
		expect(index.readOwnLines()).toHaveLength(1);
	});

	it("appends rows to the host's own shard only", () => {
		const foreignPath = path.join(roundsDir, "index.host-b");
		fs.writeFileSync(foreignPath, `${encodeVectorIndexLine([2], "b.json:prompt")}\n`);
		const foreignBefore = fs.readFileSync(foreignPath, "utf-8");
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		index.appendRow("new.json:prompt", [1], "test-model");

		expect(fs.readFileSync(foreignPath, "utf-8")).toBe(foreignBefore);
		expect(index.readOwnLines()).toHaveLength(1);
		expect(index.readMergedLines()).toHaveLength(2);
	});

	it("adopts a legacy index.csv on the first write (Q1 option b)", () => {
		fs.writeFileSync(path.join(roundsDir, "index.csv"), `${encodeVectorIndexLine([1], "legacy.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		index.appendRow("new.json:prompt", [2], "test-model");

		expect(fs.existsSync(path.join(roundsDir, "index.csv"))).toBe(false);
		expect(index.readMergedLines()).toHaveLength(2);
		expect(index.readOwnLines()).toHaveLength(2);
	});

	it("rewrites only the own shard (a foreign shard stays byte-identical)", () => {
		const foreignPath = path.join(roundsDir, "index.host-b");
		fs.writeFileSync(foreignPath, `${encodeVectorIndexLine([2], "b.json:prompt")}\n`);
		const foreignBefore = fs.readFileSync(foreignPath, "utf-8");
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);

		expect(fs.readFileSync(foreignPath, "utf-8")).toBe(foreignBefore);
		expect(index.readOwnLines()).toHaveLength(1);
		expect(index.readMergedLines()).toHaveLength(2);
	});

	it("claims a legacy index.csv before a wholesale rewrite", () => {
		fs.writeFileSync(path.join(roundsDir, "index.csv"), `${encodeVectorIndexLine([1], "legacy.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);

		expect(fs.existsSync(path.join(roundsDir, "index.csv"))).toBe(false);
		expect(index.readOwnLines()).toHaveLength(1);
	});

	it("holds the shard append lock across the rewrite window and releases it (PR #151 R1.F1)", () => {
		const own = path.join(roundsDir, "index.host-a");
		let lockSeenDuringWrite = false;
		const fsImpl = {
			...fs,
			writeFileSync: (file: Parameters<typeof fs.writeFileSync>[0], ...rest: unknown[]) => {
				// A wholesale rewrite must hold the append lock while it writes:
				// a concurrent runtime append takes the same lock, so it waits
				// instead of landing between the write and the rename.
				if (fs.existsSync(`${own}.lock`)) lockSeenDuringWrite = true;
				return (fs.writeFileSync as (...a: unknown[]) => void)(file, ...rest);
			},
		};
		const index = resolveScriptIndexContext(config, roundsDir, undefined, {
			hostname: "host-a",
			fsImpl: fsImpl as typeof fs,
		});

		index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);

		expect(lockSeenDuringWrite).toBe(true);
		expect(fs.existsSync(`${own}.lock`)).toBe(false);
	});

	it("fails fast without writing when the append lock is exhausted (PR #151 R1.F1)", () => {
		const eexist = Object.assign(new Error("EEXIST"), { code: "EEXIST" });
		const fsImpl = {
			...fs,
			openSync: (file: Parameters<typeof fs.openSync>[0], ...rest: unknown[]) => {
				if (String(file).endsWith(".lock")) throw eexist;
				return (fs.openSync as (...a: unknown[]) => number)(file, ...rest);
			},
		};
		const index = resolveScriptIndexContext(config, roundsDir, undefined, {
			hostname: "host-a",
			fsImpl: fsImpl as typeof fs,
			lockDeps: { lockRetries: 2, wait: () => {} },
		});
		const before = fs.existsSync(path.join(roundsDir, "index.host-a"));

		expect(() => index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`])).toThrow(/NOT written/);
		expect(fs.existsSync(path.join(roundsDir, "index.host-a"))).toBe(before);
	});

	it("a concurrent append that lands in the rewrite window survives (PR #151 R1.F1, SC-1)", () => {
		const own = path.join(roundsDir, "index.host-a");
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		// The caller resolves its lines first; a live session appends through
		// the runtime protocol (same `<shard>.lock`), so its row serializes
		// with the rewrite instead of racing it — it is present afterwards.
		index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);
		appendLineWithLock(own, roundsDir, `${encodeVectorIndexLine([8], "live2.json:prompt")}\n`);

		const ownLines = fs.readFileSync(own, "utf-8").trim().split("\n");
		expect(ownLines).toContain(encodeVectorIndexLine([3], "own.json:prompt"));
		expect(ownLines).toContain(encodeVectorIndexLine([8], "live2.json:prompt"));
	});

	// claim: pr151-review-r2.feature-SC-1/SC-1/rewriteOwnRows_preserves_live_append
	// claim text: a row appended after a migration's detection read is present in the own shard after the rewrite publishes
	// scenario: Scenario: SC-1 live append survives a migration rewrite (R2.F1) — Then the appended row is present in the shard after the publish
	it("rewriteOwnRows preserves a live append after the detection read", () => {
		const own = path.join(roundsDir, "index.host-a");
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });
		index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);

		// The script's detection read happens before this append.
		const detected = index.readRewriteRows();
		const live = `${encodeVectorIndexLine([8], "live.json:prompt")}\n`;
		appendLineWithLock(own, roundsDir, live);

		index.rewriteOwnRows((freshRows) =>
			freshRows.map((line) => (line.includes("own.json:prompt") ? line.replace("[3]", "[4]") : line)),
		);

		const published = fs.readFileSync(own, "utf-8").trim().split("\n");
		expect(published).toContain(detected[0].replace("[3]", "[4]"));
		expect(published).toContain(live.trim());
	});

	// claim: pr151-review-r2.feature-SC-1-EXT-1/SC-1-EXT-1/rewriteOwnRows_fails_fast_when_lock_unavailable
	// claim text: when the own-shard lock cannot be acquired within the retry budget, the rewrite throws and does not publish
	// scenario: Scenario: SC-1/EXT-1 migration rewrite fails fast without publishing when the lock is unavailable (R2.F1) — Then the migration throws and the shard is not written
	it("rewriteOwnRows fails fast without publishing when the lock is unavailable", () => {
		const eexist = Object.assign(new Error("EEXIST"), { code: "EEXIST" });
		const fsImpl = {
			...fs,
			openSync: (file: Parameters<typeof fs.openSync>[0], ...rest: unknown[]) => {
				if (String(file).endsWith(".lock")) throw eexist;
				return (fs.openSync as (...a: unknown[]) => number)(file, ...rest);
			},
		};
		const index = resolveScriptIndexContext(config, roundsDir, undefined, {
			hostname: "host-a",
			fsImpl: fsImpl as typeof fs,
			lockDeps: { lockRetries: 2, wait: () => {} },
		});
		const before = fs.existsSync(path.join(roundsDir, "index.host-a"));

		expect(() => index.rewriteOwnRows(() => [`${encodeVectorIndexLine([3], "own.json:prompt")}\n`])).toThrow(
			/NOT written/,
		);
		expect(fs.existsSync(path.join(roundsDir, "index.host-a"))).toBe(before);
	});

	it("reports foreign shards that still reference a round file", () => {
		fs.writeFileSync(path.join(roundsDir, "index.host-b"), `${encodeVectorIndexLine([2], "old.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		expect(index.foreignShardsReferencing("old.json")).toHaveLength(1);
		expect(index.foreignShardsReferencing("other.json")).toHaveLength(0);
	});

	it("counts the rewrite rows of the own shard plus the claimable legacy file", () => {
		fs.writeFileSync(path.join(roundsDir, "index.csv"), `${encodeVectorIndexLine([1], "legacy.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		// The legacy file is this host's rows until the write claims it.
		expect(index.readRewriteRows()).toHaveLength(1);
		expect(index.readOwnLines()).toHaveLength(0);
		// A claimable legacy file is not a foreign shard.
		expect(index.foreignShards()).toEqual([]);
	});

	it("reports the foreign shards and the round files they reference", () => {
		fs.writeFileSync(path.join(roundsDir, "index.host-b"), `${encodeVectorIndexLine([2], "b.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		expect(index.foreignShards().map((shard) => path.basename(shard.shardPath))).toEqual(["index.host-b"]);
		expect(index.foreignShards()[0].rows).toBe(1);
		expect([...index.foreignReferencedFiles()]).toEqual(["b.json"]);
	});

	it("records the outstanding foreign shards of a rewrite (issue #149, OI-8)", () => {
		fs.writeFileSync(path.join(roundsDir, "index.host-b"), `${encodeVectorIndexLine([2], "b.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, {
			hostname: "host-a",
			script: "migrate-model-column",
		});
		index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);

		reportForeignShards(index, { log: () => {} });

		const entries = readForeignBacklog(roundsDir);
		expect(entries).toHaveLength(1);
		expect(entries[0].shard).toBe("index.host-b");
		expect(entries[0].rows).toBe(1);
		expect(entries[0].roundFiles).toEqual(["b.json"]);
		expect(entries[0].script).toBe("migrate-model-column");
	});

	it("clears the own backlog entry when the owning host runs the same rewrite", () => {
		fs.writeFileSync(path.join(roundsDir, "index.host-b"), `${encodeVectorIndexLine([2], "b.json:prompt")}\n`);
		const hostA = resolveScriptIndexContext(config, roundsDir, undefined, {
			hostname: "host-a",
			script: "migrate-model-column",
		});
		hostA.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);
		reportForeignShards(hostA, { log: () => {} });
		expect(readForeignBacklog(roundsDir).map((entry) => entry.shard)).toEqual(["index.host-b"]);

		// The host that owns "index.host-b" runs the same rewrite.
		const hostB = resolveScriptIndexContext(config, roundsDir, undefined, {
			hostname: "host-b",
			script: "migrate-model-column",
		});
		hostB.writeOwnLines([`${encodeVectorIndexLine([2], "b.json:prompt")}\n`]);
		reportForeignShards(hostB, { log: () => {} });

		const entries = readForeignBacklog(roundsDir);
		expect(entries.map((entry) => entry.shard)).not.toContain("index.host-b");
	});

	it("records no backlog without a script name", () => {
		fs.writeFileSync(path.join(roundsDir, "index.host-b"), `${encodeVectorIndexLine([2], "b.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });
		index.writeOwnLines([`${encodeVectorIndexLine([3], "own.json:prompt")}\n`]);

		reportForeignShards(index, { log: () => {} });

		expect(readForeignBacklog(roundsDir)).toEqual([]);
	});
});
