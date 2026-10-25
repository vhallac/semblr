import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { encodeVectorIndexLine } from "./index-io.ts";
import { resolveScriptIndexContext } from "./script-config.ts";
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

	it("reports foreign shards that still reference a round file", () => {
		fs.writeFileSync(path.join(roundsDir, "index.host-b"), `${encodeVectorIndexLine([2], "old.json:prompt")}\n`);
		const index = resolveScriptIndexContext(config, roundsDir, undefined, { hostname: "host-a" });

		expect(index.foreignShardsReferencing("old.json")).toHaveLength(1);
		expect(index.foreignShardsReferencing("other.json")).toHaveLength(0);
	});
});
