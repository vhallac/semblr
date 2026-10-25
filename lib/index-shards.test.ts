import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { encodeVectorIndexLine } from "./index-io.ts";
import {
	adoptOwnShard,
	describeShardSet,
	discoverIndexShards,
	isIndexShardFilename,
	LEGACY_INDEX_FILENAME,
	ownShardPath,
	shardNames,
} from "./index-shards.ts";

describe("isIndexShardFilename", () => {
	it("accepts shard filenames, including the legacy index.csv", () => {
		expect(isIndexShardFilename("index.csv")).toBe(true);
		expect(isIndexShardFilename("index.cowork")).toBe(true);
		expect(isIndexShardFilename("index.my-host.name")).toBe(true);
	});

	it("excludes sibling artifacts by explicit rule (issue #149 ext 3.a)", () => {
		expect(isIndexShardFilename("index-tools.fulltext.csv")).toBe(false);
		expect(isIndexShardFilename("index.bm25.json")).toBe(false);
		expect(isIndexShardFilename("index.csv.bak.20260101")).toBe(false);
	});

	it("excludes lockfiles and atomic-write temps", () => {
		expect(isIndexShardFilename("index.cowork.lock")).toBe(false);
		expect(isIndexShardFilename("index.csv.lock")).toBe(false);
		expect(isIndexShardFilename("index.cowork.tmp.1234")).toBe(false);
	});

	it("rejects non-index files", () => {
		expect(isIndexShardFilename("abc.json")).toBe(false);
		expect(isIndexShardFilename("index")).toBe(false);
	});
});

describe("describeShardSet", () => {
	it("lists the shard file names", () => {
		expect(shardNames(["/rounds/index.host-a", "/rounds/index.host-b"])).toEqual(["index.host-a", "index.host-b"]);
		expect(describeShardSet(["/rounds/index.host-a", "/rounds/index.host-b"])).toBe("index.host-a, index.host-b");
	});

	it("labels an empty shard set", () => {
		expect(shardNames([])).toEqual([]);
		expect(describeShardSet([])).toBe("none (empty shard set)");
	});
});

describe("ownShardPath", () => {
	it("joins index.<hostname> inside the rounds dir", () => {
		expect(ownShardPath("/rounds", "host-a")).toBe(path.join("/rounds", "index.host-a"));
	});

	it("defaults to os.hostname()", () => {
		expect(ownShardPath("/rounds")).toBe(path.join("/rounds", `index.${os.hostname()}`));
	});
});

describe("discoverIndexShards", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-shards-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	it("returns an empty set without error for a missing rounds dir (ext 1.b)", () => {
		expect(discoverIndexShards(path.join(tempDir, "missing"))).toEqual([]);
	});

	it("returns an empty set without error for an empty rounds dir (ext 1.b)", () => {
		expect(discoverIndexShards(tempDir)).toEqual([]);
	});

	it("discovers shards and excludes sibling artifacts", () => {
		for (const name of [
			"index.cowork",
			"index.hostb",
			"index-tools.fulltext.csv",
			"index.bm25.json",
			"index.csv.bak.x",
		]) {
			fs.writeFileSync(path.join(tempDir, name), "");
		}
		expect(discoverIndexShards(tempDir)).toEqual([
			path.join(tempDir, "index.cowork"),
			path.join(tempDir, "index.hostb"),
		]);
	});

	it("includes the legacy index.csv until adopted", () => {
		fs.writeFileSync(path.join(tempDir, LEGACY_INDEX_FILENAME), "");
		expect(discoverIndexShards(tempDir)).toEqual([path.join(tempDir, LEGACY_INDEX_FILENAME)]);
	});
});

describe("adoptOwnShard", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-adopt-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	const writeIndex = (name: string, rows: string[]): string => {
		const p = path.join(tempDir, name);
		fs.writeFileSync(p, rows.map((r) => `${encodeVectorIndexLine([0.1, 0.2], r)}\n`).join(""));
		return p;
	};

	it("renames a legacy index.csv to the host's own shard and takes ownership", () => {
		writeIndex(LEGACY_INDEX_FILENAME, ["rounds/a.json"]);
		const result = adoptOwnShard(tempDir, { hostname: "host-a" });
		expect(result).toEqual({
			shardPath: path.join(tempDir, "index.host-a"),
			adoptedLegacy: true,
		});
		expect(fs.existsSync(path.join(tempDir, LEGACY_INDEX_FILENAME))).toBe(false);
		expect(fs.readFileSync(path.join(tempDir, "index.host-a"), "utf-8")).toContain("rounds/a.json");
	});

	it("keeps the host's own shard when no legacy file exists", () => {
		const result = adoptOwnShard(tempDir, { hostname: "host-a" });
		expect(result).toEqual({ shardPath: path.join(tempDir, "index.host-a"), adoptedLegacy: false });
		expect(fs.readdirSync(tempDir)).toEqual([]);
	});

	it("does not rename when the host already owns a shard (legacy file stays readable)", () => {
		writeIndex(LEGACY_INDEX_FILENAME, ["rounds/legacy.json"]);
		writeIndex("index.host-a", ["rounds/own.json"]);
		const result = adoptOwnShard(tempDir, { hostname: "host-a" });
		expect(result.adoptedLegacy).toBe(false);
		expect(fs.existsSync(path.join(tempDir, LEGACY_INDEX_FILENAME))).toBe(true);
		expect(fs.readFileSync(path.join(tempDir, "index.host-a"), "utf-8")).toContain("rounds/own.json");
	});

	it("is a no-op on a lost adoption race (legacy file already claimed)", () => {
		// Simulate another host claiming the legacy file between the check and
		// the rename: existsSync passes once, then renameSync fails.
		let existsCalls = 0;
		const fsImpl = {
			existsSync: (p: string) => {
				existsCalls++;
				return existsCalls === 1;
			},
			readdirSync: () => [],
			renameSync: () => {
				throw new Error("ENOENT");
			},
		};
		const result = adoptOwnShard(tempDir, { hostname: "host-a", fsImpl });
		expect(result).toEqual({ shardPath: path.join(tempDir, "index.host-a"), adoptedLegacy: false });
	});
});
