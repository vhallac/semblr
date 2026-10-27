import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	clearForeignBacklog,
	describeForeignShards,
	foreignBacklogPath,
	readForeignBacklog,
	recordForeignBacklog,
} from "./foreign-backlog.ts";
import { encodeVectorIndexLine } from "./index-io.ts";
import { discoverIndexShards } from "./index-shards.ts";

describe("foreign-shard backlog", () => {
	let root = "";
	let roundsDir = "";

	beforeEach(() => {
		root = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-backlog-"));
		roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
	});

	afterEach(() => {
		fs.rmSync(root, { recursive: true, force: true });
	});

	it("reads an empty backlog when no state file exists", () => {
		expect(readForeignBacklog(roundsDir)).toEqual([]);
	});

	it("records the outstanding foreign shards of a rewrite run", () => {
		recordForeignBacklog(roundsDir, "migrate-model-column", [
			{ shard: "index.beta", rows: 3, roundFiles: ["b.json"] },
		]);

		const entries = readForeignBacklog(roundsDir);
		expect(entries).toHaveLength(1);
		expect(entries[0].script).toBe("migrate-model-column");
		expect(entries[0].shard).toBe("index.beta");
		expect(entries[0].rows).toBe(3);
		expect(entries[0].roundFiles).toEqual(["b.json"]);
		expect(entries[0].recordedAt).not.toBe("");
	});

	it("replaces the entries of the same script instead of piling them up", () => {
		recordForeignBacklog(roundsDir, "migrate-model-column", [
			{ shard: "index.beta", rows: 3, roundFiles: ["b.json"] },
		]);
		recordForeignBacklog(roundsDir, "migrate-model-column", [{ shard: "index.gamma", rows: 1, roundFiles: [] }]);

		const entries = readForeignBacklog(roundsDir);
		expect(entries.map((entry) => entry.shard)).toEqual(["index.gamma"]);
	});

	it("keeps the entries of other scripts", () => {
		recordForeignBacklog(roundsDir, "migrate-model-column", [{ shard: "index.beta", rows: 3, roundFiles: [] }]);
		recordForeignBacklog(roundsDir, "migrate-content-hash", [{ shard: "index.beta", rows: 2, roundFiles: [] }]);

		expect(readForeignBacklog(roundsDir).map((entry) => entry.script)).toEqual([
			"migrate-model-column",
			"migrate-content-hash",
		]);
	});

	it("clears the entry of a shard when its owner runs the same script", () => {
		recordForeignBacklog(roundsDir, "migrate-model-column", [
			{ shard: "index.beta", rows: 3, roundFiles: [] },
			{ shard: "index.alpha", rows: 2, roundFiles: [] },
		]);

		const removed = clearForeignBacklog(roundsDir, "migrate-model-column", ["index.beta"]);

		expect(removed).toBe(true);
		expect(readForeignBacklog(roundsDir).map((entry) => entry.shard)).toEqual(["index.alpha"]);
	});

	it("keeps the entry when another script is cleared", () => {
		recordForeignBacklog(roundsDir, "migrate-model-column", [{ shard: "index.beta", rows: 3, roundFiles: [] }]);

		expect(clearForeignBacklog(roundsDir, "migrate-content-hash", ["index.beta"])).toBe(false);
		expect(readForeignBacklog(roundsDir)).toHaveLength(1);
	});

	it("treats a corrupt state file as an empty backlog", () => {
		fs.writeFileSync(foreignBacklogPath(roundsDir), "not json");
		expect(readForeignBacklog(roundsDir)).toEqual([]);
	});

	it("is not discovered as an index shard (nor are its backups and lockfile)", () => {
		fs.writeFileSync(path.join(roundsDir, "index.alpha"), `${encodeVectorIndexLine([1], "a.json:prompt")}\n`);
		recordForeignBacklog(roundsDir, "migrate-model-column", [{ shard: "index.beta", rows: 3, roundFiles: [] }]);
		fs.writeFileSync(path.join(roundsDir, "index.backlog.json.bak.1700000000"), "{}");
		fs.writeFileSync(path.join(roundsDir, "index.backlog.json.lock"), "");

		const shards = discoverIndexShards(roundsDir).map((shard) => path.basename(shard));
		expect(shards).toEqual(["index.alpha"]);
	});

	it("describes the foreign shards with their rows and referenced round files", () => {
		const own = path.join(roundsDir, "index.alpha");
		fs.writeFileSync(own, `${encodeVectorIndexLine([1], "a.json:prompt")}\n`);
		fs.writeFileSync(
			path.join(roundsDir, "index.beta"),
			`${encodeVectorIndexLine([2], "b.json:prompt")}\n${encodeVectorIndexLine([3], "c.json:prompt")}\n`,
		);

		expect(describeForeignShards(roundsDir, [own])).toEqual([
			{ shard: "index.beta", rows: 2, roundFiles: ["b.json", "c.json"] },
		]);
	});
});
