import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { appendIndexRowToShards, embeddingClientDeps, loadMergedIndex } from "../src/semblr.ts";
import { loadShardVectorIndex } from "./index-io.ts";
import { buildCurrentModelRowPredicate } from "./session-backfill.ts";

describe("embeddingClientDeps", () => {
	it("overrides only the embedding model for a model-specific query", () => {
		const modelRegistry = { marker: "registry" };
		const deps = embeddingClientDeps({ modelRegistry } as never, "historical-model");

		expect(deps.modelRegistry).toBe(modelRegistry);
		expect(deps.config.embeddingModel).toBe("historical-model");
		expect(deps.config.embeddingProvider).toBeTruthy();
		expect(deps.config.embedMaxRetries).toBeGreaterThan(0);
	});
});

describe("live append write surface (#149)", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-live-append-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	// The extension's appendToIndex routes through this seam: a live embed of a
	// round must land as a row in the host's own shard (index.<hostname>) and
	// nowhere else, with the legacy index.csv adopted on first write. Locked
	// with the shared index-writer protocol (decision 005) using the real fs.
	it("lands a live row in the host's own shard only (real fs, with lock)", () => {
		appendIndexRowToShards(
			tempDir,
			"rounds/abc.json:prompt",
			[0.1, 0.2, 0.3],
			{ hostname: os.hostname() },
			"test-model",
		);
		appendIndexRowToShards(
			tempDir,
			"rounds/abc.json:response",
			[0.4, 0.5, 0.6],
			{ hostname: os.hostname() },
			"test-model",
		);

		const files = fs.readdirSync(tempDir);
		expect(files).toEqual([`index.${os.hostname()}`]);
		const lines = fs
			.readFileSync(path.join(tempDir, `index.${os.hostname()}`), "utf-8")
			.trim()
			.split("\n");
		expect(lines).toHaveLength(2);
		expect(lines[0]).toContain("rounds/abc.json:prompt");
		expect(lines[1]).toContain("rounds/abc.json:response");
	});
});

describe("merged index read surface (#149)", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-merged-read-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	// The extension's loadIndex routes through this seam: startup status, TUI
	// counts, and search must see rows from every host's shard (merged view),
	// and a missing rounds dir / empty shard set yields an empty index without
	// error (issue #149 extension 1.b).
	it("returns rows from all shards over the merged view", () => {
		appendIndexRowToShards(tempDir, "rounds/a.json:prompt", [0.1], {}, "test-model");
		// Simulate a foreign host's shard written outside this host's appends.
		fs.writeFileSync(
			path.join(tempDir, "index.host-b"),
			`${Buffer.from(JSON.stringify([0.2])).toString("base64url")},rounds/b.json:prompt\n`,
		);

		const index = loadMergedIndex(tempDir);
		expect(index.map((entry) => entry.filePath).sort()).toEqual(["rounds/a.json:prompt", "rounds/b.json:prompt"]);
	});

	it("returns an empty index without error for a missing rounds dir", () => {
		expect(loadMergedIndex(path.join(tempDir, "missing"))).toEqual([]);
	});

	it("returns an empty index without error for an empty shard set", () => {
		expect(loadMergedIndex(tempDir)).toEqual([]);
	});
});

describe("startup pending predicate spans shards (D-010, #149)", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-pending-shards-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	// The startup embedding predicate is built over the merged shard view (the
	// same composition semblr.ts wires: loadShardVectorIndex →
	// buildCurrentModelRowPredicate). A round a foreign host already embedded
	// with the current model — its rows live only in the foreign shard — must
	// count as covered, so it is not re-embedded locally on every startup.
	it("counts a foreign-shard row as covered by the current-model predicate", () => {
		fs.writeFileSync(
			path.join(tempDir, "index.host-b"),
			`${Buffer.from(JSON.stringify([0.2])).toString("base64url")},rounds/foreign.json:prompt,test-model\n`,
		);

		const indexEntries = loadShardVectorIndex(tempDir);
		const hasCurrentModelRows = buildCurrentModelRowPredicate(indexEntries, "test-model");
		expect(hasCurrentModelRows("foreign.json")).toBe(true);
	});
});
