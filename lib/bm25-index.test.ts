import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import {
	bm25IndexPathForRoundsDir,
	buildBm25Index,
	deleteBm25Round,
	loadBm25Index,
	loadOrRebuildBm25Index,
	normalizeBm25Scores,
	ownBm25IndexPath,
	scoreBm25Query,
	tokenizeBm25,
	upsertBm25Round,
	writeBm25Index,
} from "./bm25-index.ts";

function tmpDir() {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-bm25-"));
}

describe("bm25 index", () => {
	it("stems plural words ending in ies to y", () => {
		expect(tokenizeBm25("queries")).toEqual(["query"]);
	});

	it("normalizes scores with a sigmoid scaled by the median top score", () => {
		const normalized = normalizeBm25Scores(
			new Map([
				["low.json", 1],
				["middle.json", 2],
				["high.json", 4],
			]),
		);

		expect(normalized.get("low.json")).toBeCloseTo(1 / (1 + Math.exp(-0.5)));
		expect(normalized.get("middle.json")).toBeCloseTo(1 / (1 + Math.exp(-1)));
		expect(normalized.get("high.json")).toBeCloseTo(1 / (1 + Math.exp(-2)));
	});

	it("scores exact identifier matches higher than unrelated text", () => {
		const index = buildBm25Index([
			{
				fileName: "exact.json",
				text: "Debug get_round_details failing for rounds/abc.json in the native tool registry.",
			},
			{
				fileName: "conceptual.json",
				text: "Discussed semantic retrieval and long running context memory.",
			},
		]);

		const scores = scoreBm25Query(index, "get_round_details rounds/abc.json");

		expect(scores.get("exact.json")).toBeGreaterThan(0);
		expect(scores.get("exact.json")).toBeGreaterThan(scores.get("conceptual.json") ?? 0);
	});

	it("upserts round text and persists a loadable sidecar index", () => {
		const roundsDir = tmpDir();
		const indexPath = bm25IndexPathForRoundsDir(roundsDir);
		const index = buildBm25Index([{ fileName: "old.json", text: "old query token" }]);

		upsertBm25Round(index, "old.json", "new get_tool_details token");
		upsertBm25Round(index, "new.json", "new config path token");
		writeBm25Index(indexPath, index);

		const loaded = loadBm25Index(indexPath);
		const scores = scoreBm25Query(loaded, "get_tool_details");

		expect(scores.get("old.json")).toBeGreaterThan(0);
		expect(scoreBm25Query(loaded, "old query").get("old.json") ?? 0).toBe(0);
		expect(loaded.documentCount).toBe(2);
		expect(loaded.averageDocumentLength).toBeGreaterThan(0);
	});

	it("deletes stale round documents", () => {
		const index = buildBm25Index([
			{ fileName: "old.json", text: "stale identifier" },
			{ fileName: "current.json", text: "current identifier" },
		]);

		deleteBm25Round(index, "old.json");

		expect(scoreBm25Query(index, "stale").get("old.json") ?? 0).toBe(0);
		expect(index.documentCount).toBe(1);
	});

	it.each([
		["missing", undefined],
		["corrupt", "{not json"],
		[
			"structurally malformed",
			JSON.stringify({
				version: 1,
				documentCount: 1,
				averageDocumentLength: 1,
				documents: { "broken.json": { length: "one", termFrequencies: null } },
			}),
		],
	])("rebuilds an in-memory index from existing rounds without touching a %s sidecar", (_case, sidecarContents) => {
		const roundsDir = tmpDir();
		const indexPath = bm25IndexPathForRoundsDir(roundsDir);
		fs.writeFileSync(
			path.join(roundsDir, "existing.json"),
			JSON.stringify({
				userPrompt: "Find exact_identifier",
				responseSequence: "Recovered from a saved round.",
			}),
		);
		if (sidecarContents !== undefined) fs.writeFileSync(indexPath, sidecarContents);

		const index = loadOrRebuildBm25Index(indexPath, roundsDir);

		// The load path is non-writing (PR !138 F1): persistence is the
		// caller's explicit flush, so the sidecar is untouched here.
		expect(scoreBm25Query(index, "exact_identifier").get("existing.json")).toBeGreaterThan(0);
		expect(fs.existsSync(indexPath)).toBe(Boolean(sidecarContents));
	});

	it("rebuilds a valid sidecar that omits saved rounds", () => {
		const roundsDir = tmpDir();
		const indexPath = bm25IndexPathForRoundsDir(roundsDir);
		fs.writeFileSync(
			path.join(roundsDir, "existing.json"),
			JSON.stringify({ userPrompt: "historical exact_identifier", responseSequence: "saved response" }),
		);
		writeBm25Index(indexPath, buildBm25Index([{ fileName: "new.json", text: "new round" }]));

		const index = loadOrRebuildBm25Index(indexPath, roundsDir);

		expect(scoreBm25Query(index, "exact_identifier").get("existing.json")).toBeGreaterThan(0);
		expect(index.documentCount).toBe(1);
	});

	it("persists the index exactly once on the stale-sidecar recovery path (PR !138 F1)", () => {
		const roundsDir = tmpDir();
		const indexPath = bm25IndexPathForRoundsDir(roundsDir);
		const writtenFiles: string[] = [];
		const renames: Array<[string, string]> = [];
		const fsImpl: Pick<
			typeof fs,
			"existsSync" | "mkdirSync" | "readFileSync" | "readdirSync" | "writeFileSync" | "renameSync"
		> = {
			existsSync: fs.existsSync,
			mkdirSync: fs.mkdirSync,
			readFileSync: fs.readFileSync,
			readdirSync: fs.readdirSync,
			writeFileSync: (p, data) => {
				writtenFiles.push(String(p));
				fs.writeFileSync(p, data);
			},
			renameSync: (from, to) => {
				renames.push([String(from), String(to)]);
				fs.renameSync(from, to);
			},
		};
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(
			path.join(roundsDir, "old.json"),
			JSON.stringify({ userPrompt: "old token_a", responseSequence: "old" }),
		);
		// Stale sidecar: exists but does not cover the round files.
		fs.writeFileSync(indexPath, JSON.stringify({ version: 1, documents: {} }));

		// Production wiring (src/semblr.ts): load-or-rebuild, in-memory upserts
		// for each recovered round, then one flush for the batch.
		const index = loadOrRebuildBm25Index(indexPath, roundsDir, fsImpl);
		upsertBm25Round(index, "new.json", "new token_b round");
		upsertBm25Round(index, "newer.json", "newer token_c round");
		writeBm25Index(indexPath, index, fsImpl);

		// Exactly-once flush (issue #137), now atomic (decision 015): one staged
		// tmp write, one rename publishing it onto the artifact path.
		const tmpPath = `${indexPath}.tmp.${process.pid}`;
		expect(writtenFiles).toEqual([tmpPath]);
		expect(renames).toEqual([[tmpPath, indexPath]]);
		expect(loadBm25Index(indexPath).documents["newer.json"]).toBeDefined();
	});
});

describe("per-host BM25 artifacts (issue #152, decision 024)", () => {
	it("ownBm25IndexPath places the artifact beside the shards", () => {
		expect(ownBm25IndexPath("/rounds", "laptop-x")).toBe(path.join("/rounds", "index.bm25.laptop-x.json"));
	});

	it("adopts the legacy artifact by rename on the first write to a host's own path", () => {
		const roundsDir = tmpDir();
		const legacyPath = path.join(roundsDir, "index.bm25.json");
		const ownPath = ownBm25IndexPath(roundsDir, "host-a");
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(legacyPath, JSON.stringify({ version: 1, documents: {} }));

		writeBm25Index(ownPath, buildBm25Index([{ fileName: "a.json", text: "fresh token" }]));

		// The legacy file is gone — renamed onto the own path and immediately
		// replaced by the flushed superset of its content.
		expect(fs.existsSync(legacyPath)).toBe(false);
		expect(loadBm25Index(ownPath).documents["a.json"]).toBeDefined();
	});

	it("leaves the legacy artifact untouched when the own artifact already exists (clobber guard)", () => {
		const roundsDir = tmpDir();
		const legacyPath = path.join(roundsDir, "index.bm25.json");
		const ownPath = ownBm25IndexPath(roundsDir, "host-a");
		fs.mkdirSync(roundsDir, { recursive: true });
		writeBm25Index(ownPath, buildBm25Index([{ fileName: "a.json", text: "host artifact" }]));
		const legacyBytes = JSON.stringify({ version: 1, documents: {} });
		fs.writeFileSync(legacyPath, legacyBytes);

		writeBm25Index(ownPath, buildBm25Index([{ fileName: "b.json", text: "another flush" }]));

		// Renaming would clobber the host's own artifact (POSIX rename
		// overwrites) — the orphan legacy file stays on disk untouched.
		expect(fs.readFileSync(legacyPath, "utf-8")).toBe(legacyBytes);
	});

	it("swallows a lost adoption race and still writes its own artifact", () => {
		const roundsDir = tmpDir();
		const legacyPath = path.join(roundsDir, "index.bm25.json");
		const ownPath = ownBm25IndexPath(roundsDir, "host-a");
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(legacyPath, JSON.stringify({ version: 1, documents: {} }));
		const fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "renameSync"> = {
			existsSync: fs.existsSync,
			mkdirSync: fs.mkdirSync,
			writeFileSync: fs.writeFileSync,
			renameSync: (from, to) => {
				if (String(from) === legacyPath) throw new Error("claimed by another host");
				return fs.renameSync(from, to);
			},
		};

		writeBm25Index(ownPath, buildBm25Index([{ fileName: "a.json", text: "token" }]), fsImpl);

		// The lost legacy claim does not block the flush.
		expect(loadBm25Index(ownPath).documents["a.json"]).toBeDefined();
	});

	it("writes the artifact atomically: staged tmp write, rename publish, never in-place (decision 015)", () => {
		const roundsDir = tmpDir();
		const indexPath = ownBm25IndexPath(roundsDir, "host-a");
		const writes: string[] = [];
		const renames: Array<[string, string]> = [];
		const fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "renameSync"> = {
			existsSync: fs.existsSync,
			mkdirSync: fs.mkdirSync,
			writeFileSync: (p, data) => {
				writes.push(String(p));
				fs.writeFileSync(p, data);
			},
			renameSync: (from, to) => {
				renames.push([String(from), String(to)]);
				fs.renameSync(from, to);
			},
		};

		writeBm25Index(indexPath, buildBm25Index([{ fileName: "a.json", text: "token" }]), fsImpl);

		const tmpPath = `${indexPath}.tmp.${process.pid}`;
		expect(writes).toEqual([tmpPath]);
		expect(renames).toEqual([[tmpPath, indexPath]]);
		expect(fs.existsSync(tmpPath)).toBe(false);
	});

	it("writes the legacy path directly without attempting adoption", () => {
		const roundsDir = tmpDir();
		const legacyPath = bm25IndexPathForRoundsDir(roundsDir);
		const renames: Array<[string, string]> = [];
		const fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "renameSync"> = {
			existsSync: fs.existsSync,
			mkdirSync: fs.mkdirSync,
			writeFileSync: fs.writeFileSync,
			renameSync: (from, to) => {
				renames.push([String(from), String(to)]);
				return fs.renameSync(from, to);
			},
		};

		writeBm25Index(legacyPath, buildBm25Index([{ fileName: "a.json", text: "token" }]), fsImpl);

		// Callers still on the legacy path (units 005/006 pending) are unaffected.
		expect(renames.filter(([from]) => from !== `${legacyPath}.tmp.${process.pid}`)).toEqual([]);
		expect(loadBm25Index(legacyPath).documents["a.json"]).toBeDefined();
	});
});

describe("bm25 prototype-chain hygiene (PR !131)", () => {
	it("indexes the token 'constructor' as a numeric frequency, not the inherited function", () => {
		// `tf[token] ?? 0` resolved through the prototype chain: for the token
		// "constructor" it produced "function Object() { [native code] }1".
		const doc = buildBm25Index([{ fileName: "a.json", text: "constructor constructor field" }]).documents["a.json"];
		expect(doc.termFrequencies.constructor).toBe(2);
	});

	it("a persisted index containing the token 'constructor' still validates on load", () => {
		const dir = tmpDir();
		const file = path.join(dir, "index.bm25.json");
		const index = buildBm25Index([{ fileName: "a.json", text: "constructor constructor" }]);
		writeBm25Index(file, index);
		const loaded = loadBm25Index(file);
		expect(Object.keys(loaded.documents)).toEqual(["a.json"]);
		expect(loaded.documents["a.json"].termFrequencies.constructor).toBe(2);
	});
});
