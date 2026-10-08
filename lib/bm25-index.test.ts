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
		const fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "readFileSync" | "readdirSync" | "writeFileSync"> = {
			existsSync: fs.existsSync,
			mkdirSync: fs.mkdirSync,
			readFileSync: fs.readFileSync,
			readdirSync: fs.readdirSync,
			writeFileSync: (p, data) => {
				writtenFiles.push(String(p));
				fs.writeFileSync(p, data);
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

		expect(writtenFiles).toEqual([indexPath]);
		expect(loadBm25Index(indexPath).documents["newer.json"]).toBeDefined();
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
