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
	scoreBm25Query,
	upsertBm25Round,
	writeBm25Index,
} from "./bm25-index.ts";

function tmpDir() {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-bm25-"));
}

describe("bm25 index", () => {
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
	])("backfills a %s sidecar from existing rounds", (_case, sidecarContents) => {
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

		expect(scoreBm25Query(index, "exact_identifier").get("existing.json")).toBeGreaterThan(0);
		expect(loadBm25Index(indexPath).documentCount).toBe(1);
	});
});
