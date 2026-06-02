import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { computeContentHash } from "./hash.ts";
import {
	appendVectorIndexEntry,
	encodeVectorIndexLine,
	filterIndexLinesExcludingFilenames,
	findStaleContentMatches,
	indexEntryFilename,
	indexRoundFileFromPath,
	loadIndexedRoundFiles,
	loadVectorIndex,
	migrateIndexEntries,
	migrateIndexEntryLine,
	readIndexByFilename,
	readIndexLines,
	replaceIndexLineFilename,
	writeIndexLines,
} from "./index-io.ts";

let tempDir = "";

beforeEach(() => {
	tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-index-io-"));
});

afterEach(() => {
	fs.rmSync(tempDir, { recursive: true, force: true });
});

const indexPath = () => path.join(tempDir, "index.csv");

describe("index I/O helpers", () => {
	it("encodes, appends, and loads vector index entries", () => {
		expect(readIndexLines(indexPath())).toEqual([]);
		appendVectorIndexEntry(indexPath(), [0.5, 1], "round.json:prompt");

		expect(readIndexLines(indexPath())).toEqual([encodeVectorIndexLine([0.5, 1], "round.json:prompt")]);
		expect(loadVectorIndex(indexPath())).toEqual([{ vector: [0.5, 1], filePath: "round.json:prompt" }]);
	});

	it("writes index lines and extracts round filenames", () => {
		writeIndexLines(indexPath(), [
			encodeVectorIndexLine([1], "dir/one.json:prompt"),
			encodeVectorIndexLine([2], "two.json:response"),
			encodeVectorIndexLine([3], "three.json:round"),
		]);

		expect(indexRoundFileFromPath("dir/one.json:prompt")).toBe("dir/one.json");
		expect(loadIndexedRoundFiles(indexPath())).toEqual(new Set(["one.json", "two.json", "three.json"]));
	});

	it("migrates matching index entry prefixes and leaves unrelated rows unchanged", () => {
		const oldLine = encodeVectorIndexLine([1], "old.json:prompt");
		const unrelated = encodeVectorIndexLine([2], "other.json:prompt");

		expect(migrateIndexEntryLine(oldLine, "old.json", "new.json")).toBe(
			encodeVectorIndexLine([1], "new.json:prompt"),
		);
		expect(migrateIndexEntryLine(unrelated, "old.json", "new.json")).toBe(unrelated);

		writeIndexLines(indexPath(), [oldLine, unrelated]);
		migrateIndexEntries(indexPath(), "old.json", "new.json");

		expect(readIndexLines(indexPath())).toEqual([encodeVectorIndexLine([1], "new.json:prompt"), unrelated]);
	});

	it("groups rows by filename and filters renamed filenames", () => {
		const keep = encodeVectorIndexLine([1], "keep.json:prompt");
		const oldPrompt = encodeVectorIndexLine([2], "old.json:prompt");
		const oldResponse = encodeVectorIndexLine([3], "old.json:response");
		writeIndexLines(indexPath(), [keep, oldPrompt, oldResponse, "malformed"]);

		expect(indexEntryFilename(keep)).toBe("keep.json");
		expect(indexEntryFilename("malformed")).toBeNull();
		expect(readIndexByFilename(indexPath())).toEqual(
			new Map([
				["keep.json", [keep]],
				["old.json", [oldPrompt, oldResponse]],
			]),
		);
		expect(filterIndexLinesExcludingFilenames(readIndexLines(indexPath()), new Set(["old.json"]))).toEqual([
			keep,
			"malformed",
		]);
		expect(replaceIndexLineFilename(oldPrompt, "new.json")).toBe(encodeVectorIndexLine([2], "new.json:prompt"));
	});

	it("finds stale round files by content hash and skips corrupt/current files", () => {
		const roundFile = `${computeContentHash("Prompt", "Response")}.json`;
		fs.writeFileSync(
			path.join(tempDir, roundFile),
			JSON.stringify({ userPrompt: "Prompt", responseSequence: "Response" }),
		);
		fs.writeFileSync(
			path.join(tempDir, "stale.json"),
			JSON.stringify({ userPrompt: "Prompt", responseSequence: "Response" }),
		);
		fs.writeFileSync(
			path.join(tempDir, "other.json"),
			JSON.stringify({ userPrompt: "Prompt", responseSequence: "Different" }),
		);
		fs.writeFileSync(path.join(tempDir, "bad.json"), "not json");
		fs.writeFileSync(
			path.join(tempDir, "index.json"),
			JSON.stringify({ userPrompt: "Prompt", responseSequence: "Response" }),
		);

		expect(findStaleContentMatches(tempDir, roundFile)).toEqual(["stale.json"]);
	});
});
