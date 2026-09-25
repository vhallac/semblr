import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { computeContentHash } from "./hash.ts";
import {
	ASSUMED_STAMP_PREFIX,
	appendVectorIndexEntry,
	bareEmbeddingInputHash,
	encodeVectorIndexLine,
	filterIndexLinesExcludingFilenames,
	findStaleContentMatches,
	indexEntryFilename,
	indexRoundFileFromPath,
	loadIndexedRoundFiles,
	loadRoundFilesWithDifferentModel,
	loadVectorIndex,
	makeAssumedStamp,
	migrateIndexEntries,
	migrateIndexEntryLine,
	readIndexByFilename,
	readIndexLines,
	replaceIndexEntriesForRoundFile,
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

	it("detects only explicit non-current model rows as model mismatches", () => {
		writeIndexLines(indexPath(), [
			encodeVectorIndexLine([1], "legacy.json:prompt"),
			encodeVectorIndexLine([2], "same.json:prompt", "current-model"),
			encodeVectorIndexLine([3], "different.json:prompt", "old-model"),
			encodeVectorIndexLine([4], "nested/also-different.json:response", "older-model"),
		]);

		expect(loadRoundFilesWithDifferentModel(indexPath(), "current-model")).toEqual(
			new Set(["different.json", "also-different.json"]),
		);
	});

	it("replaces all rows for one round file while preserving unrelated and malformed rows", () => {
		const keep = encodeVectorIndexLine([1], "keep.json:prompt", "old-model");
		const oldPrompt = encodeVectorIndexLine([2], "old.json:prompt", "old-model");
		const oldResponse = encodeVectorIndexLine([3], "old.json:response", "old-model");
		writeIndexLines(indexPath(), [keep, oldPrompt, oldResponse, "malformed"]);

		replaceIndexEntriesForRoundFile(indexPath(), "old.json", [
			{ vector: [4], filePath: "old.json:prompt", model: "current-model" },
			{ vector: [5], filePath: "old.json:response", model: "current-model" },
		]);

		expect(readIndexLines(indexPath())).toEqual([
			keep,
			"malformed",
			encodeVectorIndexLine([4], "old.json:prompt", "current-model"),
			encodeVectorIndexLine([5], "old.json:response", "current-model"),
		]);
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

describe("4th-column embedding-input hash (issue #106 re-embed migration)", () => {
	it("round-trips 2/3/4-column lines through loadVectorIndex", () => {
		writeIndexLines(indexPath(), [
			encodeVectorIndexLine([1], "a.json:prompt"),
			encodeVectorIndexLine([2], "b.json:prompt", "model-x"),
			encodeVectorIndexLine([3], "c.json:prompt", "model-x", "abc123"),
		]);

		expect(loadVectorIndex(indexPath())).toEqual([
			{ vector: [1], filePath: "a.json:prompt" },
			{ vector: [2], filePath: "b.json:prompt", model: "model-x" },
			{ vector: [3], filePath: "c.json:prompt", model: "model-x", embeddingInputHash: "abc123" },
		]);
	});

	it("appends entries with a hash column", () => {
		appendVectorIndexEntry(indexPath(), [0.5], "r.json:prompt", "model-x", "hash-1");
		appendVectorIndexEntry(indexPath(), [0.6], "r.json:response", "model-x");

		expect(loadVectorIndex(indexPath())).toEqual([
			{ vector: [0.5], filePath: "r.json:prompt", model: "model-x", embeddingInputHash: "hash-1" },
			{ vector: [0.6], filePath: "r.json:response", model: "model-x" },
		]);
	});

	it("keeps filename helpers correct for 4-column lines", () => {
		const line = encodeVectorIndexLine([1], "dir/one.json:prompt", "model-x", "abc123");

		expect(indexEntryFilename(line)).toBe("dir/one.json");
		expect(indexRoundFileFromPath("dir/one.json:prompt")).toBe("dir/one.json");
		expect(replaceIndexLineFilename(line, "two.json")).toBe(
			encodeVectorIndexLine([1], "two.json:prompt", "model-x", "abc123"),
		);
		expect(migrateIndexEntryLine(line, "dir/one.json", "dir/moved.json")).toBe(
			encodeVectorIndexLine([1], "dir/moved.json:prompt", "model-x", "abc123"),
		);
		expect(migrateIndexEntryLine(line, "other.json", "dir/moved.json")).toBe(line);
	});

	it("preserves the hash column when replacing rows for a round", () => {
		const keep = encodeVectorIndexLine([1], "keep.json:prompt", "model-x", "keep-hash");
		const oldPrompt = encodeVectorIndexLine([2], "old.json:prompt", "model-x", "old-hash");
		const oldResponse = encodeVectorIndexLine([3], "old.json:response", "model-x");
		writeIndexLines(indexPath(), [keep, oldPrompt, oldResponse]);

		replaceIndexEntriesForRoundFile(indexPath(), "old.json", [
			{ vector: [4], filePath: "old.json:prompt", model: "model-y", embeddingInputHash: "new-hash" },
			{ vector: [5], filePath: "old.json:response", model: "model-y" },
		]);

		expect(loadVectorIndex(indexPath())).toEqual([
			{ vector: [1], filePath: "keep.json:prompt", model: "model-x", embeddingInputHash: "keep-hash" },
			{ vector: [4], filePath: "old.json:prompt", model: "model-y", embeddingInputHash: "new-hash" },
			{ vector: [5], filePath: "old.json:response", model: "model-y" },
		]);
	});

	it("detects model mismatches on 4-column rows without treating the hash as model", () => {
		writeIndexLines(indexPath(), [
			encodeVectorIndexLine([1], "same.json:prompt", "current-model", "hash-1"),
			encodeVectorIndexLine([2], "different.json:prompt", "old-model", "hash-2"),
		]);

		expect(loadRoundFilesWithDifferentModel(indexPath(), "current-model")).toEqual(new Set(["different.json"]));
	});
});

describe("assumed- provenance stamps (#107 F1)", () => {
	it("makeAssumedStamp prefixes the hash; bareEmbeddingInputHash strips the prefix", () => {
		const hash = "abc-123_xyz"; // base64url alphabet — CSV-safe alongside the ASCII prefix
		const stamp = makeAssumedStamp(hash);
		expect(stamp).toBe(`${ASSUMED_STAMP_PREFIX}${hash}`);
		expect(stamp).not.toMatch(","); // stays a single CSV column
		expect(bareEmbeddingInputHash(stamp)).toBe(hash);
	});

	it("bareEmbeddingInputHash passes plain stamps through unchanged", () => {
		expect(bareEmbeddingInputHash("plain-hash")).toBe("plain-hash");
	});

	it("round-trips an assumed- stamp through the 4th CSV column", () => {
		writeIndexLines(indexPath(), [encodeVectorIndexLine([1], "a.json:prompt", "model-x", makeAssumedStamp("h1"))]);

		expect(loadVectorIndex(indexPath())).toEqual([
			{ vector: [1], filePath: "a.json:prompt", model: "model-x", embeddingInputHash: "assumed-h1" },
		]);
	});
});
