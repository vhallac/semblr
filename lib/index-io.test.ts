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
	flushOwnShardVectorIndex,
	indexEntryFilename,
	indexRoundFileFromPath,
	loadIndexedRoundFiles,
	loadIndexedRoundFilesSharded,
	loadRoundFilesWithDifferentModel,
	loadRoundFilesWithDifferentModelSharded,
	loadShardVectorIndex,
	loadVectorIndex,
	makeAssumedStamp,
	migrateIndexEntryLine,
	readIndexByFilename,
	readIndexLines,
	replaceIndexLineFilename,
	writeIndexLinesAtomic,
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
		writeIndexLinesAtomic(indexPath(), [
			encodeVectorIndexLine([1], "dir/one.json:prompt"),
			encodeVectorIndexLine([2], "two.json:response"),
			encodeVectorIndexLine([3], "three.json:round"),
		]);

		expect(indexRoundFileFromPath("dir/one.json:prompt")).toBe("dir/one.json");
		expect(loadIndexedRoundFiles(indexPath())).toEqual(new Set(["one.json", "two.json", "three.json"]));
	});

	it("detects only explicit non-current model rows as model mismatches", () => {
		writeIndexLinesAtomic(indexPath(), [
			encodeVectorIndexLine([1], "legacy.json:prompt"),
			encodeVectorIndexLine([2], "same.json:prompt", "current-model"),
			encodeVectorIndexLine([3], "different.json:prompt", "old-model"),
			encodeVectorIndexLine([4], "nested/also-different.json:response", "older-model"),
		]);

		expect(loadRoundFilesWithDifferentModel(indexPath(), "current-model")).toEqual(
			new Set(["different.json", "also-different.json"]),
		);
	});

	it("migrates matching index entry prefixes and leaves unrelated rows unchanged", () => {
		const oldLine = encodeVectorIndexLine([1], "old.json:prompt");
		const unrelated = encodeVectorIndexLine([2], "other.json:prompt");

		expect(migrateIndexEntryLine(oldLine, "old.json", "new.json")).toBe(
			encodeVectorIndexLine([1], "new.json:prompt"),
		);
		expect(migrateIndexEntryLine(unrelated, "old.json", "new.json")).toBe(unrelated);
	});

	it("groups rows by filename and filters renamed filenames", () => {
		const keep = encodeVectorIndexLine([1], "keep.json:prompt");
		const oldPrompt = encodeVectorIndexLine([2], "old.json:prompt");
		const oldResponse = encodeVectorIndexLine([3], "old.json:response");
		writeIndexLinesAtomic(indexPath(), [keep, oldPrompt, oldResponse, "malformed"]);

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
		writeIndexLinesAtomic(indexPath(), [
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

	it("detects model mismatches on 4-column rows without treating the hash as model", () => {
		writeIndexLinesAtomic(indexPath(), [
			encodeVectorIndexLine([1], "same.json:prompt", "current-model", "hash-1"),
			encodeVectorIndexLine([2], "different.json:prompt", "old-model", "hash-2"),
		]);

		expect(loadRoundFilesWithDifferentModel(indexPath(), "current-model")).toEqual(new Set(["different.json"]));
	});

	describe("merged shard reads (#149)", () => {
		const writeShard = (shard: string, lines: string[]) =>
			fs.writeFileSync(path.join(tempDir, shard), `${lines.join("\n")}\n`);

		it("merges rows from all discovered shards in sorted shard order", () => {
			writeShard("index.csv", [encodeVectorIndexLine([1], "legacy.json:prompt")]);
			writeShard("index.host-b", [encodeVectorIndexLine([2], "b.json:prompt")]);
			writeShard("index.host-a", [encodeVectorIndexLine([3], "a.json:prompt")]);
			fs.writeFileSync(path.join(tempDir, "index.bm25.json"), "{}");
			fs.writeFileSync(path.join(tempDir, "index-tools.fulltext.csv"), "x");

			expect(loadShardVectorIndex(tempDir)).toEqual([
				{ vector: [1], filePath: "legacy.json:prompt" }, // "index.csv" sorts before "index.host-*"
				{ vector: [3], filePath: "a.json:prompt" },
				{ vector: [2], filePath: "b.json:prompt" },
			]); // legacy index.csv read until adopted
			expect(loadIndexedRoundFilesSharded(tempDir)).toEqual(new Set(["legacy.json", "b.json", "a.json"]));
			expect(loadRoundFilesWithDifferentModelSharded(tempDir, "m2", {})).toEqual(new Set());
			expect(loadRoundFilesWithDifferentModelSharded(tempDir, "current", {})).toEqual(new Set());
			writeShard("index.host-b", [encodeVectorIndexLine([2], "b.json:prompt", "old-model")]);
			expect(loadRoundFilesWithDifferentModelSharded(tempDir, "current")).toEqual(new Set(["b.json"]));
		});

		it("deduplicates a row that two shards hold for the same round (OI-4)", () => {
			writeShard("index.host-a", [encodeVectorIndexLine([1], "round1.json:prompt", "m1")]);
			writeShard("index.host-b", [encodeVectorIndexLine([2], "round1.json:prompt", "m1")]);

			expect(loadShardVectorIndex(tempDir)).toEqual([{ vector: [1], filePath: "round1.json:prompt", model: "m1" }]);
			expect(loadShardVectorIndex(tempDir, { dedupe: false })).toHaveLength(2);
		});

		it("keeps rows of different models apart when merging (OI-4)", () => {
			writeShard("index.host-a", [encodeVectorIndexLine([1], "round1.json:prompt", "m1")]);
			writeShard("index.host-b", [encodeVectorIndexLine([2], "round1.json:prompt", "m2")]);

			expect(loadShardVectorIndex(tempDir).map((entry) => entry.model)).toEqual(["m1", "m2"]);
		});

		it("tolerates a missing rounds dir and an empty shard set", () => {
			const missing = path.join(tempDir, "nope");
			expect(loadShardVectorIndex(missing)).toEqual([]);
			expect(loadIndexedRoundFilesSharded(missing)).toEqual(new Set());
			expect(loadRoundFilesWithDifferentModelSharded(missing, "m")).toEqual(new Set());
			expect(loadShardVectorIndex(tempDir)).toEqual([]);
		});

		it("skips corrupt rows with a warning naming the shard file", () => {
			writeShard("index.host-a", [
				encodeVectorIndexLine([1], "good.json:prompt"),
				"!!!not-base64!!!,corrupt.json:prompt",
				"bm90LXZhbGlkLWpzb24=,partial.json:prompt", // decodes to non-JSON
			]);
			const warnings: string[] = [];
			const entries = loadShardVectorIndex(tempDir, {
				warn: (message) => warnings.push(message),
			});

			expect(entries).toEqual([{ vector: [1], filePath: "good.json:prompt" }]);
			expect(warnings).toHaveLength(2);
			for (const warning of warnings) {
				expect(warning).toContain(path.join(tempDir, "index.host-a"));
			}
		});

		it("flushes a wholesale rewrite to the host's own shard file", () => {
			const shardPath = flushOwnShardVectorIndex(tempDir, [{ vector: [1], filePath: "a.json:prompt" }], {
				hostname: "host-a",
			});

			expect(shardPath).toBe(path.join(tempDir, "index.host-a"));
			expect(fs.readFileSync(shardPath, "utf-8")).toBe(`${encodeVectorIndexLine([1], "a.json:prompt")}\n`);
			// A wholesale rewrite never touches the legacy file or a foreign shard.
			expect(fs.existsSync(path.join(tempDir, "index.csv"))).toBe(false);
			expect(fs.existsSync(path.join(tempDir, "index.host-b"))).toBe(false);
		});

		it("holds the shard append lock across the flush and releases it (PR #151 R1.F1)", () => {
			let lockSeenDuringWrite = false;
			const shardPath = flushOwnShardVectorIndex(tempDir, [{ vector: [1], filePath: "a.json:prompt" }], {
				hostname: "host-a",
				fsImpl: {
					...fs,
					writeFileSync: (file: Parameters<typeof fs.writeFileSync>[0], ...rest: unknown[]) => {
						if (fs.existsSync(`${path.join(tempDir, "index.host-a")}.lock`)) {
							lockSeenDuringWrite = true;
						}
						return (fs.writeFileSync as (...a: unknown[]) => ReturnType<typeof fs.writeFileSync>)(file, ...rest);
					},
				},
			});

			expect(shardPath).toBe(path.join(tempDir, "index.host-a"));
			expect(lockSeenDuringWrite).toBe(true);
			expect(fs.existsSync(`${shardPath}.lock`)).toBe(false);
			expect(fs.readFileSync(shardPath, "utf-8")).toBe(`${encodeVectorIndexLine([1], "a.json:prompt")}\n`);
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

			expect(() =>
				flushOwnShardVectorIndex(tempDir, [{ vector: [1], filePath: "a.json:prompt" }], {
					hostname: "host-a",
					fsImpl,
					lockRetries: 2,
					wait: () => {},
				}),
			).toThrow(/NOT written/);
			expect(fs.existsSync(path.join(tempDir, "index.host-a"))).toBe(false);
		});

		it("honors an explicit shard list", () => {
			writeShard("index.host-a", [encodeVectorIndexLine([1], "a.json:prompt")]);
			writeShard("index.host-b", [encodeVectorIndexLine([2], "b.json:prompt")]);
			const onlyB = path.join(tempDir, "index.host-b");

			expect(loadShardVectorIndex(tempDir, { shards: [onlyB] })).toEqual([
				{ vector: [2], filePath: "b.json:prompt" },
			]);
		});
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
		writeIndexLinesAtomic(indexPath(), [
			encodeVectorIndexLine([1], "a.json:prompt", "model-x", makeAssumedStamp("h1")),
		]);

		expect(loadVectorIndex(indexPath())).toEqual([
			{ vector: [1], filePath: "a.json:prompt", model: "model-x", embeddingInputHash: "assumed-h1" },
		]);
	});
});
