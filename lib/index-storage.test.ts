import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	acquireIndexLock,
	appendLineWithLock,
	appendToIndexPath,
	buildSessionStartStatus,
	countIndexLines,
	countUniqueIndexedRounds,
	type IndexEntry,
	loadIndexFromPath,
	loadSessionStartIndex,
} from "./index-storage.ts";

/** Chunk size small enough to force multi-chunk reads in tests. */
const TINY_CHUNK = 16;

describe("loadIndexFromPath", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-load-index-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	it("returns empty array when file does not exist", () => {
		expect(loadIndexFromPath(path.join(tempDir, "missing.csv"))).toEqual([]);
	});

	it("returns empty array for blank file", () => {
		const p = path.join(tempDir, "blank.csv");
		fs.writeFileSync(p, "  ");
		expect(loadIndexFromPath(p)).toEqual([]);
	});

	it("parses index entries from file", () => {
		const vector = [0.1, 0.2, 0.3];
		const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
		const p = path.join(tempDir, "index.csv");
		fs.writeFileSync(p, `${b64},rounds/abc.json\n${b64},rounds/def.json\n`);
		const result = loadIndexFromPath(p);
		expect(result).toHaveLength(2);
		expect(result[0].filePath).toBe("rounds/abc.json");
		expect(result[0].vector).toEqual(vector);
		expect(result[1].filePath).toBe("rounds/def.json");
	});

	it("handles single entry without trailing newline", () => {
		const vector = [1.0, 0.0];
		const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
		const p = path.join(tempDir, "index.csv");
		fs.writeFileSync(p, `${b64},rounds/test.json`);
		const result = loadIndexFromPath(p);
		expect(result).toHaveLength(1);
		expect(result[0].filePath).toBe("rounds/test.json");
	});

	describe("chunked reads", () => {
		const writeLines = (name: string, count: number): { p: string; vectors: number[][] } => {
			const p = path.join(tempDir, name);
			const vectors: number[][] = [];
			let content = "";
			for (let i = 0; i < count; i++) {
				const vector = [i / 10, 1 - i / 10];
				vectors.push(vector);
				const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
				content += `${b64},rounds/r${i}.json\n`;
			}
			fs.writeFileSync(p, content);
			return { p, vectors };
		};

		it("reconstructs all entries when lines span chunk boundaries", () => {
			// TINY_CHUNK (16 bytes) is far smaller than any line, forcing many boundary splits.
			const { p, vectors } = writeLines("chunked.csv", 12);
			const result = loadIndexFromPath(p, fs, TINY_CHUNK);
			expect(result).toHaveLength(12);
			result.forEach((entry, i) => {
				expect(entry.filePath).toBe(`rounds/r${i}.json`);
				expect(entry.vector).toEqual(vectors[i]);
			});
		});

		it("returns identical results for one-line and default-chunk reads", () => {
			const { p, vectors } = writeLines("single.csv", 1);
			const tiny = loadIndexFromPath(p, fs, TINY_CHUNK);
			const full = loadIndexFromPath(p);
			expect(tiny).toEqual(full);
			expect(tiny[0].vector).toEqual(vectors[0]);
		});

		it("reconstructs a line split exactly at a chunk boundary", () => {
			const vector = [0.25, 0.75];
			const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
			const line = `${b64},rounds/boundary.json`;
			const bytes = Buffer.from(line, "utf-8");
			const p = path.join(tempDir, "boundary.csv");
			// First chunk ends exactly where the line ends; the newline lands in the next chunk.
			fs.writeFileSync(p, Buffer.concat([bytes, Buffer.from("\n", "utf-8"), bytes, Buffer.from("\n", "utf-8")]));
			const result = loadIndexFromPath(p, fs, bytes.length);
			expect(result).toHaveLength(2);
			expect(result[0].filePath).toBe("rounds/boundary.json");
			expect(result[1].vector).toEqual(vector);
		});
	});

	it("parses the optional 4th column as embeddingInputHash, keeping model intact", () => {
		const vector = [0.5, 0.5];
		const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
		const p = path.join(tempDir, "stamped.csv");
		fs.writeFileSync(
			p,
			`${b64},rounds/stamped.json:prompt,mymodel,abc123\n${b64},rounds/legacy.json:prompt,model-x\n`,
		);
		const result = loadIndexFromPath(p);
		expect(result[0]).toEqual({
			filePath: "rounds/stamped.json:prompt",
			vector,
			model: "mymodel",
			embeddingInputHash: "abc123",
		});
		expect(result[1]).toEqual({
			filePath: "rounds/legacy.json:prompt",
			vector,
			model: "model-x",
		});
	});

	it("returns empty vector for non-array decoded data", () => {
		const b64 = Buffer.from(JSON.stringify("not-array")).toString("base64url");
		const p = path.join(tempDir, "nonarray.csv");
		fs.writeFileSync(p, `${b64},rounds/test.json\n`);
		const result = loadIndexFromPath(p);
		expect(result[0].vector).toEqual([]);
	});
});

describe("countIndexLines", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-count-index-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	it("returns 0 when file does not exist", () => {
		expect(countIndexLines(path.join(tempDir, "missing.csv"))).toBe(0);
	});

	it("counts non-empty lines and skips blank lines", () => {
		const p = path.join(tempDir, "index.csv");
		fs.writeFileSync(p, "a,b\n\nc,d\n\n");
		expect(countIndexLines(p)).toBe(2);
	});

	it("counts across chunk boundaries including a trailing partial line", () => {
		const p = path.join(tempDir, "chunked.csv");
		fs.writeFileSync(p, "line1\nline2\nline3");
		expect(countIndexLines(p, fs, 4)).toBe(3);
	});

	it("matches loadIndexFromPath entry count on realistic rows", () => {
		const b64 = Buffer.from(JSON.stringify([0.1])).toString("base64url");
		const p = path.join(tempDir, "index.csv");
		fs.writeFileSync(p, Array.from({ length: 5 }, (_, i) => `${b64},rounds/r${i}.json`).join("\n") + "\n");
		expect(countIndexLines(p, fs, 8)).toBe(loadIndexFromPath(p, fs, 8).length);
	});
});

describe("loadSessionStartIndex", () => {
	it("returns empty array when file does not exist", () => {
		const deps = {
			existsSync: () => false,
			loadIndex: () => [] as IndexEntry[],
		};
		expect(loadSessionStartIndex("/fake.csv", deps)).toEqual([]);
	});

	it("loads index when file exists", () => {
		const entries: IndexEntry[] = [{ filePath: "rounds/test.json", vector: [0.5] }];
		const deps = {
			existsSync: () => true,
			loadIndex: () => entries,
		};
		const result = loadSessionStartIndex("/fake.csv", deps);
		expect(result).toBe(entries);
	});

	it("uses default loadIndex when not provided", () => {
		const entries: IndexEntry[] = [{ filePath: "rounds/test.json", vector: [0.5] }];
		const deps = {
			existsSync: () => true,
			loadIndex: () => entries,
		};
		const result = loadSessionStartIndex("/fake.csv", deps);
		expect(result).toBe(entries);
	});
});

describe("countUniqueIndexedRounds", () => {
	it("counts unique round file paths by stripping suffixes", () => {
		const entries: IndexEntry[] = [
			{ filePath: "rounds/abc.json:prompt", vector: [0.1] },
			{ filePath: "rounds/abc.json:response", vector: [0.2] },
			{ filePath: "rounds/def.json:round", vector: [0.3] },
		];
		expect(countUniqueIndexedRounds(entries)).toBe(2);
	});

	it("returns 0 for empty index", () => {
		expect(countUniqueIndexedRounds([])).toBe(0);
	});

	it("counts entries without suffix correctly", () => {
		const entries: IndexEntry[] = [
			{ filePath: "rounds/abc.json", vector: [0.1] },
			{ filePath: "rounds/def.json", vector: [0.2] },
		];
		expect(countUniqueIndexedRounds(entries)).toBe(2);
	});
});

describe("buildSessionStartStatus", () => {
	it("builds status message with unique round count", () => {
		const entries: IndexEntry[] = [
			{ filePath: "rounds/abc.json:prompt", vector: [0.1] },
			{ filePath: "rounds/abc.json:response", vector: [0.2] },
		];
		const status = buildSessionStartStatus(entries);
		expect(status).toBe("🧠 semblr loaded — 1 rounds indexed");
	});
});

describe("appendLineWithLock", () => {
	it("appends without reading or rewriting the existing file", () => {
		const appended: string[] = [];
		appendLineWithLock("/rounds/index.csv", "/rounds", "new,line\n", {
			fsImpl: {
				mkdirSync: () => {},
				appendFileSync: (_p: unknown, data: string) => appended.push(data),
				openSync: () => 1 as unknown as number,
				closeSync: () => {},
				unlinkSync: () => {},
				statSync: () => ({ mtimeMs: Date.now() }),
			} as unknown as NonNullable<Parameters<typeof appendLineWithLock>[3]>["fsImpl"],
		});
		expect(appended).toEqual(["new,line\n"]);
	});

	describe("appendToIndexPath", () => {
		it("appends the encoded line under the lock without rewriting the file", () => {
			const appended: string[] = [];
			const fsImpl = {
				mkdirSync: () => {},
				existsSync: () => true,
				appendFileSync: (_p: unknown, data: string) => {
					appended.push(data);
				},
				openSync: () => 42 as unknown as number,
				closeSync: () => {},
				unlinkSync: () => {},
				statSync: () => ({ mtimeMs: Date.now() }),
			} as unknown as NonNullable<Parameters<typeof appendToIndexPath>[4]>["fsImpl"];

			appendToIndexPath("/rounds/index.csv", "/rounds", "rounds/test.json", [0.1, 0.2], { fsImpl });

			// The vector should be encoded in base64url and appended as one line
			const expectedB64 = Buffer.from(JSON.stringify([0.1, 0.2])).toString("base64url");
			expect(appended).toEqual([`${expectedB64},rounds/test.json\n`]);
		});

		it("falls back to appendFileSync when lock cannot be acquired", () => {
			let appendCalled = false;
			const alwaysFailOpen = () => {
				throw new Error("cannot open");
			};

			appendToIndexPath("/rounds/index.csv", "/rounds", "rounds/test.json", [0.1], {
				fsImpl: {
					mkdirSync: () => {},
					existsSync: () => true,
					appendFileSync: () => {
						appendCalled = true;
					},
					openSync: alwaysFailOpen,
					closeSync: () => {},
					unlinkSync: () => {},
					statSync: () => ({ mtimeMs: Date.now() as unknown as bigint }),
				} as unknown as NonNullable<Parameters<typeof appendToIndexPath>[4]>["fsImpl"],
				lockRetries: 2,
				lockBackoffMs: 1,
				wait: () => {},
				now: () => Date.now(),
				processId: 123,
			});

			expect(appendCalled).toBe(true);
		});

		it("handles stale lock by removing it", () => {
			let unlinkCalled = false;
			const oldTime = Date.now() - 20000; // stale
			let openAttempts = 0;

			appendToIndexPath("/rounds/index.csv", "/rounds", "rounds/test.json", [0.1], {
				fsImpl: {
					mkdirSync: () => {},
					existsSync: () => true,
					appendFileSync: () => {},
					openSync: () => {
						openAttempts++;
						if (openAttempts === 1) throw new Error("lock exists");
						return 42;
					},
					closeSync: () => {},
					unlinkSync: () => {
						unlinkCalled = true;
					},
					statSync: () => ({ mtimeMs: oldTime as unknown as bigint }),
				} as unknown as NonNullable<Parameters<typeof appendToIndexPath>[4]>["fsImpl"],
				lockRetries: 5,
				lockBackoffMs: 1,
				wait: () => {},
				now: () => Date.now(),
				processId: 123,
			});

			expect(unlinkCalled).toBe(true);
		});

		it("handles stale lock where statSync throws (lock disappeared)", () => {
			let openAttempts = 0;

			appendToIndexPath("/rounds/index.csv", "/rounds", "rounds/test.json", [0.1], {
				fsImpl: {
					mkdirSync: () => {},
					existsSync: () => true,
					appendFileSync: () => {},
					openSync: () => {
						openAttempts++;
						if (openAttempts <= 1) throw new Error("lock exists");
						return 42;
					},
					closeSync: () => {},
					unlinkSync: () => {
						// lock disappeared
					},
					statSync: () => {
						throw new Error("ENOENT");
					},
				} as unknown as NonNullable<Parameters<typeof appendToIndexPath>[4]>["fsImpl"],
				lockRetries: 5,
				lockBackoffMs: 1,
				wait: () => {},
				now: () => Date.now(),
				processId: 123,
			});

			// Should successfully acquire on retry after statSync throws
			expect(openAttempts).toBeGreaterThan(1);
		});
	});

	describe("acquireIndexLock", () => {
		let tempDir = "";

		beforeEach(() => {
			tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-acquire-index-lock-"));
		});

		afterEach(() => {
			fs.rmSync(tempDir, { recursive: true, force: true });
		});

		it("creates the lockfile on acquire and removes it on release", () => {
			const target = path.join(tempDir, "index.csv");
			const lock = acquireIndexLock(target);

			expect(lock).not.toBeNull();
			expect(lock?.lockPath).toBe(`${target}.lock`);
			expect(fs.existsSync(`${target}.lock`)).toBe(true);

			lock?.release();
			expect(fs.existsSync(`${target}.lock`)).toBe(false);
		});

		it("returns null on exhaustion without touching the holder's lockfile", () => {
			const target = path.join(tempDir, "index.csv");
			fs.writeFileSync(`${target}.lock`, "held");

			const lock = acquireIndexLock(target, { lockRetries: 2, lockBackoffMs: 1, wait: () => {} });

			expect(lock).toBeNull();
			// The existing holder's lock must survive our failed acquisition.
			expect(fs.existsSync(`${target}.lock`)).toBe(true);
		});

		it("takes over a stale lockfile", () => {
			const target = path.join(tempDir, "index.csv");
			const staleTime = Date.now() - 20_000;
			fs.writeFileSync(`${target}.lock`, "stale");
			fs.utimesSync(`${target}.lock`, staleTime / 1000, staleTime / 1000);

			const lock = acquireIndexLock(target, { lockRetries: 2, lockBackoffMs: 1, wait: () => {} });

			expect(lock).not.toBeNull();
			expect(fs.existsSync(`${target}.lock`)).toBe(true); // re-created by us
			lock?.release();
			expect(fs.existsSync(`${target}.lock`)).toBe(false);
		});
	});
});
