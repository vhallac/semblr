import { describe, expect, it } from "vitest";
import {
	appendToIndexPath,
	buildSessionStartStatus,
	countUniqueIndexedRounds,
	type IndexEntry,
	loadIndexFromPath,
	loadSessionStartIndex,
} from "./index-storage.ts";

describe("loadIndexFromPath", () => {
	it("returns empty array when file does not exist", () => {
		const fsMock = {
			existsSync: () => false,
			readFileSync: () => {
				throw new Error("should not be called");
			},
		} as unknown as Pick<typeof import("node:fs"), "existsSync" | "readFileSync">;
		expect(loadIndexFromPath("/fake.csv", fsMock)).toEqual([]);
	});

	it("returns empty array for empty file", () => {
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => "  ",
		} as unknown as Pick<typeof import("node:fs"), "existsSync" | "readFileSync">;
		expect(loadIndexFromPath("/fake.csv", fsMock)).toEqual([]);
	});

	it("parses index entries from file", () => {
		const vector = [0.1, 0.2, 0.3];
		const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
		const content = `${b64},rounds/abc.json\n${b64},rounds/def.json\n`;
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => content,
		} as unknown as Pick<typeof import("node:fs"), "existsSync" | "readFileSync">;
		const result = loadIndexFromPath("/fake.csv", fsMock);
		expect(result).toHaveLength(2);
		expect(result[0].filePath).toBe("rounds/abc.json");
		expect(result[0].vector).toEqual(vector);
		expect(result[1].filePath).toBe("rounds/def.json");
	});

	it("handles single entry without trailing newline", () => {
		const vector = [1.0, 0.0];
		const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => `${b64},rounds/test.json`,
		} as unknown as Pick<typeof import("node:fs"), "existsSync" | "readFileSync">;
		const result = loadIndexFromPath("/fake.csv", fsMock);
		expect(result).toHaveLength(1);
		expect(result[0].filePath).toBe("rounds/test.json");
	});

	it("returns empty vector for non-array decoded data", () => {
		const b64 = Buffer.from(JSON.stringify("not-array")).toString("base64url");
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => `${b64},rounds/test.json\n`,
		} as unknown as Pick<typeof import("node:fs"), "existsSync" | "readFileSync">;
		const result = loadIndexFromPath("/fake.csv", fsMock);
		expect(result[0].vector).toEqual([]);
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

describe("appendToIndexPath", () => {
	it("appends a line to the index file using lock-based atomic write", () => {
		let fileData = "existing,data\n";
		const fsImpl = {
			mkdirSync: () => {},
			existsSync: (_p: unknown) => true,
			readFileSync: () => fileData,
			writeFileSync: (_p: unknown, data: string) => {
				fileData = data;
			},
			renameSync: () => {},
			appendFileSync: () => {},
			openSync: () => 42 as unknown as number,
			closeSync: () => {},
			unlinkSync: () => {},
			statSync: () => ({ mtimeMs: Date.now() }),
		} as unknown as NonNullable<Parameters<typeof appendToIndexPath>[4]>["fsImpl"];

		appendToIndexPath("/rounds/index.csv", "/rounds", "rounds/test.json", [0.1, 0.2], { fsImpl });

		// The vector should be encoded in base64url
		const expectedB64 = Buffer.from(JSON.stringify([0.1, 0.2])).toString("base64url");
		expect(fileData).toContain(expectedB64);
		expect(fileData).toContain("rounds/test.json");
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
				readFileSync: () => "",
				writeFileSync: () => {},
				renameSync: () => {},
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
				readFileSync: () => "line1\n",
				writeFileSync: () => {},
				renameSync: () => {},
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
				readFileSync: () => "line1\n",
				writeFileSync: () => {},
				renameSync: () => {},
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
