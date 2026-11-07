import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { ToolCallDetail } from "./round-data.ts";
import {
	appendToolIndexRows,
	buildSearchableText,
	buildToolIndexRows,
	buildToolIndexRowsFromRoundsDir,
	encodeToolIndexRow,
	loadToolIndex,
	loadToolIndexedRoundFiles,
	ownToolIndexPath,
	parseToolIndexLine,
	searchToolIndex,
	type ToolIndexRow,
	toolIndexPathForRoundsDir,
	writeToolIndexRows,
} from "./search-tools.ts";

const tmpDirs: string[] = [];
function tmpDir(): string {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-search-tools-test-"));
	tmpDirs.push(dir);
	return dir;
}

afterEach(() => {
	for (const dir of tmpDirs.splice(0)) {
		fs.rmSync(dir, { recursive: true, force: true });
	}
});

describe("toolIndexPathForRoundsDir", () => {
	it("builds a sibling path to index.csv", () => {
		expect(toolIndexPathForRoundsDir("/foo/rounds")).toBe("/foo/rounds/index-tools.fulltext.csv");
	});
});

describe("buildSearchableText", () => {
	it("concatenates tool name and string argument values, lowercased and whitespace-normalized", () => {
		const args = JSON.stringify({ command: "curl -s https://api.github.com", description: "fetch  api" });
		expect(buildSearchableText("bash", args)).toBe("bash curl -s https://api.github.com fetch api");
	});

	it("recurses into nested objects and arrays for string values", () => {
		const args = JSON.stringify({ nested: { a: "Hello World", list: ["Foo", "Bar"] }, count: 3, flag: true });
		expect(buildSearchableText("tool", args)).toBe("tool hello world foo bar");
	});

	it("falls back to the raw arguments string when JSON parsing fails", () => {
		expect(buildSearchableText("bash", "not json")).toBe("bash not json");
	});

	it("collapses newlines and tabs into single spaces", () => {
		const args = JSON.stringify({ command: "line1\nline2\tline3" });
		expect(buildSearchableText("bash", args)).toBe("bash line1 line2 line3");
	});

	it("ignores non-string primitive values", () => {
		const args = JSON.stringify({ count: 5, flag: false, nothing: null });
		expect(buildSearchableText("tool", args)).toBe("tool");
	});
});

describe("buildToolIndexRows", () => {
	it("builds one row per tool call", () => {
		const toolCalls: ToolCallDetail[] = [
			{ index: 0, name: "bash", arguments: JSON.stringify({ command: "curl x" }), result_summary: "" },
			{ index: 1, name: "read", arguments: JSON.stringify({ path: "/tmp/x" }), result_summary: "" },
		];
		const rows = buildToolIndexRows("abc.json", toolCalls);
		expect(rows).toEqual([
			{ hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "bash curl x" },
			{ hash: "abc.json", toolIndex: 1, toolName: "read", searchableText: "read /tmp/x" },
		]);
	});

	it("returns an empty array for no tool calls", () => {
		expect(buildToolIndexRows("abc.json", [])).toEqual([]);
	});
});

describe("encodeToolIndexRow / parseToolIndexLine", () => {
	it("round-trips a row", () => {
		const row: ToolIndexRow = { hash: "abc.json", toolIndex: 2, toolName: "bash", searchableText: "bash curl x" };
		const line = encodeToolIndexRow(row);
		expect(line).toBe("abc.json,2,bash,bash curl x");
		expect(parseToolIndexLine(line)).toEqual(row);
	});

	it("preserves commas embedded in the searchable text (last field)", () => {
		const row: ToolIndexRow = { hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "a, b, c" };
		const line = encodeToolIndexRow(row);
		expect(parseToolIndexLine(line)).toEqual(row);
	});

	it("returns null for malformed lines", () => {
		expect(parseToolIndexLine("no-commas-here")).toBeNull();
		expect(parseToolIndexLine("abc.json,notanumber,bash,text")).toBeNull();
		expect(parseToolIndexLine("abc.json,0")).toBeNull();
		expect(parseToolIndexLine("abc.json,0,bash")).toBeNull();
	});
});

describe("loadToolIndex / loadToolIndexedRoundFiles", () => {
	it("returns empty array when the file does not exist", () => {
		expect(loadToolIndex("/does/not/exist.csv")).toEqual([]);
		expect(loadToolIndexedRoundFiles("/does/not/exist.csv")).toEqual(new Set());
	});

	it("loads and parses rows, skipping malformed lines", () => {
		const dir = tmpDir();
		const indexPath = path.join(dir, "index-tools.fulltext.csv");
		fs.writeFileSync(indexPath, "abc.json,0,bash,curl x\nmalformed\ndef.json,1,read,/tmp/y\n");
		const rows = loadToolIndex(indexPath);
		expect(rows).toEqual([
			{ hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "curl x" },
			{ hash: "def.json", toolIndex: 1, toolName: "read", searchableText: "/tmp/y" },
		]);
		expect(loadToolIndexedRoundFiles(indexPath)).toEqual(new Set(["abc.json", "def.json"]));
	});

	it("handles an empty file", () => {
		const dir = tmpDir();
		const indexPath = path.join(dir, "index-tools.fulltext.csv");
		fs.writeFileSync(indexPath, "");
		expect(loadToolIndex(indexPath)).toEqual([]);
	});
});

describe("appendToolIndexRows", () => {
	it("appends rows to a new file", () => {
		const dir = tmpDir();
		const indexPath = path.join(dir, "index-tools.fulltext.csv");
		appendToolIndexRows(indexPath, dir, [
			{ hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "curl x" },
		]);
		expect(loadToolIndex(indexPath)).toEqual([
			{ hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "curl x" },
		]);
	});

	it("appends additional rows to an existing file", () => {
		const dir = tmpDir();
		const indexPath = path.join(dir, "index-tools.fulltext.csv");
		appendToolIndexRows(indexPath, dir, [
			{ hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "curl x" },
		]);
		appendToolIndexRows(indexPath, dir, [
			{ hash: "def.json", toolIndex: 0, toolName: "read", searchableText: "/tmp/y" },
		]);
		expect(loadToolIndex(indexPath)).toHaveLength(2);
	});

	it("does nothing for an empty rows array", () => {
		const dir = tmpDir();
		const indexPath = path.join(dir, "index-tools.fulltext.csv");
		appendToolIndexRows(indexPath, dir, []);
		expect(fs.existsSync(indexPath)).toBe(false);
	});

	// claim: issue-152/legacy-keyword-adoption-rename-first-write/adopts-the-legacy-artifact-on-the-first-append-to-a-hosts-own-path
	// claim text: the tool-index append path adopts the legacy artifact by rename on the first write (same clobber guard and lost-race semantics as the rewrite path)
	// scenario: a store with only index-tools.fulltext.csv — the first runtime append renames it onto the own path and keeps the legacy rows
	it("adopts the legacy artifact on the first append to a host's own path", () => {
		const dir = tmpDir();
		const legacyPath = path.join(dir, "index-tools.fulltext.csv");
		const ownPath = ownToolIndexPath(dir, "host-a");
		fs.mkdirSync(dir, { recursive: true });
		fs.writeFileSync(legacyPath, "legacy.json,0,bash,legacy text\n");

		appendToolIndexRows(ownPath, dir, [
			{ hash: "new.json", toolIndex: 0, toolName: "read", searchableText: "appended row" },
		]);

		// The legacy file is gone — renamed onto the own path, and the appended
		// row lands beside the legacy rows (no full rewrite needed).
		expect(fs.existsSync(legacyPath)).toBe(false);
		expect(loadToolIndex(ownPath)).toEqual([
			{ hash: "legacy.json", toolIndex: 0, toolName: "bash", searchableText: "legacy text" },
			{ hash: "new.json", toolIndex: 0, toolName: "read", searchableText: "appended row" },
		]);
	});
});

describe("per-host tool index artifacts (issue #152, decision 024)", () => {
	it("ownToolIndexPath places the artifact beside the shards", () => {
		expect(ownToolIndexPath("/rounds", "laptop-x")).toBe(path.join("/rounds", "index-tools.laptop-x.fulltext.csv"));
	});

	it("adopts the legacy artifact by rename on the first write to a host's own path", () => {
		const dir = tmpDir();
		const legacyPath = path.join(dir, "index-tools.fulltext.csv");
		const ownPath = ownToolIndexPath(dir, "host-a");
		fs.mkdirSync(dir, { recursive: true });
		fs.writeFileSync(legacyPath, "legacy.json,0,bash,legacy text\n");

		writeToolIndexRows(ownPath, [{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "fresh text" }]);

		// The legacy file is gone — renamed onto the own path and immediately
		// replaced by the rewrite built from locally visible rounds.
		expect(fs.existsSync(legacyPath)).toBe(false);
		expect(loadToolIndex(ownPath)).toEqual([
			{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "fresh text" },
		]);
	});

	it("leaves the legacy artifact untouched when the own artifact already exists (clobber guard)", () => {
		const dir = tmpDir();
		const legacyPath = path.join(dir, "index-tools.fulltext.csv");
		const ownPath = ownToolIndexPath(dir, "host-a");
		fs.mkdirSync(dir, { recursive: true });
		writeToolIndexRows(ownPath, [
			{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "host artifact" },
		]);
		const legacyBytes = "legacy.json,0,bash,legacy text\n";
		fs.writeFileSync(legacyPath, legacyBytes);

		writeToolIndexRows(ownPath, [
			{ hash: "b.json", toolIndex: 0, toolName: "bash", searchableText: "another rewrite" },
		]);

		// Renaming would clobber the host's own artifact (POSIX rename
		// overwrites) — the orphan legacy file stays on disk untouched.
		expect(fs.readFileSync(legacyPath, "utf-8")).toBe(legacyBytes);
	});

	it("swallows a lost adoption race and still writes its own artifact", () => {
		const dir = tmpDir();
		const legacyPath = path.join(dir, "index-tools.fulltext.csv");
		const ownPath = ownToolIndexPath(dir, "host-a");
		fs.mkdirSync(dir, { recursive: true });
		fs.writeFileSync(legacyPath, "legacy.json,0,bash,legacy text\n");
		const fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "renameSync"> = {
			existsSync: fs.existsSync,
			mkdirSync: fs.mkdirSync,
			writeFileSync: fs.writeFileSync,
			renameSync: (from, to) => {
				if (String(from) === legacyPath) throw new Error("claimed by another host");
				return fs.renameSync(from, to);
			},
		};

		writeToolIndexRows(ownPath, [{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "text" }], fsImpl);

		// The lost legacy claim does not block the rewrite.
		expect(loadToolIndex(ownPath)).toEqual([
			{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "text" },
		]);
	});

	it("writes the artifact atomically: staged tmp write, rename publish, never in-place (decision 015)", () => {
		const dir = tmpDir();
		const indexPath = ownToolIndexPath(dir, "host-a");
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

		writeToolIndexRows(
			indexPath,
			[{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "text" }],
			fsImpl,
		);

		const tmpPath = `${indexPath}.tmp.${process.pid}`;
		expect(writes).toEqual([tmpPath]);
		expect(renames).toEqual([[tmpPath, indexPath]]);
		expect(fs.existsSync(tmpPath)).toBe(false);
	});

	it("writes the legacy path directly without attempting adoption", () => {
		const dir = tmpDir();
		const legacyPath = toolIndexPathForRoundsDir(dir);
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

		writeToolIndexRows(
			legacyPath,
			[{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "text" }],
			fsImpl,
		);

		// Only the atomic-publish rename runs — no legacy adoption rename.
		expect(renames).toEqual([[`${legacyPath}.tmp.${process.pid}`, legacyPath]]);
		expect(loadToolIndex(legacyPath)).toEqual([
			{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "text" },
		]);
	});
});

describe("writeToolIndexRows", () => {
	it("overwrites the file with exactly the given rows", () => {
		const dir = tmpDir();
		const indexPath = path.join(dir, "index-tools.fulltext.csv");
		fs.writeFileSync(indexPath, "stale.json,0,bash,old\n");
		writeToolIndexRows(indexPath, [{ hash: "new.json", toolIndex: 0, toolName: "bash", searchableText: "new" }]);
		expect(loadToolIndex(indexPath)).toEqual([
			{ hash: "new.json", toolIndex: 0, toolName: "bash", searchableText: "new" },
		]);
	});

	it("writes an empty file for an empty rows array", () => {
		const dir = tmpDir();
		const indexPath = path.join(dir, "index-tools.fulltext.csv");
		writeToolIndexRows(indexPath, []);
		expect(fs.readFileSync(indexPath, "utf-8")).toBe("");
	});
});

describe("buildToolIndexRowsFromRoundsDir", () => {
	it("scans round files and builds rows, skipping rounds with no tool calls", () => {
		const dir = tmpDir();
		fs.writeFileSync(
			path.join(dir, "abc.json"),
			JSON.stringify({
				userPrompt: "hi",
				responseSequence: "hi",
				toolCalls: [{ index: 0, name: "bash", arguments: JSON.stringify({ command: "curl x" }) }],
			}),
		);
		fs.writeFileSync(path.join(dir, "def.json"), JSON.stringify({ userPrompt: "hi", responseSequence: "hi" }));
		fs.writeFileSync(path.join(dir, "index.csv"), "not-a-round");

		const rows = buildToolIndexRowsFromRoundsDir(dir);
		expect(rows).toEqual([{ hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "bash curl x" }]);
	});

	it("skips corrupt round files", () => {
		const dir = tmpDir();
		fs.writeFileSync(path.join(dir, "broken.json"), "{not valid json");
		expect(buildToolIndexRowsFromRoundsDir(dir)).toEqual([]);
	});

	it("returns an empty array for an empty directory", () => {
		const dir = tmpDir();
		expect(buildToolIndexRowsFromRoundsDir(dir)).toEqual([]);
	});
});

describe("searchToolIndex", () => {
	const rows: ToolIndexRow[] = [
		{ hash: "a.json", toolIndex: 0, toolName: "bash", searchableText: "bash curl https://thinkerer.dev search" },
		{ hash: "a.json", toolIndex: 1, toolName: "read", searchableText: "read /tmp/x" },
		{ hash: "b.json", toolIndex: 0, toolName: "bash", searchableText: "bash curl https://example.com" },
		{ hash: "c.json", toolIndex: 0, toolName: "bash", searchableText: "bash curl https://example.com" },
		{ hash: "c.json", toolIndex: 1, toolName: "bash", searchableText: "bash curl https://example.com" },
		{ hash: "c.json", toolIndex: 2, toolName: "bash", searchableText: "bash curl https://example.com" },
	];

	it("returns empty array for an empty query", () => {
		expect(searchToolIndex(rows, "   ")).toEqual([]);
	});

	it("matches a single keyword case-insensitively", () => {
		const results = searchToolIndex(rows, "CURL");
		expect(results.map((r) => r.hash).sort()).toEqual(["a.json", "b.json", "c.json"]);
		for (const r of results) {
			expect(r.matchedKeywordCount).toBe(1);
			expect(r.totalKeywords).toBe(1);
		}
	});

	it("scores by unique keyword coverage, not row count", () => {
		// "curl" matches in call 0 and "thinkerer" also matches in call 0 for a.json (score 2).
		// c.json has "curl" matching 3 times across 3 calls but only 1 unique keyword (score 1).
		const results = searchToolIndex(rows, "curl thinkerer");
		const aResult = results.find((r) => r.hash === "a.json");
		const cResult = results.find((r) => r.hash === "c.json");
		expect(aResult?.matchedKeywordCount).toBe(2);
		expect(cResult?.matchedKeywordCount).toBe(1);
		// a.json (score 2) should rank above b.json/c.json (score 1)
		expect(results[0].hash).toBe("a.json");
	});

	it("collects matched tool indices and names, deduplicated and sorted by index", () => {
		const results = searchToolIndex(rows, "curl");
		const cResult = results.find((r) => r.hash === "c.json");
		expect(cResult?.matchedTools).toEqual([
			{ index: 0, name: "bash" },
			{ index: 1, name: "bash" },
			{ index: 2, name: "bash" },
		]);
	});

	it("respects rounds[] scoping", () => {
		const results = searchToolIndex(rows, "curl", ["b.json"]);
		expect(results.map((r) => r.hash)).toEqual(["b.json"]);
	});

	it("ignores scoping when the list is empty", () => {
		const results = searchToolIndex(rows, "curl", []);
		expect(results.map((r) => r.hash).sort()).toEqual(["a.json", "b.json", "c.json"]);
	});

	it("returns an empty array when no rows match any keyword", () => {
		expect(searchToolIndex(rows, "nonexistentkeyword")).toEqual([]);
	});

	it("returns an empty array for an empty index", () => {
		expect(searchToolIndex([], "curl")).toEqual([]);
	});
});
