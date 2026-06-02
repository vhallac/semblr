import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	createDefaultStatsState,
	flushStatsFile,
	formatChainReadStatsReport,
	loadStatsFile,
	normalizeStatsState,
	recordPresented,
	recordRead,
} from "./stats.ts";

let tempDir = "";

beforeEach(() => {
	tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-stats-"));
});

afterEach(() => {
	fs.rmSync(tempDir, { recursive: true, force: true });
});

const statsPath = () => path.join(tempDir, "chain-read-stats.json");

describe("chain read stats", () => {
	it("creates default v2 stats", () => {
		const stats = createDefaultStatsState("now");

		expect(stats.version).toBe(2);
		expect(stats.lastUpdated).toBe("now");
		expect(stats.positionScores).toHaveLength(5);
		expect(stats.positionScores[0]).toEqual({ presentedCount: 0, readCount: 0, presentedHash: null });
	});

	it("migrates v1 position5 stats to v2", () => {
		const stats = normalizeStatsState({
			version: 1,
			positionScores: [],
			position5: { presentedCount: 9, readCount: 3 },
		});

		expect(stats.version).toBe(2);
		expect(stats.position5).toBeUndefined();
		expect(stats.positionScores[4]).toEqual({ presentedCount: 9, readCount: 3, presentedHash: null });
	});

	it("loads missing, valid, and corrupt stats files", () => {
		expect(loadStatsFile(statsPath(), "missing").lastUpdated).toBe("missing");

		fs.writeFileSync(statsPath(), JSON.stringify({ version: 2, positionScores: [] }));
		expect(loadStatsFile(statsPath()).positionScores).toEqual([]);

		fs.writeFileSync(statsPath(), "not json");
		expect(loadStatsFile(statsPath(), "fallback").lastUpdated).toBe("fallback");
	});

	it("flushes stats atomically and removes temp file after write failure", () => {
		const stats = createDefaultStatsState("old");
		flushStatsFile(stats, statsPath(), tempDir, "new", 123);

		expect(JSON.parse(fs.readFileSync(statsPath(), "utf-8"))).toMatchObject({ lastUpdated: "new" });
		expect(fs.existsSync(`${statsPath()}.tmp.123`)).toBe(false);

		const fileAsDir = path.join(tempDir, "not-a-dir");
		fs.writeFileSync(fileAsDir, "x");
		flushStatsFile(stats, path.join(fileAsDir, "stats.json"), fileAsDir, "ignored", 456);
		expect(fs.existsSync(path.join(fileAsDir, "stats.json.tmp.456"))).toBe(false);
	});

	it("records presented hashes newest first and reads matching positions", () => {
		const stats = createDefaultStatsState("now");
		const presented = [null, null, null, null, null];

		recordPresented(stats, presented, [{ fileName: "old" }, { fileName: "mid" }, { fileName: "new" }]);
		recordRead(stats, presented, "new");
		recordRead(stats, presented, "old");
		recordRead(stats, presented, "missing");

		expect(presented).toEqual(["new", "mid", "old", null, null]);
		expect(stats.positionScores.map((p) => p.presentedCount)).toEqual([1, 1, 1, 0, 0]);
		expect(stats.positionScores.map((p) => p.readCount)).toEqual([1, 0, 1, 0, 0]);
	});

	it("formats chain read report with percentages and empty positions", () => {
		const stats = createDefaultStatsState("now");
		stats.positionScores[0] = { presentedCount: 4, readCount: 2, presentedHash: null };

		expect(formatChainReadStatsReport(stats)).toContain("- immediate parent: 50% (2/4)");
		expect(formatChainReadStatsReport(stats)).toContain("- 2nd: — (0/0)");
	});
});
