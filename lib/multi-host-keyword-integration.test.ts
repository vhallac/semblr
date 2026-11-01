import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runRebuildBm25 } from "../scripts/rebuild-bm25.ts";
import {
	type Bm25WriteFs,
	buildBm25Index,
	loadBm25Index,
	ownBm25IndexPath,
	scoreBm25Query,
	writeBm25Index,
} from "./bm25-index.ts";
import {
	buildToolIndexRows,
	buildToolIndexRowsFromRoundsDir,
	loadToolIndex,
	ownToolIndexPath,
	searchToolIndex,
	type ToolIndexWriteFs,
	writeToolIndexRows,
} from "./search-tools.ts";

/**
 * Two-host integration simulation for the keyword artifacts (issue #152):
 * one temp rounds directory shared by two hosts, each with its own host id.
 * Every anchor below is an acceptance scenario of the issue's feature file:
 *
 * - host B's rebuild leaves host A's BM25 and tool-index artifacts
 *   byte-identical;
 * - a foreign host's keyword artifacts are never read as the local index
 *   (an absent own artifact behaves as an empty index, not an error);
 * - an interrupted rewrite leaves the previous artifact byte-identical and
 *   the next run rebuilds it from the round files.
 */

const HOST_A = "host-a";
const HOST_B = "host-b";

const TOOL_CALL_A = {
	index: 0,
	name: "get_round_details",
	arguments: JSON.stringify({ round: "a.json" }),
	result_summary: "returned round a.json",
};

const TOOL_CALL_B = {
	index: 0,
	name: "search_interactions",
	arguments: JSON.stringify({ query: "basaltcrane" }),
	result_summary: "found the basaltcrane round",
};

let tempDir = "";
let roundsDir = "";

beforeEach(() => {
	tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-two-host-keyword-"));
	roundsDir = path.join(tempDir, "rounds");
	fs.mkdirSync(roundsDir, { recursive: true });
});

afterEach(() => {
	fs.rmSync(tempDir, { recursive: true, force: true });
});

function writeRoundFile(
	fileName: string,
	userPrompt: string,
	responseSequence: string,
	toolCalls: Array<typeof TOOL_CALL_A> = [],
): void {
	fs.writeFileSync(
		path.join(roundsDir, fileName),
		JSON.stringify({ userPrompt, responseSequence, toolCalls, turnIndex: 0 }, null, 2),
	);
}

function ownBm25(hostname: string): string {
	return ownBm25IndexPath(roundsDir, hostname);
}

function ownTools(hostname: string): string {
	return ownToolIndexPath(roundsDir, hostname);
}

const silentStdout = { log: () => {} };

describe("two-host keyword artifacts (issue #152)", () => {
	it("host B's rebuild leaves host A's keyword artifacts byte-identical", async () => {
		writeRoundFile("a.json", "The round carries the aetherwave marker", "host A answered with aetherwave", [
			TOOL_CALL_A,
		]);
		writeRoundFile("b.json", "The round carries the basaltcrane marker", "host B answered with basaltcrane", [
			TOOL_CALL_B,
		]);

		// Host A establishes its own artifacts (BM25 flush + tool-index rows).
		writeBm25Index(ownBm25(HOST_A), buildBm25Index([{ fileName: "a.json", text: "host A round about aetherwave" }]));
		writeToolIndexRows(ownTools(HOST_A), buildToolIndexRows("a.json", [TOOL_CALL_A]));
		const bm25ABefore = fs.readFileSync(ownBm25(HOST_A));
		const toolsABefore = fs.readFileSync(ownTools(HOST_A));

		// Host B rebuilds both keyword artifacts from the locally visible round
		// files — exactly what `just rebuild-bm25` and `just digest-all --tools-only`
		// do on the second machine.
		await expect(runRebuildBm25({ roundsDir, hostname: HOST_B, stdout: silentStdout })).resolves.toBe(0);
		writeToolIndexRows(ownTools(HOST_B), buildToolIndexRowsFromRoundsDir(roundsDir));

		// Host A's artifacts are byte-identical: host B's rebuild never touched them.
		expect(fs.readFileSync(ownBm25(HOST_A)).equals(bm25ABefore)).toBe(true);
		expect(fs.readFileSync(ownTools(HOST_A)).equals(toolsABefore)).toBe(true);

		// Host B's artifacts are separate files and cover the rounds it can see.
		expect(fs.existsSync(ownBm25(HOST_B))).toBe(true);
		expect(fs.existsSync(ownTools(HOST_B))).toBe(true);
		const bm25B = loadBm25Index(ownBm25(HOST_B));
		expect(scoreBm25Query(bm25B, "aetherwave").get("a.json")).toBeGreaterThan(0);
		expect(scoreBm25Query(bm25B, "basaltcrane").get("b.json")).toBeGreaterThan(0);
		// The tool-index rebuild covers every locally visible round (a.json and
		// b.json are both in the shared dir) — but lands only in host B's artifact.
		const toolRowsB = loadToolIndex(ownTools(HOST_B));
		expect(toolRowsB.map((row) => row.hash).sort()).toEqual(["a.json", "b.json"]);
	});

	it("never reads a foreign host's keyword artifacts as the local index", async () => {
		writeRoundFile("b.json", "The round carries the basaltcrane marker", "host B answered with basaltcrane");

		// Foreign host A artifacts hold rows for a round host B has never embedded.
		writeBm25Index(ownBm25(HOST_A), buildBm25Index([{ fileName: "a.json", text: "foreign round about aetherwave" }]));
		writeToolIndexRows(ownTools(HOST_A), buildToolIndexRows("a.json", [TOOL_CALL_A]));
		const bm25ABefore = fs.readFileSync(ownBm25(HOST_A));
		const toolsABefore = fs.readFileSync(ownTools(HOST_A));

		// Host B has no own artifacts yet: absent behaves as an empty index,
		// not an error, and the foreign rows are invisible.
		const bm25B = loadBm25Index(ownBm25(HOST_B));
		expect(bm25B.documentCount).toBe(0);
		expect([...scoreBm25Query(bm25B, "aetherwave")]).toEqual([]);
		expect(loadToolIndex(ownTools(HOST_B))).toEqual([]);
		expect(searchToolIndex(loadToolIndex(ownTools(HOST_B)), "get_round_details")).toEqual([]);

		// Host B's rebuild builds from the rounds it can see — not the foreign
		// document set — and leaves the foreign artifacts untouched.
		await expect(runRebuildBm25({ roundsDir, hostname: HOST_B, stdout: silentStdout })).resolves.toBe(0);
		const rebuilt = loadBm25Index(ownBm25(HOST_B));
		expect(Object.keys(rebuilt.documents)).toEqual(["b.json"]);
		expect(scoreBm25Query(rebuilt, "basaltcrane").get("b.json")).toBeGreaterThan(0);
		expect(fs.readFileSync(ownBm25(HOST_A)).equals(bm25ABefore)).toBe(true);
		expect(fs.readFileSync(ownTools(HOST_A)).equals(toolsABefore)).toBe(true);
	});

	it("an interrupted rewrite leaves the previous artifact byte-identical and the next run rebuilds it", async () => {
		writeRoundFile("b.json", "The round carries the basaltcrane marker", "host B answered with basaltcrane", [
			TOOL_CALL_B,
		]);
		await expect(runRebuildBm25({ roundsDir, hostname: HOST_B, stdout: silentStdout })).resolves.toBe(0);
		writeToolIndexRows(ownTools(HOST_B), buildToolIndexRows("b.json", [TOOL_CALL_B]));
		const bm25Before = fs.readFileSync(ownBm25(HOST_B));
		const toolsBefore = fs.readFileSync(ownTools(HOST_B));

		// Simulate a crash mid-rewrite: the tmp staging write throws before the
		// rename can publish anything (decision 015).
		const interruptedFs = {
			existsSync: fs.existsSync,
			mkdirSync: fs.mkdirSync,
			writeFileSync: () => {
				throw new Error("interrupted");
			},
			renameSync: fs.renameSync,
		} satisfies Bm25WriteFs & ToolIndexWriteFs;

		expect(() =>
			writeBm25Index(
				ownBm25(HOST_B),
				buildBm25Index([{ fileName: "b.json", text: "updated aetherwave" }]),
				interruptedFs,
			),
		).toThrow("interrupted");
		expect(() =>
			writeToolIndexRows(ownTools(HOST_B), buildToolIndexRows("b.json", [TOOL_CALL_B]), interruptedFs),
		).toThrow("interrupted");

		// Both previous artifacts are byte-identical.
		expect(fs.readFileSync(ownBm25(HOST_B)).equals(bm25Before)).toBe(true);
		expect(fs.readFileSync(ownTools(HOST_B)).equals(toolsBefore)).toBe(true);

		// The next run rebuilds the artifacts from the round files.
		await expect(runRebuildBm25({ roundsDir, hostname: HOST_B, stdout: silentStdout })).resolves.toBe(0);
		writeToolIndexRows(ownTools(HOST_B), buildToolIndexRowsFromRoundsDir(roundsDir));
		expect(scoreBm25Query(loadBm25Index(ownBm25(HOST_B)), "basaltcrane").get("b.json")).toBeGreaterThan(0);
		expect(loadToolIndex(ownTools(HOST_B)).map((row) => row.hash)).toEqual(["b.json"]);
	});
});
