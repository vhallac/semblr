import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runDigestAll } from "../scripts/digest-all.ts";
import { computeContentHash } from "./hash.ts";
import { encodeVectorIndexLine, loadShardVectorIndex } from "./index-io.ts";
import { ownShardPath } from "./index-shards.ts";
import {
	appendIndexRowToShards,
	buildSessionStartStatus,
	countShardLines,
	loadIndexFromShards,
} from "./index-storage.ts";
import { readRoundFileFromDir } from "./round-io.ts";
import { collectSearchRoundScores } from "./search-interactions.ts";
import { loadSemblrConfig } from "./semblr-config.ts";
import { buildCurrentModelRowPredicate } from "./session-backfill.ts";

/**
 * Two-host integration simulation (issue #149): one temp rounds directory
 * shared by two hosts, each with its own hostname. Every anchor below is an
 * acceptance criterion of the issue:
 *
 * - host B appends only to `index.<B>`; the other host's index artifact stays
 *   byte-identical;
 * - search returns rows from both shards;
 * - an empty rounds dir starts up and searches without error;
 * - a compaction (the `just index` sweep) on host A leaves host B's shard
 *   byte-identical;
 * - a round embedded by host B is not re-embedded by host A (D-010).
 */

const HOST_A = "host-a";
const HOST_B = "host-b";
const MODEL = "test-model";
const OLD_MODEL = "old-model";

/** A prompt above the 20-word threshold, so the sweep derives a `:prompt` row. */
const LONG_PROMPT =
	"Explain how the per-host index shard model keeps a synced multi-host rounds store free of write conflicts between two machines";
const LONG_RESPONSE =
	"The index is split into one file per host, each host appends only to its own shard, and every read merges the whole shard set.";

let tempDir = "";
let roundsDir = "";

beforeEach(() => {
	tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-two-host-"));
	roundsDir = path.join(tempDir, "rounds");
	fs.mkdirSync(roundsDir, { recursive: true });
});

afterEach(() => {
	fs.rmSync(tempDir, { recursive: true, force: true });
});

function shardPath(hostname: string): string {
	return ownShardPath(roundsDir, hostname);
}

function shardBytes(hostname: string): Buffer {
	return fs.readFileSync(shardPath(hostname));
}

function indexRow(filePath: string, vector: number[], model: string = MODEL): string {
	return encodeVectorIndexLine(vector, filePath, model);
}

function writeShard(hostname: string, rows: string[]): void {
	fs.writeFileSync(shardPath(hostname), rows.map((row) => `${row}\n`).join(""));
}

function writeLegacyIndex(rows: string[]): void {
	fs.writeFileSync(path.join(roundsDir, "index.csv"), rows.map((row) => `${row}\n`).join(""));
}

function writeRoundFile(fileName: string, userPrompt: string, responseSequence: string): void {
	fs.writeFileSync(
		path.join(roundsDir, fileName),
		JSON.stringify({ userPrompt, responseSequence, toolCalls: [], turnIndex: 0 }, null, 2),
	);
}

/** Parse a shard's rows into `{ filePath, model }` pairs. */
function shardRows(hostname: string): Array<{ filePath: string; model?: string }> {
	return fs
		.readFileSync(shardPath(hostname), "utf-8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => {
			const rest = line.slice(line.indexOf(",") + 1);
			const [filePath, model] = rest.split(",");
			return { filePath, model };
		});
}

/** Fake embedding endpoint: records every request, answers with queued vectors. */
function embeddingFetch(vectors: number[][], requests: unknown[] = []): typeof fetch {
	return vi.fn(async (input, init) => {
		requests.push({ input, body: JSON.parse(String(init?.body)) });
		return new Response(JSON.stringify({ data: [{ embedding: vectors.shift() ?? [0.5, 0.5] }] }), { status: 200 });
	}) as typeof fetch;
}

function sweepOptions(hostname: string, fetchImpl: typeof fetch) {
	return {
		config: loadSemblrConfig({
			agentDir: path.join(tempDir, "agent"),
			cwd: tempDir,
			env: {
				SEMBLR_ROUNDS_DIR: roundsDir,
				SEMBLR_EMBEDDING_MODEL: MODEL,
				// An explicit API URL bypasses the model-registry lookup, so the
				// test never depends on pi's installed model list.
				SEMBLR_EMBEDDING_API_URL: "https://embedding.test/v1/embeddings",
			},
		}),
		sessionsDir: path.join(tempDir, "sessions"),
		roundsDir,
		apiKey: "test-key",
		fetchImpl,
		hostname,
		stdout: { log: () => {} },
		stderr: { error: () => {} },
	};
}

describe("two-host index shards (issue #149)", () => {
	it("lands host B's append only in index.host-b and leaves host A's shard byte-identical", () => {
		appendIndexRowToShards(roundsDir, "a.json:prompt", [0.1, 0.2], { hostname: HOST_A }, MODEL);
		const hostABefore = shardBytes(HOST_A);

		appendIndexRowToShards(roundsDir, "b.json:prompt", [0.3, 0.4], { hostname: HOST_B }, MODEL);

		expect(shardBytes(HOST_A).equals(hostABefore)).toBe(true);
		const hostBRows = shardRows(HOST_B);
		expect(hostBRows).toHaveLength(1);
		expect(hostBRows[0].filePath).toBe("b.json:prompt");
	});

	it("adopts the legacy index.csv on host A's first write and leaves the adopted shard byte-identical for host B", () => {
		writeLegacyIndex([indexRow("legacy.json:prompt", [0.5, 0.5])]);

		appendIndexRowToShards(roundsDir, "a.json:prompt", [0.1, 0.2], { hostname: HOST_A }, MODEL);

		expect(fs.existsSync(path.join(roundsDir, "index.csv"))).toBe(false);
		const adoptedRows = shardRows(HOST_A).map((row) => row.filePath);
		expect(adoptedRows).toContain("legacy.json:prompt");
		expect(adoptedRows).toContain("a.json:prompt");

		const hostABefore = shardBytes(HOST_A);
		appendIndexRowToShards(roundsDir, "b.json:prompt", [0.3, 0.4], { hostname: HOST_B }, MODEL);
		expect(shardBytes(HOST_A).equals(hostABefore)).toBe(true);
	});

	it("does not claim a legacy index.csv when the host already owns a shard", () => {
		appendIndexRowToShards(roundsDir, "b.json:prompt", [0.1, 0.2], { hostname: HOST_B }, MODEL);
		writeLegacyIndex([indexRow("legacy.json:prompt", [0.5, 0.5])]);
		const legacyBefore = fs.readFileSync(path.join(roundsDir, "index.csv"));

		appendIndexRowToShards(roundsDir, "b2.json:prompt", [0.3, 0.4], { hostname: HOST_B }, MODEL);

		expect(fs.readFileSync(path.join(roundsDir, "index.csv")).equals(legacyBefore)).toBe(true);
		const hostBRows = shardRows(HOST_B).map((row) => row.filePath);
		expect(hostBRows).toContain("b.json:prompt");
		expect(hostBRows).toContain("b2.json:prompt");
	});

	it("searches the merged view and returns rounds from both shards", () => {
		writeRoundFile("a.json", "round hosted by host A", "answer from host A");
		writeRoundFile("b.json", "round hosted by host B", "answer from host B");
		writeShard(HOST_A, [indexRow("a.json:prompt", [1, 0])]);
		writeShard(HOST_B, [indexRow("b.json:prompt", [0.9, 0.1])]);

		const index = loadIndexFromShards(roundsDir);
		expect(index.map((entry) => entry.filePath).sort()).toEqual(["a.json:prompt", "b.json:prompt"]);

		const scores = collectSearchRoundScores(index, [1, 0], (filePath) => readRoundFileFromDir(filePath, roundsDir));
		expect(scores.map((score) => score.fileName).sort()).toEqual(["a.json", "b.json"]);
		expect(scores.every((score) => score.bestScore > 0)).toBe(true);
	});

	it("counts a round embedded on two hosts once (OI-4)", () => {
		writeRoundFile("round1.json", "a round both hosts embedded", "the shared answer");
		writeShard(HOST_A, [indexRow("round1.json:prompt", [1, 0])]);
		writeShard(HOST_B, [indexRow("round1.json:prompt", [0.99, 0.01])]);

		const index = loadIndexFromShards(roundsDir);
		expect(index.map((entry) => entry.filePath)).toEqual(["round1.json:prompt"]);
		expect(buildSessionStartStatus(index)).toContain("1 rounds indexed");
		// The raw merged read still sees both rows — the dedup is read-time only.
		expect(loadShardVectorIndex(roundsDir, { dedupe: false })).toHaveLength(2);
		// The shards on disk are untouched by the dedup.
		expect(shardRows(HOST_A)).toHaveLength(1);
		expect(shardRows(HOST_B)).toHaveLength(1);
	});

	it("starts up and searches an empty rounds dir without error", () => {
		expect(loadIndexFromShards(roundsDir)).toEqual([]);
		expect(countShardLines(roundsDir)).toBe(0);
		expect(buildSessionStartStatus(loadIndexFromShards(roundsDir))).toContain("0 rounds indexed");
		const scores = collectSearchRoundScores([], [1, 0], () => null);
		expect(scores).toEqual([]);
	});

	it("rewrites only host A's shard on a sweep and leaves host B's shard byte-identical", async () => {
		const roundFile = `${computeContentHash(LONG_PROMPT, LONG_RESPONSE)}.json`;
		writeRoundFile(roundFile, LONG_PROMPT, LONG_RESPONSE);
		// Host A's own shard holds stale rows for the round; the `:summary` row
		// is not reproduced by the reindex, so it is an orphan (D-007).
		writeShard(HOST_A, [
			indexRow(`${roundFile}:prompt`, [0.1, 0.2], OLD_MODEL),
			indexRow(`${roundFile}:summary`, [0.2, 0.3], OLD_MODEL),
		]);
		// Host B's shard holds a stale row for the same round: it is a foreign
		// shard and must survive host A's rewrite untouched (D-005).
		writeShard(HOST_B, [indexRow(`${roundFile}:prompt`, [0.3, 0.4], OLD_MODEL)]);
		const hostBBefore = shardBytes(HOST_B);
		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch(
			[
				[0.9, 0.1],
				[0.8, 0.2],
				[0.7, 0.3],
			],
			requests,
		);

		await runDigestAll(sweepOptions(HOST_A, fetchImpl));

		expect(requests.length).toBeGreaterThan(0);
		expect(shardBytes(HOST_B).equals(hostBBefore)).toBe(true);
		const hostARows = shardRows(HOST_A);
		const freshPromptRow = hostARows.find((row) => row.filePath === `${roundFile}:prompt`);
		expect(freshPromptRow?.model).toBe(MODEL);
		const orphanRow = hostARows.find((row) => row.filePath === `${roundFile}:summary`);
		expect(orphanRow?.model).toBe(OLD_MODEL);
	});

	it("does not re-embed a round host B already embedded (D-010)", async () => {
		const roundFile = `${computeContentHash(LONG_PROMPT, LONG_RESPONSE)}.json`;
		writeRoundFile(roundFile, LONG_PROMPT, LONG_RESPONSE);
		writeShard(HOST_B, [
			indexRow(`${roundFile}:prompt`, [0.9, 0.1], MODEL),
			indexRow(`${roundFile}:response`, [0.8, 0.2], MODEL),
		]);
		const hostBBefore = shardBytes(HOST_B);
		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch([], requests);

		await runDigestAll(sweepOptions(HOST_A, fetchImpl));

		expect(requests).toHaveLength(0);
		expect(shardBytes(HOST_B).equals(hostBBefore)).toBe(true);
		if (fs.existsSync(shardPath(HOST_A))) {
			expect(shardRows(HOST_A).map((row) => row.filePath)).not.toContain(`${roundFile}:prompt`);
		}
		const hasCurrentModelRows = buildCurrentModelRowPredicate(loadShardVectorIndex(roundsDir), MODEL);
		expect(hasCurrentModelRows(roundFile)).toBe(true);
	});
});
