import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { createRoundFilePath } from "../lib/hash.ts";
import type { RoundData } from "../lib/round-data.ts";
import { createDefaultStatsState } from "../lib/stats.ts";
import { runEvalRetrieval } from "./eval-retrieval.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-eval-retrieval-test-"));
}

function line(value: unknown): string {
	return JSON.stringify(value);
}

function prompt(words: string): string {
	return words;
}

function writeRound(roundsDir: string, round: RoundData): string {
	const roundFile = createRoundFilePath(round.userPrompt, round.responseSequence, round.toolCalls);
	fs.writeFileSync(path.join(roundsDir, roundFile), JSON.stringify(round, null, 2));
	return roundFile;
}

function encodeEntry(vector: number[], filePath: string): string {
	return `${Buffer.from(JSON.stringify(vector)).toString("base64url")},${filePath}`;
}

function writeSessionWithExpansion(
	sessionFile: string,
	queryPrompt: string,
	queryResponse: string,
	expandedRound: string,
): void {
	fs.mkdirSync(path.dirname(sessionFile), { recursive: true });
	fs.writeFileSync(
		sessionFile,
		[
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 40, content: [{ type: "text", text: queryPrompt }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "toolCall", name: "get_round_details", arguments: { round: expandedRound } },
						{ type: "text", text: queryResponse },
					],
				},
			}),
		].join("\n"),
	);
}

describe("eval-retrieval script", () => {
	it("replays retrieval over a synthetic corpus and produces deterministic weak-label metrics", async () => {
		const root = tmpDir();
		const corpusDir = path.join(root, "snapshot-a");
		const roundsDir = path.join(corpusDir, "rounds");
		const sessionsDir = path.join(root, "sessions");
		const outFile = path.join(root, "out", "eval.json");
		const secondOutFile = path.join(root, "out", "eval-second.json");
		fs.mkdirSync(roundsDir, { recursive: true });

		const olderA = {
			userPrompt: prompt(
				"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon",
			),
			responseSequence: "older A response",
			turnIndex: 0,
			userTimestamp: 10,
			promptEmbedding: [0.99, 0.1, 0, 0],
		} satisfies RoundData;
		const olderB = {
			userPrompt: prompt(
				"one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
			),
			responseSequence: "older B response",
			turnIndex: 1,
			userTimestamp: 20,
			promptEmbedding: [0.8, 0.2, 0, 0],
		} satisfies RoundData;
		const noLabels = {
			userPrompt: prompt(
				"red orange yellow green blue indigo violet black white gray silver gold bronze cyan magenta teal navy maroon olive lime",
			),
			responseSequence: "no labels response",
			turnIndex: 2,
			userTimestamp: 25,
			promptEmbedding: [0.2, 0.98, 0, 0],
		} satisfies RoundData;
		const shortRound = {
			userPrompt: "too short",
			responseSequence: "short response",
			turnIndex: 3,
			userTimestamp: 30,
			promptEmbedding: [1, 0, 0, 0],
		} satisfies RoundData;
		const noEmbedding = {
			userPrompt: prompt(
				"this prompt has enough words to pass the replay gate and still lacks any stored embedding for deterministic evaluation coverage today",
			),
			responseSequence: "missing embedding response",
			turnIndex: 4,
		} satisfies RoundData;
		const query = {
			userPrompt: prompt(
				"query prompt has enough words to replay retrieval against earlier rounds and produce a stable deterministic ranking for this script test case",
			),
			responseSequence: "query response",
			turnIndex: 5,
			userTimestamp: 40,
			promptEmbedding: [1, 0, 0, 0],
			parentId: "",
		} satisfies RoundData;
		const missingTimestamp = {
			userPrompt: prompt(
				"timestamp missing prompt still has enough words to pass the gate while verifying the skipped no timestamp path works correctly now",
			),
			responseSequence: "missing timestamp response",
			turnIndex: 6,
			promptEmbedding: [0, 0, 1, 0],
		} satisfies RoundData;
		const future = {
			userPrompt: prompt(
				"future round text has enough words to qualify and must still be excluded by the strict as of timestamp filter here",
			),
			responseSequence: "future response",
			turnIndex: 7,
			userTimestamp: 50,
			promptEmbedding: [1, 0, 0, 0],
		} satisfies RoundData;

		const olderAFile = writeRound(roundsDir, olderA);
		const olderBFile = writeRound(roundsDir, olderB);
		writeRound(roundsDir, noLabels);
		writeRound(roundsDir, shortRound);
		writeRound(roundsDir, noEmbedding);
		const queryFile = writeRound(roundsDir, {
			...query,
			parentId: olderAFile,
			toolCallCount: 1,
			toolCallNames: ["get_round_details"],
			toolCalls: [
				{
					index: 0,
					name: "get_round_details",
					arguments: JSON.stringify({ round: olderBFile }),
					result_summary: "",
				},
			],
		});
		writeRound(roundsDir, missingTimestamp);
		const futureFile = writeRound(roundsDir, future);

		fs.writeFileSync(
			path.join(roundsDir, "index.csv"),
			`${[
				encodeEntry([0.99, 0.1, 0, 0], `${olderAFile}:prompt`),
				encodeEntry([0.8, 0.2, 0, 0], `${olderBFile}:prompt`),
				encodeEntry(
					[0.2, 0.98, 0, 0],
					`${createRoundFilePath(noLabels.userPrompt, noLabels.responseSequence)}:prompt`,
				),
				encodeEntry([1, 0, 0, 0], `${queryFile}:prompt`),
				encodeEntry(
					[0, 0, 1, 0],
					`${createRoundFilePath(missingTimestamp.userPrompt, missingTimestamp.responseSequence)}:prompt`,
				),
				encodeEntry([1, 0, 0, 0], `${futureFile}:prompt`),
			].join("\n")}
`,
		);

		const stats = createDefaultStatsState("2026-06-12T03:01:09.000Z", 3);
		stats.positionScores[0].presentedCount = 4;
		stats.positionScores[0].readCount = 1;
		stats.positionScores[1].presentedCount = 2;
		stats.positionScores[1].readCount = 1;
		fs.writeFileSync(path.join(corpusDir, "chain-read-stats.json"), JSON.stringify(stats, null, 2));

		writeSessionWithExpansion(
			path.join(sessionsDir, "--session-a", "rounds.jsonl"),
			query.userPrompt,
			query.responseSequence,
			olderBFile,
		);

		await expect(
			runEvalRetrieval({
				corpusDir,
				sessionsDir,
				outFile,
				args: ["--corpus", corpusDir, "--sessions", sessionsDir, "--out", outFile],
				gitRev: "test-rev",
			}),
		).resolves.toBe(0);

		const first = JSON.parse(fs.readFileSync(outFile, "utf-8"));
		expect(first).toEqual({
			kind: "weak",
			config_hash: expect.any(String),
			corpus: "snapshot-a",
			counts: {
				total_rounds: 8,
				replayed: 5,
				skipped_short: 1,
				skipped_no_embedding: 1,
				skipped_no_timestamp: 1,
				skipped_no_labels: 4,
			},
			hit_at_5: 1,
			recall_at_5: 1,
			mrr: 1,
			noise_rate: 1 - 2 / 6,
			per_query: [
				{
					query: queryFile,
					labels: [olderAFile, olderBFile],
					top5: [
						{ file: olderAFile, score: 0.99 },
						{ file: olderBFile, score: 0.8 },
						{ file: createRoundFilePath(noLabels.userPrompt, noLabels.responseSequence), score: 0.2 },
					],
					first_hit_rank: 1,
				},
			],
		});

		await expect(
			runEvalRetrieval({
				corpusDir,
				sessionsDir,
				outFile,
				args: ["--corpus", corpusDir, "--sessions", sessionsDir, "--out", outFile],
				gitRev: "test-rev",
			}),
		).resolves.toBe(0);
		const second = JSON.parse(fs.readFileSync(outFile, "utf-8"));
		expect(second).toEqual(first);

		await expect(
			runEvalRetrieval({
				corpusDir,
				sessionsDir,
				outFile: secondOutFile,
				args: ["--corpus", corpusDir, "--sessions", sessionsDir, "--out", secondOutFile],
				gitRev: "test-rev",
			}),
		).resolves.toBe(0);
		const third = JSON.parse(fs.readFileSync(secondOutFile, "utf-8"));
		expect(third).toEqual(first);
	});

	it("uses primary for MRR and labels for Recall@5 when --golden is provided", async () => {
		const root = tmpDir();
		const corpusDir = path.join(root, "snapshot-a");
		const roundsDir = path.join(corpusDir, "rounds");
		const sessionsDir = path.join(corpusDir, "sessions");
		const outFile = path.join(root, "out", "eval-golden.json");
		const goldenFile = path.join(root, "docs", "eval", "golden-labels.json");
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.mkdirSync(sessionsDir, { recursive: true });

		const primaryRound = {
			userPrompt: prompt(
				"primary candidate prompt has enough words to stay in the corpus and act as the golden primary result for this test",
			),
			responseSequence: "primary response",
			turnIndex: 0,
			userTimestamp: 10,
			promptEmbedding: [0.6, 0.8, 0, 0],
		} satisfies RoundData;
		const relevantRound = {
			userPrompt: prompt(
				"relevant candidate prompt has enough words to stay in the corpus and rank ahead of the primary in this test",
			),
			responseSequence: "relevant response",
			turnIndex: 1,
			userTimestamp: 20,
			promptEmbedding: [0.9, 0.9, 0, 0],
		} satisfies RoundData;
		const queryRound = {
			userPrompt: prompt(
				"golden query prompt has enough words to produce deterministic retrieval ordering for the primary and supporting labels in this evaluation",
			),
			responseSequence: "golden query response",
			turnIndex: 2,
			userTimestamp: 30,
			promptEmbedding: [1, 1, 0, 0],
		} satisfies RoundData;

		const primaryFile = writeRound(roundsDir, primaryRound);
		const relevantFile = writeRound(roundsDir, relevantRound);
		const queryFile = writeRound(roundsDir, queryRound);
		fs.writeFileSync(
			path.join(roundsDir, "index.csv"),
			`${[
				encodeEntry([0.6, 0.8, 0, 0], `${primaryFile}:prompt`),
				encodeEntry([0.9, 0.9, 0, 0], `${relevantFile}:prompt`),
				encodeEntry([1, 1, 0, 0], `${queryFile}:prompt`),
			].join("\n")}
`,
		);
		fs.writeFileSync(
			path.join(corpusDir, "chain-read-stats.json"),
			JSON.stringify(createDefaultStatsState("2026-06-12T03:01:09.000Z"), null, 2),
		);
		fs.mkdirSync(path.dirname(goldenFile), { recursive: true });
		fs.writeFileSync(
			goldenFile,
			JSON.stringify(
				{
					kind: "golden-labels",
					version: 1,
					source_pool: "/tmp/golden-pool.local.json",
					queries: [
						{
							query: queryFile,
							prompt: queryRound.userPrompt,
							difficulty: "hard",
							primary: primaryFile,
							labels: [primaryFile, relevantFile],
						},
					],
				},
				null,
				2,
			),
		);

		await expect(
			runEvalRetrieval({
				args: ["--corpus", corpusDir, "--out", outFile, "--golden", goldenFile],
				outFile,
				gitRev: "test-rev",
			}),
		).resolves.toBe(0);
		const result = JSON.parse(fs.readFileSync(outFile, "utf-8"));
		expect(result.kind).toBe("golden");
		expect(result.recall_at_5).toBe(1);
		expect(result.mrr).toBe(0.5);
		expect(result.per_query).toEqual([
			{
				query: queryFile,
				labels: [primaryFile, relevantFile],
				top5: [
					{ file: relevantFile, score: 1.8 },
					{ file: primaryFile, score: 1.4 },
				],
				first_hit_rank: 2,
			},
		]);
	});

	it("routes golden queries by mode and keeps top-level metrics similarity-only", async () => {
		const root = tmpDir();
		const corpusDir = path.join(root, "snapshot-a");
		const roundsDir = path.join(corpusDir, "rounds");
		const sessionsDir = path.join(corpusDir, "sessions");
		const outFile = path.join(root, "out", "eval-golden.json");
		const toolOnlyOutFile = path.join(root, "out", "eval-tool.json");
		const goldenFile = path.join(root, "docs", "eval", "golden-labels.json");
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.mkdirSync(sessionsDir, { recursive: true });

		const similarityTarget = {
			userPrompt: prompt(
				"similarity target has enough words to qualify as an earlier candidate in this golden retrieval evaluation test with stable deterministic ranking today",
			),
			responseSequence: "similarity target response",
			turnIndex: 0,
			userTimestamp: 10,
			promptEmbedding: [1, 0],
		} satisfies RoundData;
		const toolTarget = {
			userPrompt: prompt(
				"tool target has enough words to qualify as an earlier candidate in this golden retrieval evaluation test with stable deterministic ranking today",
			),
			responseSequence: "tool target response",
			turnIndex: 1,
			userTimestamp: 20,
			promptEmbedding: [0, 1],
		} satisfies RoundData;
		const similarityQuery = {
			userPrompt: prompt(
				"similarity query has enough words to replay the existing semantic retrieval behavior without a mode field and preserve the historical baseline output today",
			),
			responseSequence: "similarity query response",
			turnIndex: 2,
			userTimestamp: 30,
			promptEmbedding: [1, 0],
		} satisfies RoundData;
		const toolQuery = {
			userPrompt: prompt(
				"tool query has enough words but deliberately uses an unrelated embedding because explicit tool query text drives retrieval in this golden evaluation case",
			),
			responseSequence: "tool query response",
			turnIndex: 3,
			userTimestamp: 40,
			promptEmbedding: [1, 0],
		} satisfies RoundData;
		const futureToolTarget = {
			userPrompt: prompt(
				"future tool target has enough words and must be excluded by strict timestamp filtering in tool mode during this deterministic golden evaluation case",
			),
			responseSequence: "future tool target response",
			turnIndex: 4,
			userTimestamp: 50,
			promptEmbedding: [0, 1],
		} satisfies RoundData;

		const similarityTargetFile = writeRound(roundsDir, similarityTarget);
		const toolTargetFile = writeRound(roundsDir, toolTarget);
		const similarityQueryFile = writeRound(roundsDir, similarityQuery);
		const toolQueryFile = writeRound(roundsDir, toolQuery);
		const futureToolTargetFile = writeRound(roundsDir, futureToolTarget);
		fs.writeFileSync(
			path.join(roundsDir, "index.csv"),
			`${[
				encodeEntry([1, 0], `${similarityTargetFile}:prompt`),
				encodeEntry([0, 1], `${toolTargetFile}:prompt`),
				encodeEntry([1, 0], `${similarityQueryFile}:prompt`),
				encodeEntry([1, 0], `${toolQueryFile}:prompt`),
				encodeEntry([0, 1], `${futureToolTargetFile}:prompt`),
			].join("\n")}\n`,
		);
		fs.writeFileSync(
			path.join(roundsDir, "index-tools.fulltext.csv"),
			[`${toolTargetFile},0,bash,bash ssh prod-db-2`, `${futureToolTargetFile},0,bash,bash ssh prod-db-2`].join(
				"\n",
			),
		);
		fs.writeFileSync(
			path.join(corpusDir, "chain-read-stats.json"),
			JSON.stringify(createDefaultStatsState("2026-06-12T03:01:09.000Z"), null, 2),
		);
		fs.mkdirSync(path.dirname(goldenFile), { recursive: true });
		fs.writeFileSync(
			goldenFile,
			JSON.stringify({
				kind: "golden-labels",
				version: 1,
				source_pool: "/tmp/golden-pool.local.json",
				queries: [
					{
						query: similarityQueryFile,
						prompt: similarityQuery.userPrompt,
						difficulty: "control",
						primary: similarityTargetFile,
						labels: [similarityTargetFile],
					},
					{
						query: toolQueryFile,
						prompt: toolQuery.userPrompt,
						mode: "tool",
						tool_query: "ssh prod-db-2",
						difficulty: "tool",
						primary: toolTargetFile,
						labels: [toolTargetFile],
					},
				],
			}),
		);

		await expect(
			runEvalRetrieval({
				args: ["--corpus", corpusDir, "--out", outFile, "--golden", goldenFile],
				outFile,
				gitRev: "test-rev",
			}),
		).resolves.toBe(0);
		const result = JSON.parse(fs.readFileSync(outFile, "utf-8"));
		expect(result.per_query).toHaveLength(1);
		expect(result.per_query[0].query).toBe(similarityQueryFile);
		expect(result.hit_at_5).toBe(1);
		expect(result.by_mode.similarity.per_query).toEqual(result.per_query);
		expect(result.by_mode.tool).toMatchObject({ hit_at_5: 1, recall_at_5: 1, mrr: 1 });
		expect(result.by_mode.tool.per_query[0]).toEqual({
			query: toolQueryFile,
			labels: [toolTargetFile],
			top5: [{ file: toolTargetFile, score: 1 }],
			first_hit_rank: 1,
		});

		await expect(
			runEvalRetrieval({
				args: ["--corpus", corpusDir, "--out", toolOnlyOutFile, "--golden", goldenFile, "--mode", "tool"],
				outFile: toolOnlyOutFile,
				gitRev: "test-rev",
			}),
		).resolves.toBe(0);
		const toolOnly = JSON.parse(fs.readFileSync(toolOnlyOutFile, "utf-8"));
		expect(Object.keys(toolOnly.by_mode)).toEqual(["tool"]);
		expect(toolOnly.per_query).toEqual([]);
		expect(toolOnly.hit_at_5).toBe(0);
	});

	it("rejects an invalid --mode value", async () => {
		const stderr: string[] = [];

		await expect(
			runEvalRetrieval({
				args: ["--corpus", "/tmp/unused", "--mode", "bogus"],
				stderr: { error: (message: string) => stderr.push(message) },
			}),
		).resolves.toBe(1);
		expect(stderr[0]).toContain("Invalid --mode: bogus");
	});

	it("rejects tool-mode golden labels when the tool index is missing", async () => {
		const root = tmpDir();
		const corpusDir = path.join(root, "snapshot-a");
		const roundsDir = path.join(corpusDir, "rounds");
		const goldenFile = path.join(root, "golden.json");
		const stderr: string[] = [];
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(path.join(roundsDir, "index.csv"), "");
		fs.writeFileSync(
			goldenFile,
			JSON.stringify({
				kind: "golden-labels",
				version: 1,
				source_pool: "test",
				queries: [
					{
						query: "query.json",
						prompt: "tool query",
						mode: "tool",
						tool_query: "ssh prod-db-2",
						difficulty: "tool",
						primary: "target.json",
						labels: ["target.json"],
					},
				],
			}),
		);

		await expect(
			runEvalRetrieval({
				args: ["--corpus", corpusDir, "--golden", goldenFile],
				gitRev: "test-rev",
				stderr: { error: (message: string) => stderr.push(message) },
			}),
		).resolves.toBe(1);
		expect(stderr[0]).toBe(`Corpus tool index does not exist: ${path.join(roundsDir, "index-tools.fulltext.csv")}`);
	});

	it("rejects a tool-mode golden query without tool_query", async () => {
		const root = tmpDir();
		const corpusDir = path.join(root, "snapshot-a");
		const roundsDir = path.join(corpusDir, "rounds");
		const goldenFile = path.join(root, "golden.json");
		fs.mkdirSync(roundsDir, { recursive: true });
		const query = {
			userPrompt: prompt(
				"tool query has enough words to qualify but lacks the required explicit tool query text for matching in this deterministic golden evaluation test case",
			),
			responseSequence: "query response",
			turnIndex: 0,
			userTimestamp: 20,
			promptEmbedding: [1, 0],
		} satisfies RoundData;
		const queryFile = writeRound(roundsDir, query);
		fs.writeFileSync(path.join(roundsDir, "index.csv"), `${encodeEntry([1, 0], `${queryFile}:prompt`)}\n`);
		fs.writeFileSync(path.join(roundsDir, "index-tools.fulltext.csv"), "");
		fs.writeFileSync(
			goldenFile,
			JSON.stringify({
				kind: "golden-labels",
				version: 1,
				source_pool: "test",
				queries: [
					{
						query: queryFile,
						prompt: query.userPrompt,
						mode: "tool",
						difficulty: "tool",
						primary: "target",
						labels: ["target"],
					},
				],
			}),
		);
		await expect(
			runEvalRetrieval({ args: ["--corpus", corpusDir, "--golden", goldenFile], gitRev: "test-rev" }),
		).rejects.toThrow(`Tool-mode golden query ${queryFile} is missing tool_query`);
	});

	it("defaults --sessions to <corpus>/sessions when omitted", async () => {
		const root = tmpDir();
		const corpusDir = path.join(root, "snapshot-a");
		const roundsDir = path.join(corpusDir, "rounds");
		const sessionsDir = path.join(corpusDir, "sessions");
		const outFile = path.join(root, "out", "eval.json");
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.mkdirSync(path.join(sessionsDir, "--session-a"), { recursive: true });

		const older = {
			userPrompt: prompt(
				"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon",
			),
			responseSequence: "older response",
			turnIndex: 0,
			userTimestamp: 10,
			promptEmbedding: [1, 0, 0, 0],
		} satisfies RoundData;
		const query = {
			userPrompt: prompt(
				"query prompt has enough words to replay retrieval against earlier rounds and produce a stable deterministic ranking for this script test case",
			),
			responseSequence: "query response",
			turnIndex: 1,
			userTimestamp: 20,
			promptEmbedding: [1, 0, 0, 0],
			parentId: "",
		} satisfies RoundData;

		const olderFile = writeRound(roundsDir, older);
		const queryFile = writeRound(roundsDir, {
			...query,
			parentId: olderFile,
			toolCallCount: 1,
			toolCallNames: ["get_round_details"],
			toolCalls: [
				{
					index: 0,
					name: "get_round_details",
					arguments: JSON.stringify({ round: olderFile }),
					result_summary: "",
				},
			],
		});

		fs.writeFileSync(
			path.join(roundsDir, "index.csv"),
			`${[encodeEntry([1, 0, 0, 0], `${olderFile}:prompt`), encodeEntry([1, 0, 0, 0], `${queryFile}:prompt`)].join("\n")}\n`,
		);
		fs.writeFileSync(
			path.join(corpusDir, "chain-read-stats.json"),
			JSON.stringify(createDefaultStatsState("2026-06-12T03:01:09.000Z"), null, 2),
		);
		writeSessionWithExpansion(
			path.join(sessionsDir, "--session-a", "rounds.jsonl"),
			query.userPrompt,
			query.responseSequence,
			olderFile,
		);

		await expect(
			runEvalRetrieval({ args: ["--corpus", corpusDir, "--out", outFile], outFile, gitRev: "test-rev" }),
		).resolves.toBe(0);
		const result = JSON.parse(fs.readFileSync(outFile, "utf-8"));
		expect(result.per_query).toEqual([
			{
				query: queryFile,
				labels: [olderFile],
				top5: [{ file: olderFile, score: 1 }],
				first_hit_rank: 1,
			},
		]);
	});

	it("requires --corpus", async () => {
		const stderr: string[] = [];
		await expect(
			runEvalRetrieval({ args: [], stderr: { error: (line: string) => stderr.push(line) } }),
		).resolves.toBe(1);
		expect(stderr).toEqual([
			"Usage: npx tsx scripts/eval-retrieval.ts --corpus <dir> [--sessions <dir>] [--out <file>] [--golden <file>] [--mode <similarity|tool>]",
		]);
	});
});
