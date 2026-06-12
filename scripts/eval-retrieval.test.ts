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
			userTimestamp: 35,
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

	it("requires --corpus", async () => {
		const stderr: string[] = [];
		await expect(
			runEvalRetrieval({ args: [], stderr: { error: (line: string) => stderr.push(line) } }),
		).resolves.toBe(1);
		expect(stderr).toEqual([
			"Usage: npx tsx scripts/eval-retrieval.ts --corpus <dir> [--sessions <dir>] [--out <file>]",
		]);
	});
});
