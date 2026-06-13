import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { createRoundFilePath } from "../lib/hash.ts";
import type { RoundData } from "../lib/round-data.ts";
import { collectQueryInfo, runBuildGoldenPool } from "./build-golden-pool.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-build-golden-pool-test-"));
}

function line(value: unknown): string {
	return JSON.stringify(value);
}

describe("build-golden-pool script", () => {
	it("defaults sessions to <corpus>/sessions and writes .local pool plus worksheet with excerpts", async () => {
		const root = tmpDir();
		const corpusDir = path.join(root, "snapshot-a");
		const roundsDir = path.join(corpusDir, "rounds");
		const sessionsDir = path.join(corpusDir, "sessions", "--session-a");
		const baselineFile = path.join(root, "baseline-weak.json");
		const outFile = path.join(root, "docs", "eval", "golden-pool.local.json");
		const worksheetFile = path.join(root, "docs", "eval", "golden-worksheet.local.md");
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.mkdirSync(sessionsDir, { recursive: true });

		const older = {
			userPrompt:
				"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon with enough extra words to make the excerpt test meaningful here today",
			responseSequence: "older response contains extra context for the excerpt rendering path",
			turnIndex: 0,
			userTimestamp: 10,
			promptEmbedding: [1, 0, 0, 0],
		} satisfies RoundData;
		const olderFile = createRoundFilePath(older.userPrompt, older.responseSequence);
		const query = {
			userPrompt:
				"query prompt has enough words to replay retrieval against earlier rounds and produce a stable deterministic ranking for this script test case",
			responseSequence: "query response",
			turnIndex: 1,
			userTimestamp: 20,
			promptEmbedding: [1, 0, 0, 0],
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
		} satisfies RoundData;

		const queryFile = createRoundFilePath(query.userPrompt, query.responseSequence, query.toolCalls);
		fs.writeFileSync(path.join(roundsDir, olderFile), JSON.stringify(older, null, 2));
		fs.writeFileSync(path.join(roundsDir, queryFile), JSON.stringify(query, null, 2));
		fs.writeFileSync(
			path.join(roundsDir, "index.csv"),
			`${Buffer.from(JSON.stringify([1, 0, 0, 0])).toString("base64url")},${olderFile}:prompt\n${Buffer.from(JSON.stringify([1, 0, 0, 0])).toString("base64url")},${queryFile}:prompt\n`,
		);
		fs.writeFileSync(
			path.join(sessionsDir, "rounds.jsonl"),
			[
				line({
					type: "message",
					id: "u1",
					message: { role: "user", timestamp: 20, content: [{ type: "text", text: query.userPrompt }] },
				}),
				line({
					type: "message",
					message: {
						role: "assistant",
						content: [
							{ type: "toolCall", name: "get_round_details", arguments: { round: olderFile } },
							{ type: "text", text: query.responseSequence },
						],
					},
				}),
			].join("\n"),
		);
		fs.writeFileSync(
			baselineFile,
			JSON.stringify(
				{
					per_query: [
						{ query: queryFile, labels: [olderFile], top5: [{ file: olderFile, score: 1 }], first_hit_rank: 1 },
					],
				},
				null,
				2,
			),
		);

		await expect(
			runBuildGoldenPool({
				args: [
					"--corpus",
					corpusDir,
					"--baseline",
					baselineFile,
					"--out",
					outFile,
					"--worksheet",
					worksheetFile,
					"--count",
					"1",
				],
			}),
		).resolves.toBe(0);

		const pool = JSON.parse(fs.readFileSync(outFile, "utf-8"));
		expect(pool.kind).toBe("golden-pool");
		expect(pool.queries).toHaveLength(1);
		expect(pool.queries[0].query).toBe(queryFile);
		expect(pool.queries[0].candidates).toEqual([
			{
				file: olderFile,
				excerpt: expect.stringContaining("alpha beta gamma delta epsilon"),
			},
		]);
		const worksheet = fs.readFileSync(worksheetFile, "utf-8");
		expect(worksheet).toContain("# Golden worksheet");
		expect(worksheet).toContain(queryFile);
		expect(worksheet).toContain(`- [ ] ${olderFile}`);
		expect(worksheet).toContain("excerpt:");
		expect(worksheet).toContain("(primary)");
	});

	it("only collects query info from sessions that show semblr activity", () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		fs.mkdirSync(path.join(sessionsDir, "--session-active"), { recursive: true });
		fs.mkdirSync(path.join(sessionsDir, "--session-inactive"), { recursive: true });
		fs.writeFileSync(
			path.join(sessionsDir, "--session-active", "rounds.jsonl"),
			[
				line({
					type: "message",
					id: "u1",
					message: {
						role: "user",
						content: [
							{
								type: "text",
								text: "active prompt words words words words words words words words words words",
							},
						],
					},
				}),
				line({
					type: "message",
					message: {
						role: "assistant",
						content: [
							{ type: "toolCall", name: "get_round_details", arguments: { round: "older.json" } },
							{ type: "text", text: "assistant text" },
						],
					},
				}),
			].join("\n"),
		);
		fs.writeFileSync(
			path.join(sessionsDir, "--session-inactive", "rounds.jsonl"),
			[
				line({
					type: "message",
					id: "u1",
					message: {
						role: "user",
						content: [
							{
								type: "text",
								text: "inactive prompt words words words words words words words words words words",
							},
						],
					},
				}),
				line({
					type: "message",
					message: {
						role: "assistant",
						content: [{ type: "text", text: "plain assistant text without semblr markers" }],
					},
				}),
			].join("\n"),
		);

		const info = collectQueryInfo(sessionsDir);
		expect([...info.values()]).toHaveLength(2);
		expect([...info.values()].map((entry) => entry.semblrActive)).toEqual(expect.arrayContaining([true, false]));
	});
});
