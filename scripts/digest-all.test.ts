import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { bm25IndexPathForRoundsDir } from "../lib/bm25-index.ts";
import { computeContentHash } from "../lib/hash.ts";
import { encodeVectorIndexLine, loadVectorIndex, readIndexLines } from "../lib/index-io.ts";
import { hashEmbeddingInput } from "../lib/round-capture.ts";
import { loadToolIndex, toolIndexPathForRoundsDir } from "../lib/search-tools.ts";
import { SEMBLR_CONFIG_DEFAULTS } from "../lib/semblr-config.ts";
import { embedRecoveredRounds } from "../lib/session-backfill.ts";
import { isMainModule, runDigestAll } from "./digest-all.ts";

// The short-prompt drop is shared-core policy (lib/embed-round.ts, covered in
// lib/embed-round.test.ts and lib/session-backfill.test.ts). The redrive tests
// below target clip, row, and marker mechanics, so keep fixture prompts
// embeddable unless a test opts into the documented default threshold.
process.env.RELEVANCE_LIST_MIN_WORDS = "1";

// The model a reindex stamps onto fresh rows (and, after the F1 fix, onto a
// preserved orphan row) when no explicit model is configured.
const DEBUG_EMBEDDING_MODEL = SEMBLR_CONFIG_DEFAULTS.embeddingModel;

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-digest-all-test-"));
}

function logger() {
	const stdout: string[] = [];
	const stderr: string[] = [];
	return {
		stdout,
		stderr,
		out: { log: (line: string) => stdout.push(line) },
		err: { error: (line: string) => stderr.push(line) },
	};
}

/**
 * Real filesystem with `writeFileSync`/`renameSync`/`readFileSync` recorded, so
 * a test can assert write ordering/atomicity and whole-store read counts while
 * every other call passes through.
 */
function spyFs(): {
	fsImpl: typeof fs;
	writes: string[];
	renames: Array<[string, string]>;
	reads: string[];
} {
	const writes: string[] = [];
	const renames: Array<[string, string]> = [];
	const reads: string[] = [];
	const fsImpl = {
		...fs,
		writeFileSync: ((p: fs.PathLike, data: unknown, opts?: unknown) => {
			writes.push(String(p));
			return (fs.writeFileSync as (...a: unknown[]) => void)(p, data, opts);
		}) as typeof fs.writeFileSync,
		renameSync: ((from: fs.PathLike, to: fs.PathLike) => {
			renames.push([String(from), String(to)]);
			return fs.renameSync(from, to);
		}) as typeof fs.renameSync,
		readFileSync: ((p: fs.PathLike, opts?: unknown) => {
			reads.push(String(p));
			return (fs.readFileSync as (...a: unknown[]) => unknown)(p, opts);
		}) as typeof fs.readFileSync,
	} as typeof fs;
	return { fsImpl, writes, renames, reads };
}

function line(value: unknown): string {
	return JSON.stringify(value);
}

/**
 * Write a pi-style session JSONL file with user + assistant round pairs.
 * Each pair is one user message followed by one assistant message.
 */
function writeSession(filePath: string, pairs: Array<{ userPrompt: string; responseSequence: string }>): void {
	const entries: unknown[] = [];
	for (let i = 0; i < pairs.length; i++) {
		entries.push({
			type: "message",
			id: `u${i}`,
			message: { role: "user", content: [{ type: "text", text: pairs[i].userPrompt }] },
		});
		entries.push({
			type: "message",
			id: `a${i}`,
			message: {
				role: "assistant",
				content: [{ type: "text", text: pairs[i].responseSequence }],
			},
		});
	}
	fs.writeFileSync(filePath, entries.map(line).join("\n"));
}

/**
 * Write a pi-style session JSONL whose single round ends with a
 * `semblr_checkpoint` tool call carrying `summary` and the accepted-result
 * text the extractor keys on. Mirrors the live capture shape
 * (src/semblr.ts) so `extractCheckpointSummary` recovers it.
 */
function writeSessionWithCheckpoint(
	filePath: string,
	userPrompt: string,
	responseSequence: string,
	summary: Record<string, unknown>,
): void {
	const entries: unknown[] = [
		{ type: "message", id: "u0", message: { role: "user", content: [{ type: "text", text: userPrompt }] } },
		{
			type: "message",
			id: "a0",
			message: {
				role: "assistant",
				content: [{ type: "toolCall", id: "t1", name: "semblr_checkpoint", arguments: summary }],
			},
		},
		{
			type: "message",
			id: "r0",
			message: {
				role: "toolResult",
				toolName: "semblr_checkpoint",
				toolCallId: "t1",
				content: [
					{
						type: "text",
						text: "Checkpoint recorded. Your progress summary has been saved. You may now stop — do not start new work.",
					},
				],
			},
		},
		{
			type: "message",
			id: "a1",
			message: { role: "assistant", content: [{ type: "text", text: responseSequence }] },
		},
	];
	fs.writeFileSync(filePath, entries.map(line).join("\n"));
}

function embeddingFetch(vectors: number[][], requests: unknown[] = []): typeof fetch {
	return vi.fn(async (input, init) => {
		requests.push({
			input,
			method: init?.method,
			headers: init?.headers,
			body: JSON.parse(String(init?.body)),
		});
		return new Response(JSON.stringify({ data: [{ embedding: vectors.shift() ?? [1] }] }), { status: 200 });
	}) as typeof fetch;
}

describe("digest-all script", () => {
	it("reports missing API key", async () => {
		const logs = logger();
		const sessionsDir = tmpDir();
		const roundsDir = tmpDir();

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				apiKey: "",
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(1);

		expect(logs.stderr).toContain("❌ OPENROUTER_API_KEY environment variable required");
	});

	it("handles empty or nonexistent sessions directory", async () => {
		const logs = logger();
		const sessionsDir = tmpDir();
		const roundsDir = tmpDir();

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				apiKey: "key",
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(logs.stdout.join("\n")).toContain("📂 Found 0 session files");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to embed: 0");
	});

	it("processes multiple sessions, embeds prompts and responses, and writes round files", async () => {
		const root = tmpDir();
		// Create a sessions directory structure: --session1/file.jsonl, --session2/file.jsonl
		const sessionsDir = path.join(root, "sessions");
		const s1Dir = path.join(sessionsDir, "--session1");
		const s2Dir = path.join(sessionsDir, "--session2");
		fs.mkdirSync(s1Dir, { recursive: true });
		fs.mkdirSync(s2Dir, { recursive: true });

		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const logs = logger();
		const requests: unknown[] = [];

		const userPrompt1 = `${"p".repeat(8100)}`;
		const resp1 = `${"r".repeat(8100)}`;
		const userPrompt2 = "What is the answer to life?";
		const resp2 = "The answer is forty two.";

		writeSession(path.join(s1Dir, "session.jsonl"), [{ userPrompt: userPrompt1, responseSequence: resp1 }]);
		writeSession(path.join(s2Dir, "session.jsonl"), [{ userPrompt: userPrompt2, responseSequence: resp2 }]);

		const fetchImpl = embeddingFetch(
			[
				[3, 4], // prompt1
				[0, 0], // resp1
				[3, 0], // combined1 (stored raw as promptEmbedding)
				[1, 1], // prompt2
				[2, 2], // resp2
				[1, 0], // combined2
			],
			requests,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// Verify round files exist
		const roundFile1 = `${computeContentHash(userPrompt1, resp1, [])}.json`;
		const roundFile2 = `${computeContentHash(userPrompt2, resp2, [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, roundFile1))).toBe(true);
		expect(fs.existsSync(path.join(roundsDir, roundFile2))).toBe(true);

		// Verify index
		const index = loadVectorIndex(indexPath);
		expect(index).toHaveLength(4); // 2 prompts + 2 responses

		// Prompt text is noise-cleaned (full cleaned text, no slice); the response
		// stays whole — 8,100 bytes fit the shared 24,000-byte response budget (F1).
		expect(requests).toEqual([
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: "[REPEAT: 'p' × 8100]" },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: "r".repeat(8100) },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: `[REPEAT: 'p' × 8100]\n\n${"r".repeat(8100)}` },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: userPrompt2 },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: resp2 },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: `${userPrompt2}\n\n${resp2}` },
			},
		]);

		expect(logs.stdout.join("\n")).toContain("📂 Found 2 session files across 2 directories");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to embed: 2");
		expect(logs.stdout.join("\n")).toContain("✅ Done");
	});

	it("skips already-indexed rounds on second run", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "What is the capital of France?";
		const responseSequence = "The capital of France is Paris.";

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);

		// First run — embeds everything
		const firstFetch = embeddingFetch(
			[
				[1, 0],
				[0, 1],
				[2, 3],
			],
			[],
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: firstFetch,
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		expect(firstFetch).toHaveBeenCalledTimes(3);

		// Second run — skip all
		const secondLogs = logger();
		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: secondFetch,
				stdout: secondLogs.out,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);

		expect(secondFetch).not.toHaveBeenCalled();
		expect(secondLogs.stdout.join("\n")).toContain(
			"📊 New rounds to embed: 0 (1 already indexed, 0 swept from rounds dir)",
		);
		expect(secondLogs.stdout.join("\n")).toContain("✨ Nothing to do — all sessions already indexed!");
	});

	it("does not re-embed legacy two-column rows because missing model means current model", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "What is a legacy indexed prompt?";
		const responseSequence = "It is an old two-column index row.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [] }),
		);
		fs.writeFileSync(
			indexPath,
			`${encodeVectorIndexLine([1], `${roundFile}:prompt`)}\n${encodeVectorIndexLine([2], `${roundFile}:response`)}\n`,
		);

		const logs = logger();
		const fetchImpl = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// Legacy rows count as current-model, so the round is healed from its
		// `:response` row (no API call) and its rows are untouched.
		expect(fetchImpl).not.toHaveBeenCalled();
		expect(readIndexLines(indexPath)).toEqual([
			encodeVectorIndexLine([1], `${roundFile}:prompt`),
			encodeVectorIndexLine([2], `${roundFile}:response`),
		]);
		expect(logs.stdout.join("\n")).toContain("📊 Model-mismatched rounds to re-index: 0");
	});

	it("re-indexes rounds whose index rows were generated with a different explicit model", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "What should be re-indexed?";
		const responseSequence = "Rounds embedded with an old model should be refreshed.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		const unrelated = encodeVectorIndexLine([9], "unrelated.json:prompt", "old-model");
		const requests: unknown[] = [];

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [] }),
		);
		fs.writeFileSync(
			indexPath,
			`${[
				encodeVectorIndexLine([1], `${roundFile}:prompt`, "old-model"),
				encodeVectorIndexLine([2], `${roundFile}:response`, "old-model"),
				unrelated,
			].join("\n")}\n`,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[3, 4],
						[0, 5],
						[6, 7],
					],
					requests,
				),
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		expect(requests).toHaveLength(3);
		expect((requests[0] as any).body).toEqual({
			model: "openai/text-embedding-3-small",
			input: userPrompt,
		});
		expect((requests[1] as any).body).toEqual({ model: "openai/text-embedding-3-small", input: responseSequence });
		expect((requests[2] as any).body).toEqual({
			model: "openai/text-embedding-3-small",
			input: `${userPrompt}\n\n${responseSequence}`,
		});
		expect(loadVectorIndex(indexPath)).toEqual([
			{ vector: [9], filePath: "unrelated.json:prompt", model: "old-model" },
			{
				vector: [0.6, 0.8],
				filePath: `${roundFile}:prompt`,
				model: "openai/text-embedding-3-small",
				embeddingInputHash: hashEmbeddingInput(userPrompt),
			},
			{ vector: [0, 1], filePath: `${roundFile}:response`, model: "openai/text-embedding-3-small" },
		]);
	});

	it("a model-change reindex reproduces the :summary row derived from a session checkpoint call (F3)", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "What is the checkpoint state right now?";
		const responseSequence = "Checkpoint saved with the mid-task state.";
		const summary = {
			currentTask: "F3 parity",
			progressMade: ["derived summary"],
			currentState: ["mid"],
			nextSteps: ["guard"],
			keyFindings: ["rows matter"],
		};
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [
			{
				arguments: JSON.stringify(summary),
				result_summary:
					"Checkpoint recorded. Your progress summary has been saved. You may now stop — do not start new work.",
			},
		])}.json`;

		writeSessionWithCheckpoint(path.join(sDir, "session.jsonl"), userPrompt, responseSequence, summary);
		// Old-model rows for prompt+response+summary: the reindex must reproduce all
		// three from the session, not just prompt+response.
		fs.writeFileSync(
			indexPath,
			`${[
				encodeVectorIndexLine([1], `${roundFile}:prompt`, "old-model"),
				encodeVectorIndexLine([2], `${roundFile}:response`, "old-model"),
				encodeVectorIndexLine([3], `${roundFile}:summary`, "old-model"),
			].join("\n")}\n`,
		);

		const requests: unknown[] = [];
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[3, 4],
						[0, 5],
						[6, 7],
						[8, 9],
					],
					requests,
				),
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		// The summary row is present and carries the derived checkpoint text.
		const rows = loadVectorIndex(indexPath);
		expect(rows.map((e) => e.filePath)).toEqual([
			`${roundFile}:prompt`,
			`${roundFile}:response`,
			`${roundFile}:summary`,
		]);
		const summaryRequest = (requests as Array<{ body: { input: string } }>).find((r) =>
			r.body.input.includes("F3 parity"),
		);
		expect(summaryRequest).toBeDefined();
	});

	it("a model-change reindex of a session round with no checkpoint call emits no :summary row (F3)", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "No checkpoint here, just a plain round?";
		const responseSequence = "A plain answer without any checkpoint call.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);
		fs.writeFileSync(
			indexPath,
			`${[
				encodeVectorIndexLine([1], `${roundFile}:prompt`, "old-model"),
				encodeVectorIndexLine([2], `${roundFile}:response`, "old-model"),
			].join("\n")}\n`,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([
					[3, 4],
					[0, 5],
					[6, 7],
				]),
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		const rows = loadVectorIndex(indexPath);
		expect(rows.map((e) => e.filePath)).toEqual([`${roundFile}:prompt`, `${roundFile}:response`]);
	});

	it("a model-change reindex preserves an index row it did not reproduce (F3 guard)", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "Does the reindex silently drop my orphan row?";
		const responseSequence = "It must be reported and preserved.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);
		// A `:round` row exists on disk with no source to reproduce it from (the
		// round file carries no such field and there is no checkpoint call).
		fs.writeFileSync(
			indexPath,
			`${[
				encodeVectorIndexLine([1], `${roundFile}:prompt`, "old-model"),
				encodeVectorIndexLine([2], `${roundFile}:response`, "old-model"),
				encodeVectorIndexLine([7, 7], `${roundFile}:round`, "old-model"),
			].join("\n")}\n`,
		);

		const logs = logger();
		const requests: unknown[] = [];
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[3, 4],
						[0, 5],
						[6, 7],
					],
					requests,
				),
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const rows = loadVectorIndex(indexPath);
		// prompt+response are replaced with current-model rows; the orphan `:round`
		// row survives.
		expect(rows.map((e) => e.filePath)).toEqual([
			`${roundFile}:prompt`,
			`${roundFile}:response`,
			`${roundFile}:round`,
		]);
		expect(rows.find((e) => e.filePath === `${roundFile}:round`)?.vector).toEqual([7, 7]);
		expect(logs.stderr.join("\n")).toContain(`Preserving non-reproduced index row: ${roundFile}:round`);

		// F1 (PR #141 review): the preserved orphan keeps its own model — its vector
		// was computed by another model, and re-stamping it to the current model
		// would score a foreign-space vector against the current-model query vector
		// (a cross-model cosine). Convergence instead comes from the mismatch
		// predicate ignoring the non-reproducible `:round` suffix: a second run must
		// resolve and spend zero embedding calls without relabelling the row.
		expect(rows.find((e) => e.filePath === `${roundFile}:round`)?.model).toBe("old-model");
		const firstRunCalls = requests.length;
		const secondLogs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([], requests),
				stdout: secondLogs.out,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);
		expect(requests.length).toBe(firstRunCalls);
		expect(secondLogs.stderr.join("\n")).not.toContain("Model-mismatched rounds to re-index: 1");
		const rowsAfterSecondRun = loadVectorIndex(indexPath);
		expect(rowsAfterSecondRun.map((e) => e.filePath)).toEqual(rows.map((e) => e.filePath));
		for (const row of rowsAfterSecondRun.filter((e) => !e.filePath.endsWith(":round"))) {
			expect(row.model).toBe(DEBUG_EMBEDDING_MODEL);
		}
		expect(rowsAfterSecondRun.find((e) => e.filePath.endsWith(":round"))?.model).toBe("old-model");
	});

	it("a model-change reindex still rewrites a mismatched reproducible row", async () => {
		// Guard for the F1 fix: excluding non-reproducible suffixes from the mismatch
		// predicate must not exclude reproducible ones. A stale `:prompt` row is
		// reproducible, so its round is re-embedded and the row replaced.
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "Does a stale reproducible row still converge?";
		const responseSequence = "It must be re-embedded under the current model.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);
		fs.writeFileSync(indexPath, `${[encodeVectorIndexLine([1], `${roundFile}:prompt`, "old-model")].join("\n")}\n`);

		const logs = logger();
		const requests: unknown[] = [];
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[3, 4],
						[0, 5],
					],
					requests,
				),
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const rows = loadVectorIndex(indexPath);
		expect(rows.find((e) => e.filePath === `${roundFile}:prompt`)?.model).toBe(DEBUG_EMBEDDING_MODEL);
		for (const row of rows) {
			expect(row.model).toBe(DEBUG_EMBEDDING_MODEL);
		}
	});

	it("a legacy :round-only round is reindexed once and then converges", async () => {
		// F1 (PR #141 review), case 2: a round whose only row is a non-reproducible
		// legacy `:round` row has no current-model reproducible row, so it must be
		// reindexed once to gain `:prompt`/`:response`. The preserved `:round` orphan
		// then keeps its own model, and the round must not be reindexed again.
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "Only a legacy combined row exists for this round, with plenty of words to keep the prompt.";
		const responseSequence = "It must gain current-model rows and then stop being reindexed.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);
		fs.writeFileSync(indexPath, `${[encodeVectorIndexLine([7, 7], `${roundFile}:round`, "old-model")].join("\n")}\n`);

		const logs = logger();
		const requests: unknown[] = [];
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[3, 4],
						[0, 5],
						[6, 8],
					],
					requests,
				),
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const rows = loadVectorIndex(indexPath);
		expect(rows.map((e) => e.filePath).sort()).toEqual(
			[`${roundFile}:prompt`, `${roundFile}:response`, `${roundFile}:round`].sort(),
		);
		expect(rows.find((e) => e.filePath === `${roundFile}:round`)?.model).toBe("old-model");
		expect(rows.find((e) => e.filePath === `${roundFile}:prompt`)?.model).toBe(DEBUG_EMBEDDING_MODEL);

		const firstRunCalls = requests.length;
		const secondLogs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([], requests),
				stdout: secondLogs.out,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);
		expect(requests.length).toBe(firstRunCalls);
		expect(secondLogs.stderr.join("\n")).not.toContain("Model-mismatched rounds to re-index: 1");
	});

	it("a stale :prompt row on a short-prompt round is an orphan and converges", async () => {
		// F1 (PR !141 review): a short prompt reproduces no `:prompt` row
		// (lib/embed-round.ts short-prompt drop), so an old-model `:prompt` row on
		// such a round can never be replaced by a reindex. It must be treated as a
		// non-reproducible orphan: the round is reindexed once for its stale
		// `:response` row, the preserved `:prompt` orphan keeps its own model, and a
		// second run must resolve and spend zero embedding calls.
		const prevMinWords = process.env.RELEVANCE_LIST_MIN_WORDS;
		process.env.RELEVANCE_LIST_MIN_WORDS = "20"; // the documented default threshold
		try {
			const root = tmpDir();
			const sessionsDir = tmpDir(); // sweep-only
			const roundsDir = path.join(root, "rounds");
			fs.mkdirSync(roundsDir, { recursive: true });
			const indexPath = path.join(roundsDir, "index.csv");
			const userPrompt = "What was recovered?"; // 3 words — under the threshold
			const responseSequence = "Short prompts never carry prompt-side embeddings on the live path.";
			const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
			fs.writeFileSync(
				path.join(roundsDir, roundFile),
				JSON.stringify({ userPrompt, responseSequence, toolCalls: [] }),
			);
			fs.writeFileSync(
				indexPath,
				`${[
					encodeVectorIndexLine([1], `${roundFile}:prompt`, "old-model"),
					encodeVectorIndexLine([2], `${roundFile}:response`, "old-model"),
				].join("\n")}\n`,
			);

			const requests: unknown[] = [];
			const logs = logger();
			await expect(
				runDigestAll({
					sessionsDir,
					roundsDir,
					indexPath,
					apiKey: "key",
					fetchImpl: embeddingFetch([[9, 9]], requests),
					stdout: logs.out,
					stderr: logs.err,
				}),
			).resolves.toBe(0);

			const rows = loadVectorIndex(indexPath);
			// The stale `:response` is replaced; the unreproducible `:prompt` orphan is
			// preserved with its own model (never re-stamped — cross-model cosine).
			expect(rows.find((e) => e.filePath === `${roundFile}:response`)?.model).toBe(DEBUG_EMBEDDING_MODEL);
			expect(rows.find((e) => e.filePath === `${roundFile}:prompt`)?.model).toBe("old-model");

			const firstRunCalls = requests.length;
			const secondLogs = logger();
			await expect(
				runDigestAll({
					sessionsDir,
					roundsDir,
					indexPath,
					apiKey: "key",
					fetchImpl: embeddingFetch([], requests),
					stdout: secondLogs.out,
					stderr: secondLogs.err,
				}),
			).resolves.toBe(0);
			expect(requests.length).toBe(firstRunCalls);
			expect(secondLogs.stderr.join("\n")).not.toContain("Model-mismatched rounds to re-index: 1");
			expect(secondLogs.stdout.join("\n")).toContain("Model-mismatched rounds to re-index: 0");
		} finally {
			if (prevMinWords === undefined) delete process.env.RELEVANCE_LIST_MIN_WORDS;
			else process.env.RELEVANCE_LIST_MIN_WORDS = prevMinWords;
		}
	});

	it("a stale :summary orphan does not re-flag a round that has current prompt/response rows", async () => {
		// F1 (PR #141 review), case 3: `:summary` is auxiliary and content-dependent.
		// A round whose `:prompt`/`:response` rows are current but whose `:summary`
		// row is stale must converge — `:summary` staleness alone must not force a
		// reindex, or an unreproducible `:summary` orphan would loop forever.
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "A round with current retrieval rows and a stale summary row, long enough to keep the prompt.";
		const responseSequence = "The summary staleness must not force a reindex.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);
		fs.writeFileSync(
			indexPath,
			`${[
				encodeVectorIndexLine([1], `${roundFile}:prompt`, DEBUG_EMBEDDING_MODEL),
				encodeVectorIndexLine([2], `${roundFile}:response`, DEBUG_EMBEDDING_MODEL),
				encodeVectorIndexLine([3], `${roundFile}:summary`, "old-model"),
			].join("\n")}\n`,
		);

		const logs = logger();
		const requests: unknown[] = [];
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([], requests),
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(requests.length).toBe(0);
		expect(logs.stderr.join("\n")).not.toContain("Model-mismatched rounds to re-index: 1");
		const rows = loadVectorIndex(indexPath);
		expect(rows.find((e) => e.filePath === `${roundFile}:summary`)?.model).toBe("old-model");
	});

	it("migrates stale round filenames before deciding a round is already indexed", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		const userPrompt = "What is the same prompt?";
		const responseSequence = "The same prompt always returns the same answer.";

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);

		// Pre-create a stale round file and index entry
		fs.writeFileSync(
			path.join(roundsDir, "legacy.json"),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [] }),
		);
		fs.writeFileSync(indexPath, `${encodeVectorIndexLine([1], "legacy.json:prompt")}\n`);

		const logs = logger();
		const fetchImpl = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, "legacy.json"))).toBe(false);
		expect(fs.existsSync(path.join(roundsDir, roundFile))).toBe(true);
		expect(readIndexLines(indexPath)).toEqual([encodeVectorIndexLine([1], `${roundFile}:prompt`)]);
		expect(logs.stderr.join("\n")).toContain(`Migrated stale: legacy.json → ${roundFile}`);
		expect(logs.stdout.join("\n")).toContain("1 rounds embedded, 0 errors");
		expect(fetchImpl).not.toHaveBeenCalled();
	});

	it("handles embedding API errors gracefully and continues processing", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const s1Dir = path.join(sessionsDir, "--session1");
		const s2Dir = path.join(sessionsDir, "--session2");
		fs.mkdirSync(s1Dir, { recursive: true });
		fs.mkdirSync(s2Dir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");

		const userPrompt1 = "What is a good prompt?";
		const resp1 = "A good prompt is clear and specific.";
		const userPrompt2 = "What is a failing prompt?";
		const resp2 = "A failing prompt causes an API error.";

		writeSession(path.join(s1Dir, "session.jsonl"), [{ userPrompt: userPrompt1, responseSequence: resp1 }]);
		writeSession(path.join(s2Dir, "session.jsonl"), [{ userPrompt: userPrompt2, responseSequence: resp2 }]);

		// First round: success. Second round: prompt fails, embeds response anyway.
		// But since our processRound tries prompt first, a prompt failure skips response embedding.
		// Let's make the first call (first round prompt) succeed, second call (first round response) succeed,
		// third call (second round prompt) fail.
		const logs = logger();
		const fetchImpl = vi.fn(async (_input: unknown, init: any) => {
			const body = JSON.parse(String((init as any).body));
			if (body.input === userPrompt2) {
				return new Response("server error", { status: 500 });
			}
			return new Response(JSON.stringify({ data: [{ embedding: [1, 2] }] }), { status: 200 });
		}) as typeof fetch;

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const roundFile1 = `${computeContentHash(userPrompt1, resp1, [])}.json`;
		const roundFile2 = `${computeContentHash(userPrompt2, resp2, [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, roundFile1))).toBe(true);
		expect(fs.existsSync(path.join(roundsDir, roundFile2))).toBe(true);

		expect(logs.stderr.join("\n")).toContain("[ERROR]");
		expect(logs.stdout.join("\n")).toContain("1 rounds embedded, 1 errors");
	});

	it("noise-cleans long prompts and keeps a 12k-char response whole (24,000-byte budget)", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		const longPrompt = `${"A".repeat(10000)}`;
		const longResponse = `${"B".repeat(12000)}`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt: longPrompt, responseSequence: longResponse }]);

		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
				[1, 1],
			],
			requests,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		// Prompt input is the noise-cleaned full text (repetition collapsed). The
		// response is NOT clipped at 8,000 code units (the old redrive behavior) —
		// 12,000 ASCII chars fit the shared 24,000-byte response budget intact (F1).
		expect(requests).toHaveLength(3);
		expect((requests[0] as any).body.input).toBe("[REPEAT: 'A' × 10000]");
		expect((requests[1] as any).body.input).toBe("B".repeat(12000));
		expect((requests[2] as any).body.input).toBe(`[REPEAT: 'A' × 10000]\n\n${"B".repeat(12000)}`);
	});

	it("clips a response longer than 24,000 bytes at the byte budget", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		writeSession(path.join(sDir, "session.jsonl"), [
			{ userPrompt: "Please clip the response at the byte budget", responseSequence: "C".repeat(30000) },
		]);

		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
				[1, 1],
			],
			requests,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		expect(requests).toHaveLength(3);
		expect((requests[1] as any).body.input).toBe("C".repeat(24000));
		expect((requests[2] as any).body.input).toBe(
			`Please clip the response at the byte budget\n\n${"C".repeat(24000)}`,
		);
	});

	it("skips non-session (non-dash-dash) directories", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		fs.mkdirSync(path.join(sessionsDir, "--real"), { recursive: true });
		fs.mkdirSync(path.join(sessionsDir, "not-a-session"), { recursive: true });

		// Put a session file in the real dir only
		writeSession(path.join(sessionsDir, "--real", "session.jsonl"), [
			{ userPrompt: "What is the capital?", responseSequence: "The capital of France is Paris." },
		]);
		// Put a file in the non-session dir that should be ignored
		writeSession(path.join(sessionsDir, "not-a-session", "session.jsonl"), [
			{ userPrompt: "ignored", responseSequence: "This response should never be seen." },
		]);

		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const logs = logger();
		const requests: unknown[] = [];

		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
				[2, 3],
			],
			requests,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// Only the real session was picked up
		expect(logs.stdout.join("\n")).toContain("📂 Found 1 session files across 1 directories");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to embed: 1");
		expect(requests).toHaveLength(3);
	});

	it("uses environment and fetch defaults when options are not injected", async () => {
		const root = tmpDir();
		const _sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");

		const oldKey = process.env.OPENROUTER_API_KEY;
		const oldFetch = globalThis.fetch;
		const fetchSpy = vi.fn(
			async () => new Response(JSON.stringify({ data: [{ embedding: [9, 9] }] }), { status: 200 }),
		) as typeof fetch;
		process.env.OPENROUTER_API_KEY = "env-key";
		globalThis.fetch = fetchSpy;
		const sessionsOnlyDir = path.join(root, "sessions");
		const sDir = path.join(sessionsOnlyDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		writeSession(path.join(sDir, "session.jsonl"), [
			{ userPrompt: "prompt from env", responseSequence: "response from environment variable." },
		]);

		try {
			await expect(runDigestAll({ sessionsDir: sessionsOnlyDir, roundsDir, indexPath })).resolves.toBe(0);

			expect(fetchSpy).toHaveBeenCalledWith(
				"https://openrouter.ai/api/v1/embeddings",
				expect.objectContaining({
					headers: expect.objectContaining({ Authorization: "Bearer env-key" }),
				}),
			);
		} finally {
			if (oldKey === undefined) delete process.env.OPENROUTER_API_KEY;
			else process.env.OPENROUTER_API_KEY = oldKey;
			globalThis.fetch = oldFetch;
		}
	});

	it("sweeps recovered rounds from the rounds dir even when no session file exists", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir(); // no session files at all
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		const userPrompt = "What was recovered after the crash?";
		const responseSequence = "This round was recovered by the startup backfill.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
		);

		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[1, 0],
						[0, 1],
						[3, 4],
					],
					requests,
				),
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// Prompt + response rows embedded; the combined embed call carries no row.
		expect(requests).toHaveLength(3);
		expect(loadVectorIndex(indexPath)).toHaveLength(2);
		const written = JSON.parse(fs.readFileSync(path.join(roundsDir, roundFile), "utf-8"));
		expect(written.recovered).toBe(true);
		// F2: redriven rounds now carry promptEmbedding — the raw combined vector
		// (unnormalized, live scale convention), not an index-row vector.
		expect(written.promptEmbedding).toEqual([3, 4]);
		expect(logs.stdout.join("\n")).toContain("1 swept from rounds dir");
		expect(logs.stdout.join("\n")).toContain("1 rounds embedded, 0 errors");
	});

	it("rounds-dir sweep is idempotent — re-run adds no duplicate index rows", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		const userPrompt = "What is swept exactly once?";
		const responseSequence = "The sweep must not duplicate index rows on re-run.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([
					[1, 0],
					[0, 1],
					[2, 3],
				]),
				stdout: logger().out,
			}),
		).resolves.toBe(0);
		const linesAfterFirst = readIndexLines(indexPath);
		expect(linesAfterFirst).toHaveLength(2);

		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: secondFetch,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(secondFetch).not.toHaveBeenCalled();
		expect(readIndexLines(indexPath)).toEqual(linesAfterFirst);
		expect(logs.stdout.join("\n")).toContain("0 swept from rounds dir");
	});

	it("sweep heals a marker-less round that already has rows, reusing the :response vector without embedding", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		const userPrompt = "A recovered round whose rows survived but whose marker did not.";
		const responseSequence = "The startup pending counter must agree with what just index can do.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		// Round file has rows in the index but no promptEmbedding marker.
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
		);
		fs.writeFileSync(
			indexPath,
			[
				encodeVectorIndexLine([0.25, 0.5], `${roundFile}:prompt`, "openai/text-embedding-3-small"),
				encodeVectorIndexLine([0.75, 0.125], `${roundFile}:response`, "openai/text-embedding-3-small"),
			].join("\n") + "\n",
		);
		const linesBefore = readIndexLines(indexPath);

		const fetchImpl = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// No embed call — healing reuses the existing :response row vector.
		expect(fetchImpl).not.toHaveBeenCalled();
		// Index rows are untouched.
		expect(readIndexLines(indexPath)).toEqual(linesBefore);
		const written = JSON.parse(fs.readFileSync(path.join(roundsDir, roundFile), "utf-8"));
		expect(written.promptEmbedding).toEqual([0.75, 0.125]);
		expect(written.recovered).toBe(true);
		expect(logs.stdout.join("\n")).toContain("1 markers healed");
	});

	it("sweep re-embeds a marker-less round with no :response row, replacing rows and writing the marker", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		const userPrompt = "A round with a prompt row but no response row.";
		const responseSequence = "Re-embedded because no truthful vector was available to heal from.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
		);
		fs.writeFileSync(
			indexPath,
			`${encodeVectorIndexLine([0.25, 0.5], `${roundFile}:prompt`, "openai/text-embedding-3-small")}\n`,
		);

		// F1 (issue #140, D1/D2): rows are the retrieval truth, so a round with
		// current-model rows but no :response row to heal from is a retrieval gap
		// and is re-embedded from its round file. The sweep spends the call and the
		// full replace reproduces prompt+response rows and the marker together.
		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
			],
			requests,
		) as typeof fetch;
		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(fetchImpl).toHaveBeenCalledTimes(3);
		const rows = loadVectorIndex(indexPath);
		expect(rows.map((e) => e.filePath).sort()).toEqual([`${roundFile}:prompt`, `${roundFile}:response`].sort());
		const written = JSON.parse(fs.readFileSync(path.join(roundsDir, roundFile), "utf-8"));
		expect(written.promptEmbedding).toBeDefined();
		expect(logs.stdout.join("\n")).not.toContain("markers healed");

		// Idempotence (D4): a second run sees current-model rows and embeds nothing.
		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		const secondLogs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: secondFetch,
				stdout: secondLogs.out,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);
		expect(secondFetch).not.toHaveBeenCalled();
	});

	it("sweep skips unreadable round files and files without a user prompt", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		fs.writeFileSync(path.join(roundsDir, "corrupt.json"), "{not json");
		fs.writeFileSync(path.join(roundsDir, "no-prompt.json"), JSON.stringify({ responseSequence: "orphan" }));

		const logs = logger();
		const fetchImpl = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(fetchImpl).not.toHaveBeenCalled();
		expect(loadVectorIndex(indexPath)).toHaveLength(0);
		expect(logs.stderr.join("\n")).toContain("Skipping unreadable round file: corrupt.json");
		expect(logs.stderr.join("\n")).toContain("Skipping invalid round file: no-prompt.json");
		expect(logs.stdout.join("\n")).toContain("0 swept from rounds dir");
		// D4: the unresolvable files are named as a set, not just warned about one
		// at a time, so the operator sees the store will not converge.
		expect(logs.stdout.join("\n")).toContain("2 non-convergent rounds");
		expect(logs.stdout.join("\n")).toContain("• corrupt.json");
		expect(logs.stdout.join("\n")).toContain("• no-prompt.json");
	});

	it("a resolvable round still embeds and converges; a second run embeds nothing (D4)", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		// One round the sweep cannot resolve, one it can.
		fs.writeFileSync(path.join(roundsDir, "broken.json"), "{not json");
		const userPrompt = "A resolvable swept round sitting next to an unresolvable one.";
		const responseSequence = "It still embeds, and a second run spends nothing.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
		);

		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
			],
			requests,
		) as typeof fetch;
		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// The resolvable round embeds; the unresolvable one is named rather than
		// silently dropped from the count.
		expect(fetchImpl).toHaveBeenCalledTimes(3);
		expect(logs.stdout.join("\n")).toContain("1 non-convergent round");
		expect(logs.stdout.join("\n")).toContain("• broken.json");
		expect(
			loadVectorIndex(indexPath)
				.map((e) => e.filePath)
				.sort(),
		).toEqual([`${roundFile}:prompt`, `${roundFile}:response`].sort());

		// D4 convergence proof: a second run reaches no embeddable round and spends
		// no embedding call — the only rounds left are the non-convergent one.
		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		const secondLogs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: secondFetch,
				stdout: secondLogs.out,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);
		expect(secondFetch).not.toHaveBeenCalled();
		expect(secondLogs.stdout.join("\n")).toContain("📊 New rounds to embed: 0");
		expect(secondLogs.stdout.join("\n")).toContain("✨ Nothing to do — all sessions already indexed!");
		expect(secondLogs.stdout.join("\n")).toContain("• broken.json");
	});

	it("sweep skips parseable-but-incomplete round files by name instead of failing the whole run", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		// F5 shape: parseable JSON whose missing fields would throw inside
		// processRound before its try/catch and abort the run.
		fs.writeFileSync(path.join(roundsDir, "incomplete.json"), JSON.stringify({ userPrompt: "hi" }));

		const userPrompt = "What survives an incomplete sibling round file?";
		const responseSequence = "Well-formed rounds embed while invalid files are skipped by name.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
		);

		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[1, 0],
						[0, 1],
						[2, 3],
					],
					requests,
				),
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0); // a fatal abort would reject instead of resolving 0

		// The well-formed round is embedded (prompt + response rows, 3 embed calls
		// including the combined vector); the bad file is named on stderr.
		expect(requests).toHaveLength(3);
		expect(loadVectorIndex(indexPath)).toHaveLength(2);
		expect(logs.stderr.join("\n")).toContain("Skipping invalid round file: incomplete.json");
		expect(logs.stdout.join("\n")).toContain("1 swept from rounds dir");
		expect(logs.stdout.join("\n")).toContain("1 rounds embedded, 0 errors");
	});

	it("sweep skips round files whose toolCalls hold non-object elements instead of failing the whole run", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		// F3 shape: Array.isArray passes, but a null element throws in
		// deriveRoundFile/computeContentHash before processRound's try/catch.
		fs.writeFileSync(
			path.join(roundsDir, "null-tool.json"),
			JSON.stringify({ userPrompt: "hi", responseSequence: "there", toolCalls: [null] }),
		);

		const userPrompt = "Do non-object toolCalls elements abort the sweep?";
		const responseSequence = "Well-formed rounds embed while malformed element arrays are skipped by name.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
		);

		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[1, 0],
						[0, 1],
						[2, 3],
					],
					requests,
				),
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0); // a fatal abort would reject instead of resolving 0

		expect(requests).toHaveLength(3);
		expect(loadVectorIndex(indexPath)).toHaveLength(2);
		expect(logs.stderr.join("\n")).toContain("Skipping invalid round file: null-tool.json");
		expect(logs.stdout.join("\n")).toContain("1 swept from rounds dir");
		expect(logs.stdout.join("\n")).toContain("1 rounds embedded, 0 errors");
	});

	it("writes round files and markers atomically via tmp+rename, leaving no tmp residue", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		// Recovered round (no session JSONL) forces the sweep -> processRound path,
		// which performs both writes under test: the round file, then the
		// promptEmbedding marker through writeRoundEmbedding.
		const userPrompt = "Are the digest round-file writes atomic yet?";
		const responseSequence = "They should stage to a pid-suffixed temp file and rename over the target.";
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		const roundPath = path.join(roundsDir, roundFile);
		fs.writeFileSync(roundPath, JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }));

		const { fsImpl, writes, renames } = spyFs();
		const logs = logger();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([
					[1, 0],
					[0, 1],
					[2, 3],
				]),
				fsImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const tmpSuffix = `.tmp.${process.pid}`;
		const roundWrites = writes.filter((w) => w.endsWith(".json") || w.endsWith(tmpSuffix));
		// The round file is never written in place; every durable write is staged.
		expect(roundWrites).not.toContain(roundPath);
		const stagedWrites = roundWrites.filter((w) => w === `${roundPath}${tmpSuffix}`);
		expect(stagedWrites.length).toBeGreaterThanOrEqual(2); // round file, then marker
		// Each staged write is committed with a rename onto the target.
		for (const staged of stagedWrites) {
			expect(renames).toContainEqual([staged, roundPath]);
		}
		// No temp residue survives the successful run.
		expect(fs.readdirSync(roundsDir).some((name) => name.includes(".tmp."))).toBe(false);
		// The atomic path really ran: the marker landed on the durable file.
		expect(JSON.parse(fs.readFileSync(roundPath, "utf-8")).promptEmbedding).toEqual([2, 3]);
	});

	// Issue #139 regression anchor: the whole-store work per run must not scale
	// with the round count. Runs the sweep over two fixture sizes and asserts the
	// index.csv read count and the index.bm25.json write count are identical —
	// before the fix each round added a full-store scan, two index.csv parses, a
	// bm25 load, and a bm25 write. Uses the sweep path (no session files) so every
	// round goes through processRound.
	async function runSweepFixture(roundCount: number): Promise<{
		bm25Writes: number;
		indexReads: number;
		bm25Reads: number;
		toolIndexReads: number;
	}> {
		const root = tmpDir();
		const sessionsDir = tmpDir(); // no session files: force the rounds-dir sweep
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		// Seed an existing index.csv so the run genuinely reads it (a missing file
		// read would also count, but a populated index exercises the dedup path).
		fs.writeFileSync(
			indexPath,
			encodeVectorIndexLine(
				[0.1, 0.2],
				"00000000000000000000000000000000.json:prompt",
				"openai/text-embedding-3-small",
			) + "\n",
		);
		// Seed a valid bm25 sidecar so the run's single load actually reads it
		// (loadBm25Index skips a missing file without a read).
		fs.writeFileSync(
			bm25IndexPathForRoundsDir(roundsDir),
			JSON.stringify({ version: 1, documentCount: 0, averageDocumentLength: 0, documents: {} }),
		);
		// Seed the tool index too, so its once-per-run load is exercised.
		fs.writeFileSync(toolIndexPathForRoundsDir(roundsDir), "00000000000000000000000000000000,0,bash,placeholder\n");

		const vectors: number[][] = [];
		for (let i = 0; i < roundCount; i++) {
			const userPrompt = `Recovered round number ${i} with a distinct prompt body.`;
			const responseSequence = `Recovered response number ${i} with a distinct response body.`;
			const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
			// A sweep round with tool calls exercises the tool-index load path.
			if (i === 0) {
				fs.writeFileSync(
					path.join(roundsDir, roundFile),
					JSON.stringify({
						userPrompt,
						responseSequence,
						toolCalls: [{ index: 0, name: "bash", arguments: '{"command":"ls"}', result_summary: "ok" }],
						recovered: true,
					}),
				);
			} else {
				fs.writeFileSync(
					path.join(roundsDir, roundFile),
					JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true }),
				);
			}
			// prompt + response rows per round; the combined embed call consumes a
			// vector but writes no row.
			vectors.push([i + 1, 0], [0, i + 1], [i + 1, i + 1]);
		}

		const { fsImpl, writes, reads } = spyFs();
		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(vectors),
				fsImpl,
				stdout: logger().out,
				stderr: logger().err,
			}),
		).resolves.toBe(0);

		return {
			bm25Writes: writes.filter((w) => w === bm25IndexPathForRoundsDir(roundsDir)).length,
			indexReads: reads.filter((r) => r === indexPath).length,
			bm25Reads: reads.filter((r) => r === bm25IndexPathForRoundsDir(roundsDir)).length,
			toolIndexReads: reads.filter((r) => r === toolIndexPathForRoundsDir(roundsDir)).length,
		};
	}

	it("loads whole-store structures once per run, not once per round", async () => {
		const small = await runSweepFixture(3);
		const large = await runSweepFixture(6);

		// O(1) evidence: doubling the round count must not change any whole-store count.
		expect(large).toEqual(small);
		// The bm25 index is written exactly once for the whole batch (flushBm25 seam)
		// and loaded exactly once. The tool index is loaded once at start plus a
		// single lockfile read for the one appending round — both O(1) in N.
		expect(small.bm25Writes).toBe(1);
		expect(small.bm25Reads).toBe(1);
		expect(small.toolIndexReads).toBe(2);
	});

	it("short prompts follow the shared drop policy: no :prompt row, response vector stored as promptEmbedding", async () => {
		const prevMinWords = process.env.RELEVANCE_LIST_MIN_WORDS;
		process.env.RELEVANCE_LIST_MIN_WORDS = "20"; // the documented default threshold
		try {
			const root = tmpDir();
			const sessionsDir = tmpDir(); // sweep-only
			const roundsDir = path.join(root, "rounds");
			fs.mkdirSync(roundsDir, { recursive: true });
			const indexPath = path.join(roundsDir, "index.csv");
			const requests: unknown[] = [];

			const userPrompt = "What was recovered?"; // 3 words — under the threshold
			const responseSequence = "Short prompts never carry prompt-side embeddings on the live path.";
			const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
			fs.writeFileSync(
				path.join(roundsDir, roundFile),
				JSON.stringify({ userPrompt, responseSequence, toolCalls: [] }),
			);

			await expect(
				runDigestAll({
					sessionsDir,
					roundsDir,
					indexPath,
					apiKey: "key",
					fetchImpl: embeddingFetch([[1, 0]], requests),
					stdout: logger().out,
				}),
			).resolves.toBe(0);

			const index = loadVectorIndex(indexPath);
			expect(index.map((e) => e.filePath)).toEqual([`${roundFile}:response`]);
			const written = JSON.parse(fs.readFileSync(path.join(roundsDir, roundFile), "utf-8"));
			expect(written.promptEmbedding).toEqual([1, 0]); // normalized response vector
			expect(requests).toHaveLength(1); // response only — no prompt, no combined
		} finally {
			if (prevMinWords === undefined) delete process.env.RELEVANCE_LIST_MIN_WORDS;
			else process.env.RELEVANCE_LIST_MIN_WORDS = prevMinWords;
		}
	});

	it("a swept checkpoint round with a summary embeds a :summary row", async () => {
		const root = tmpDir();
		const sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		const userPrompt = "Where did the checkpoint stand?";
		const responseSequence = "The checkpoint captured the mid-task state.";
		const summary = {
			currentTask: "checkpoint parity",
			progressMade: ["embedded the round"],
			currentState: ["mid-task"],
			nextSteps: ["continue"],
			keyFindings: ["summary rows matter"],
		};
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		fs.writeFileSync(
			path.join(roundsDir, roundFile),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [], summary }),
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[1, 0],
						[0, 1],
						[3, 4],
						[5, 6],
					],
					requests,
				),
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		const index = loadVectorIndex(indexPath);
		expect(index.map((e) => e.filePath)).toEqual([
			`${roundFile}:prompt`,
			`${roundFile}:response`,
			`${roundFile}:summary`,
		]);
		expect((requests[3] as any).body.input).toContain("checkpoint parity");
	});

	it("sweep and embedRecoveredRounds produce identical inputs, rows, and promptEmbedding (F1/F2 parity)", async () => {
		const userPrompt = "What must the offline sweep have in common with the startup recovery path?";
		const responseSequence = "Identical embed inputs, index labels, and promptEmbedding values.";
		const summary = {
			currentTask: "parity check",
			progressMade: ["routed redrive through embedRound"],
			currentState: ["tests green"],
			nextSteps: ["commit"],
			keyFindings: ["parity by construction"],
		};
		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		const roundJson = JSON.stringify({ userPrompt, responseSequence, toolCalls: [], recovered: true, summary });

		// Path A: rounds-dir sweep in digest-all (fetch-backed embeds).
		const roundsDirA = path.join(tmpDir(), "rounds");
		fs.mkdirSync(roundsDirA, { recursive: true });
		fs.writeFileSync(path.join(roundsDirA, roundFile), roundJson);
		const requestsA: unknown[] = [];
		await expect(
			runDigestAll({
				sessionsDir: tmpDir(), // no session files — sweep only
				roundsDir: roundsDirA,
				indexPath: path.join(roundsDirA, "index.csv"),
				apiKey: "key",
				fetchImpl: embeddingFetch(
					[
						[1, 0],
						[0, 1],
						[3, 4],
						[5, 6],
					],
					requestsA,
				),
				stdout: logger().out,
			}),
		).resolves.toBe(0);
		const inputsA = requestsA.map((r) => (r as { body: { input: string } }).body.input);
		const rowsA = loadVectorIndex(path.join(roundsDirA, "index.csv"));
		const writtenA = JSON.parse(fs.readFileSync(path.join(roundsDirA, roundFile), "utf-8"));

		// Path B: startup recovery embedding (injected embeds).
		const roundsDirB = path.join(tmpDir(), "rounds");
		fs.mkdirSync(roundsDirB, { recursive: true });
		fs.writeFileSync(path.join(roundsDirB, roundFile), roundJson);
		const inputsB: string[] = [];
		const rowsB: Array<{ vector: number[]; filePath: string; model?: string; embeddingInputHash?: string }> = [];
		const seq: number[][] = [
			[1, 0],
			[0, 1],
			[3, 4],
			[5, 6],
		];
		await embedRecoveredRounds([roundFile], roundsDirB, {
			embed: (text) => {
				inputsB.push(text);
				return Promise.resolve(seq.shift()!);
			},
			appendIndexRow: (label, vec, hash) =>
				rowsB.push({
					vector: vec,
					filePath: label,
					model: "openai/text-embedding-3-small",
					embeddingInputHash: hash,
				}),
			writeRoundEmbedding: (name, vec) => {
				const p = path.join(roundsDirB, name);
				const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
				existing.promptEmbedding = vec;
				fs.writeFileSync(p, JSON.stringify(existing, null, 2));
			},
		});
		const writtenB = JSON.parse(fs.readFileSync(path.join(roundsDirB, roundFile), "utf-8"));

		expect(inputsB).toEqual(inputsA);
		expect(rowsB).toEqual(rowsA);
		expect(writtenB.promptEmbedding).toEqual(writtenA.promptEmbedding);
		// Anchor the shared row policy: prompt + response + checkpoint summary.
		expect(rowsA.map((e) => e.filePath)).toEqual([
			`${roundFile}:prompt`,
			`${roundFile}:response`,
			`${roundFile}:summary`,
		]);
		// The summary embed input carries the checkpoint text on both paths.
		expect(inputsA[3]).toContain("parity check");
	});

	it("detects direct CLI execution", () => {
		expect(isMainModule("file:///tmp/digest-all.ts", "/tmp/digest-all.ts")).toBe(true);
		expect(isMainModule("file:///tmp/digest-all.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/digest-all.ts", "")).toBe(false);
	});

	describe("tool index", () => {
		function writeSessionWithToolCall(filePath: string, userPrompt: string, finalText: string): void {
			const entries: unknown[] = [
				{ type: "message", id: "u0", message: { role: "user", content: [{ type: "text", text: userPrompt }] } },
				{
					type: "message",
					id: "a0",
					message: {
						role: "assistant",
						content: [
							{
								type: "toolCall",
								id: "t1",
								name: "bash",
								arguments: { command: "curl https://api.github.com/repos/vhallac/semblr" },
							},
						],
					},
				},
				{
					type: "message",
					id: "r0",
					message: {
						role: "toolResult",
						toolName: "bash",
						toolCallId: "t1",
						content: [{ type: "text", text: "200 OK" }],
					},
				},
				{
					type: "message",
					id: "a1",
					message: { role: "assistant", content: [{ type: "text", text: finalText }] },
				},
			];
			fs.writeFileSync(filePath, entries.map(line).join("\n"));
		}

		it("appends tool-call rows to the tool index while embedding", async () => {
			const root = tmpDir();
			const sessionsDir = path.join(root, "sessions");
			const sDir = path.join(sessionsDir, "--test");
			fs.mkdirSync(sDir, { recursive: true });
			const roundsDir = path.join(root, "rounds");
			const indexPath = path.join(roundsDir, "index.csv");

			writeSessionWithToolCall(
				path.join(sDir, "session.jsonl"),
				"Please curl the repo",
				"Done, fetched the repo info.",
			);

			await expect(
				runDigestAll({
					sessionsDir,
					roundsDir,
					indexPath,
					apiKey: "key",
					fetchImpl: embeddingFetch([
						[1, 0],
						[0, 1],
						[2, 3],
					]),
					stdout: logger().out,
				}),
			).resolves.toBe(0);

			const toolIndexPath = toolIndexPathForRoundsDir(roundsDir);
			const rows = loadToolIndex(toolIndexPath);
			expect(rows).toHaveLength(1);
			expect(rows[0]).toMatchObject({ toolIndex: 0, toolName: "bash" });
			expect(rows[0].searchableText).toContain("curl https://api.github.com/repos/vhallac/semblr");
		});

		it("does not duplicate tool index rows when a round is re-processed (model mismatch)", async () => {
			const root = tmpDir();
			const sessionsDir = path.join(root, "sessions");
			const sDir = path.join(sessionsDir, "--test");
			fs.mkdirSync(sDir, { recursive: true });
			const roundsDir = path.join(root, "rounds");
			fs.mkdirSync(roundsDir, { recursive: true });
			const indexPath = path.join(roundsDir, "index.csv");

			writeSessionWithToolCall(
				path.join(sDir, "session.jsonl"),
				"Please curl the repo",
				"Done, fetched the repo info.",
			);

			await runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([
					[1, 0],
					[0, 1],
					[2, 3],
				]),
				stdout: logger().out,
			});

			const toolIndexPath = toolIndexPathForRoundsDir(roundsDir);
			expect(loadToolIndex(toolIndexPath)).toHaveLength(1);

			// Force a model mismatch so the round is re-processed on the next run.
			const lines = fs.readFileSync(indexPath, "utf-8").trim().split("\n");
			fs.writeFileSync(indexPath, `${lines.map((l) => `${l},old-model`).join("\n")}\n`);

			await runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: embeddingFetch([
					[1, 0],
					[0, 1],
					[2, 3],
				]),
				stdout: logger().out,
			});

			expect(loadToolIndex(toolIndexPath)).toHaveLength(1);
		});

		it("--tools-only rebuilds the tool index from existing round files without embedding", async () => {
			const roundsDir = tmpDir();
			fs.writeFileSync(
				path.join(roundsDir, "abc.json"),
				JSON.stringify({
					userPrompt: "hi",
					responseSequence: "hi",
					toolCalls: [{ index: 0, name: "bash", arguments: JSON.stringify({ command: "curl x" }) }],
				}),
			);
			const logs = logger();
			const fetchImpl = vi.fn(async () => new Response("should not be called")) as typeof fetch;

			await expect(
				runDigestAll({
					roundsDir,
					toolsOnly: true,
					fetchImpl,
					stdout: logs.out,
					stderr: logs.err,
				}),
			).resolves.toBe(0);

			expect(fetchImpl).not.toHaveBeenCalled();
			const toolIndexPath = toolIndexPathForRoundsDir(roundsDir);
			expect(loadToolIndex(toolIndexPath)).toEqual([
				{ hash: "abc.json", toolIndex: 0, toolName: "bash", searchableText: "bash curl x" },
			]);
			expect(logs.stdout.join("\n")).toContain("Rebuilt tool index: 1 rows across 1 rounds");
		});

		it("--tools-only handles a missing rounds directory", async () => {
			const roundsDir = path.join(tmpDir(), "nonexistent");
			const logs = logger();

			await expect(runDigestAll({ roundsDir, toolsOnly: true, stdout: logs.out })).resolves.toBe(0);
			expect(logs.stdout.join("\n")).toContain("Rebuilt tool index: 0 rows across 0 rounds");
		});
	});
});
