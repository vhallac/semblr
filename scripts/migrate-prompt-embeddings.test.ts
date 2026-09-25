import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { encodeVectorIndexLine, loadVectorIndex, makeAssumedStamp, readIndexLines } from "../lib/index-io.ts";
import { hashEmbeddingInput } from "../lib/round-capture.ts";
import { isMainModule, runPromptEmbeddingsMigration } from "./migrate-prompt-embeddings.ts";

let tempDir = "";

beforeEach(() => {
	tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-migrate-prompt-embeddings-"));
});

afterEach(() => {
	fs.rmSync(tempDir, { recursive: true, force: true });
});

const NOISY_PROMPT = `explain\n${"-".repeat(500)}\nend`;
const CLEANED_NOISY_PROMPT = "explain\n[REPEAT: '-' × 500]\nend";
const PLAIN_PROMPT = "What is the capital of France?";
const RESPONSE = "The answer is 42.";

function logger() {
	const stdout: string[] = [];
	const stderr: string[] = [];
	return {
		stdout,
		stderr,
		out: { log: (line: string) => stdout.push(line) },
		err: {
			error: (line: string) => stderr.push(line),
			warn: (line: string) => stderr.push(line),
		},
	};
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

function writeRound(
	roundFile: string,
	data: {
		userPrompt: string;
		responseSequence?: string;
		promptEmbedding?: number[];
		/** Present (string) marks a post-#107-F3 capture; absent = legacy. */
		promptVecHash?: string;
	},
): void {
	fs.writeFileSync(path.join(tempDir, roundFile), JSON.stringify({ toolCalls: [], ...data }, null, 2));
}

function writeIndex(lines: string[]): void {
	fs.writeFileSync(path.join(tempDir, "index.csv"), `${lines.join("\n")}\n`);
}

function baseOptions(stdout: Pick<typeof console, "log">, extra: Record<string, unknown> = {}) {
	return {
		roundsDir: tempDir,
		indexPath: path.join(tempDir, "index.csv"),
		apiKey: "key",
		configDeps: { cwd: tempDir, agentDir: path.join(tempDir, "agent"), env: {} },
		...extra,
		stdout,
	};
}

describe("migrate-prompt-embeddings script", () => {
	it("re-embeds a legacy noisy round's prompt and combined vector, stamping assumed- provenance", async () => {
		const roundFile = "legacy-noisy.json";
		writeRound(roundFile, { userPrompt: NOISY_PROMPT, responseSequence: RESPONSE, promptEmbedding: [0.5] });
		writeIndex([
			encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model"),
			encodeVectorIndexLine([8, 8], `${roundFile}:response`, "old-model"),
		]);
		const requests: unknown[] = [];
		const logs = logger();
		const fetchImpl = embeddingFetch(
			[
				[3, 4], // prompt
				[0, 5], // combined
			],
			requests,
		);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out), fetchImpl, stderr: logs.err }),
		).resolves.toBe(0);

		// Prompt row: normalized fresh vector, current model, assumed- stamp over the
		// cleaned input (legacy capture — original capture input unverifiable, #107 F1)
		const index = loadVectorIndex(path.join(tempDir, "index.csv"));
		expect(index).toEqual([
			{
				vector: [0.6, 0.8],
				filePath: `${roundFile}:prompt`,
				model: "openai/text-embedding-3-small",
				embeddingInputHash: makeAssumedStamp(hashEmbeddingInput(CLEANED_NOISY_PROMPT)),
			},
			{ vector: [8, 8], filePath: `${roundFile}:response`, model: "old-model" },
		]);

		// Two embedding calls: cleaned prompt, then combined (cleaned prompt + response)
		expect(requests).toHaveLength(2);
		expect((requests[0] as any).body.input).toBe(CLEANED_NOISY_PROMPT);
		expect((requests[1] as any).body.input).toBe(`${CLEANED_NOISY_PROMPT}\n\n${RESPONSE}`);

		// round.json combined vector updated with the un-normalized vector
		const round = JSON.parse(fs.readFileSync(path.join(tempDir, roundFile), "utf-8")) as {
			promptEmbedding?: number[];
		};
		expect(round.promptEmbedding).toEqual([0, 5]);

		expect(logs.stdout.join("\n")).toContain("(legacy)");
		expect(logs.stdout.join("\n")).toContain(
			"1 rounds re-embedded (1 combined vectors), 1 legacy rows re-stamped with assumed- provenance",
		);
	});

	it("re-embeds a legacy no-op round with an assumed- stamp; the next run is a no-op", async () => {
		// The old stamp-only branch trusted a matching plain stamp on legacy rows and
		// skipped the API call — but the stored vector was computed from the capture
		// hook's augmented prompt, so the stamp asserted provenance it could not know
		// (#107 F1). The sweep now re-embeds even when cleanup is a no-op, stamps
		// assumed-, and a following run recognizes the marker (idempotent, 0 API calls).
		const roundFile = "legacy-plain.json";
		writeRound(roundFile, { userPrompt: PLAIN_PROMPT, responseSequence: RESPONSE, promptEmbedding: [0.5] });
		const legacyLine = encodeVectorIndexLine([9, 9], `${roundFile}:prompt`);
		writeIndex([legacyLine]);
		const requests: unknown[] = [];
		const logs = logger();
		const fetchImpl = embeddingFetch(
			[
				[3, 4], // prompt
				[0, 5], // combined
			],
			requests,
		);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out), fetchImpl, stderr: logs.err }),
		).resolves.toBe(0);

		// Re-embedded (not stamped in place): fresh vector, current model, assumed- stamp
		expect(loadVectorIndex(path.join(tempDir, "index.csv"))).toEqual([
			{
				vector: [0.6, 0.8],
				filePath: `${roundFile}:prompt`,
				model: "openai/text-embedding-3-small",
				embeddingInputHash: makeAssumedStamp(hashEmbeddingInput(PLAIN_PROMPT)),
			},
		]);
		// Cleanup is a no-op for this prompt, but the legacy vector is unverified →
		// the sweep still pays for prompt + combined embeddings.
		expect(requests).toHaveLength(2);
		expect((requests[0] as any).body.input).toBe(PLAIN_PROMPT);

		// round.json combined vector updated alongside the :prompt row
		const round = JSON.parse(fs.readFileSync(path.join(tempDir, roundFile), "utf-8")) as {
			promptEmbedding?: number[];
		};
		expect(round.promptEmbedding).toEqual([0, 5]);
		expect(logs.stdout.join("\n")).toContain("(legacy)");

		// Second run: the assumed- stamp matches the current derivation → current
		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		const secondLogs = logger();
		await expect(
			runPromptEmbeddingsMigration({
				...baseOptions(secondLogs.out),
				fetchImpl: secondFetch,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);
		expect(secondFetch).not.toHaveBeenCalled();
		expect(secondLogs.stdout.join("\n")).toContain("All :prompt rows are current. Nothing to migrate.");
	});

	it("skips a post-#107-F3 row whose plain stamp matches the recomputed input", async () => {
		const roundFile = "current.json";
		writeRound(roundFile, {
			userPrompt: PLAIN_PROMPT,
			responseSequence: RESPONSE,
			promptEmbedding: [0.5],
			promptVecHash: hashEmbeddingInput(PLAIN_PROMPT),
		});
		writeIndex([
			encodeVectorIndexLine(
				[9, 9],
				`${roundFile}:prompt`,
				"openai/text-embedding-3-small",
				hashEmbeddingInput(PLAIN_PROMPT),
			),
		]);
		const requests: unknown[] = [];
		const logs = logger();
		const fetchImpl = embeddingFetch([], requests);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out), fetchImpl, stderr: logs.err }),
		).resolves.toBe(0);

		expect(fetchImpl).not.toHaveBeenCalled();
		expect(readIndexLines(path.join(tempDir, "index.csv"))).toEqual([
			encodeVectorIndexLine(
				[9, 9],
				`${roundFile}:prompt`,
				"openai/text-embedding-3-small",
				hashEmbeddingInput(PLAIN_PROMPT),
			),
		]);
		expect(logs.stdout.join("\n")).toContain("All :prompt rows are current. Nothing to migrate.");
	});

	it("re-embeds a legacy row even when its plain stamp matches the current derivation (#107 F1)", async () => {
		// The pre-F1 stamp-only branch wrote hash(clean(raw)) over rows whose vectors
		// came from the hook's augmented prompt; such a false stamp is byte-identical
		// to a digest-corrected one, so a plain stamp on a legacy row is untrusted.
		// The row must re-enter the sweep and gain an assumed- stamp (otherwise the
		// false provenance would freeze the row as "current" forever).
		const roundFile = "false-stamp.json";
		writeRound(roundFile, { userPrompt: PLAIN_PROMPT, responseSequence: RESPONSE });
		writeIndex([
			encodeVectorIndexLine(
				[9, 9],
				`${roundFile}:prompt`,
				"openai/text-embedding-3-small",
				hashEmbeddingInput(PLAIN_PROMPT), // exactly what stamp-only wrote
			),
		]);
		const requests: unknown[] = [];
		const logs = logger();
		const fetchImpl = embeddingFetch([[3, 4]], requests);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out), fetchImpl, stderr: logs.err }),
		).resolves.toBe(0);

		expect(requests).toHaveLength(1); // re-embedded despite the "matching" stamp
		expect((requests[0] as any).body.input).toBe(PLAIN_PROMPT);
		expect(loadVectorIndex(path.join(tempDir, "index.csv"))).toEqual([
			{
				vector: [0.6, 0.8],
				filePath: `${roundFile}:prompt`,
				model: "openai/text-embedding-3-small",
				embeddingInputHash: makeAssumedStamp(hashEmbeddingInput(PLAIN_PROMPT)),
			},
		]);
		expect(logs.stdout.join("\n")).toContain("(legacy)");
	});

	it("re-embeds a post-#107-F3 stamped row when the stored hash no longer matches (heuristic change)", async () => {
		// Post-F3 capture: the plain stamp was written over a verifiable input, so a
		// bare-hash mismatch means the convention changed → re-embed and restamp PLAIN
		// (no assumed- marker — provenance is verifiable for this round).
		const roundFile = "stale-stamp.json";
		writeRound(roundFile, {
			userPrompt: PLAIN_PROMPT,
			responseSequence: RESPONSE,
			promptEmbedding: [0.5],
			promptVecHash: "stale-hash-value",
		});
		writeIndex([encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model", "stale-hash-value")]);
		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch(
			[
				[3, 4],
				[0, 5],
			],
			requests,
		);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logger().out), fetchImpl, stderr: logger().err }),
		).resolves.toBe(0);

		expect((requests[0] as any).body.input).toBe(PLAIN_PROMPT);
		expect((requests[1] as any).body.input).toBe(`${PLAIN_PROMPT}\n\n${RESPONSE}`);
		expect(loadVectorIndex(path.join(tempDir, "index.csv"))).toEqual([
			{
				vector: [0.6, 0.8],
				filePath: `${roundFile}:prompt`,
				model: "openai/text-embedding-3-small",
				embeddingInputHash: hashEmbeddingInput(PLAIN_PROMPT),
			},
		]);
	});

	it("re-embeds over-budget prompts when the clip changes the embedding input (#107 F4)", async () => {
		// A row stamped under the pre-F4 convention (hash over the full cleaned text,
		// no clip) no longer matches once the clip fires on a prompt cleanup cannot
		// shrink (varied prose — no fences, JSON, or repeat runs): the sweep must
		// re-embed over the exact clipped input and restamp (assumed- — the row's
		// round file predates #107 F3, so the pre-F4 stamp is also unverified).
		const longPrompt = Array.from({ length: 1500 }, (_, i) => `word${i}`).join(" "); // ~10.5K chars
		const roundFile = "over-budget.json";
		writeRound(roundFile, { userPrompt: longPrompt, responseSequence: RESPONSE, promptEmbedding: [0.5] });
		writeIndex([encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model", hashEmbeddingInput(longPrompt))]);
		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch(
			[
				[3, 4],
				[0, 5],
			],
			requests,
		);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logger().out), fetchImpl, stderr: logger().err }),
		).resolves.toBe(0);

		const clipped = longPrompt.slice(0, 8000); // default embeddingMaxTokens
		expect((requests[0] as any).body.input).toBe(clipped);
		expect((requests[1] as any).body.input).toBe(`${clipped}\n\n${RESPONSE}`);
		expect(loadVectorIndex(path.join(tempDir, "index.csv"))).toEqual([
			{
				vector: [0.6, 0.8],
				filePath: `${roundFile}:prompt`,
				model: "openai/text-embedding-3-small",
				embeddingInputHash: makeAssumedStamp(hashEmbeddingInput(clipped)),
			},
		]);
	});

	it("re-embeds only the prompt row for digest-shaped rounds without a combined vector", async () => {
		const roundFile = "digest-shaped.json";
		writeRound(roundFile, { userPrompt: NOISY_PROMPT, responseSequence: RESPONSE });
		writeIndex([encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model")]);
		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch([[3, 4]], requests);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logger().out), fetchImpl, stderr: logger().err }),
		).resolves.toBe(0);

		expect(requests).toHaveLength(1);
		expect((requests[0] as any).body.input).toBe(CLEANED_NOISY_PROMPT);
		expect(loadVectorIndex(path.join(tempDir, "index.csv"))).toEqual([
			{
				vector: [0.6, 0.8],
				filePath: `${roundFile}:prompt`,
				model: "openai/text-embedding-3-small",
				embeddingInputHash: makeAssumedStamp(hashEmbeddingInput(CLEANED_NOISY_PROMPT)),
			},
		]);
		const round = JSON.parse(fs.readFileSync(path.join(tempDir, roundFile), "utf-8")) as {
			promptEmbedding?: number[];
		};
		expect(round.promptEmbedding).toBeUndefined();
	});

	it("supports --dry-run without writing anything", async () => {
		const roundFile = "dry.json";
		writeRound(roundFile, { userPrompt: NOISY_PROMPT, responseSequence: RESPONSE, promptEmbedding: [0.5] });
		writeIndex([encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model")]);
		const logs = logger();
		const fetchImpl = embeddingFetch([], []);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out), fetchImpl, dryRun: true, stderr: logs.err }),
		).resolves.toBe(0);

		expect(fetchImpl).not.toHaveBeenCalled();
		expect(readIndexLines(path.join(tempDir, "index.csv"))).toEqual([
			encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model"),
		]);
		expect(logs.stdout.join("\n")).toContain("Dry-run mode: no changes written.");
		expect(logs.stdout.join("\n")).toContain("[legacy]");
	});

	it("creates a backup of the index when --backup is set", async () => {
		const roundFile = "backed-up.json";
		writeRound(roundFile, { userPrompt: NOISY_PROMPT, responseSequence: RESPONSE });
		writeIndex([encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model")]);
		const fetchImpl = embeddingFetch([[3, 4]]);

		await expect(
			runPromptEmbeddingsMigration({
				...baseOptions(logger().out),
				fetchImpl,
				backup: true,
				stderr: logger().err,
			}),
		).resolves.toBe(0);

		const backups = fs.readdirSync(tempDir).filter((f) => f.startsWith("index.csv.bak."));
		expect(backups).toHaveLength(1);
		expect(fs.readFileSync(path.join(tempDir, backups[0]), "utf-8")).toBe(
			`${encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model")}\n`,
		);
	});

	it("leaves :prompt rows without a round file untouched and warns", async () => {
		const orphanLine = encodeVectorIndexLine([9, 9], "missing.json:prompt", "old-model");
		writeIndex([orphanLine]);
		const logs = logger();
		const fetchImpl = embeddingFetch([]);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out), fetchImpl, stderr: logs.err }),
		).resolves.toBe(0);

		expect(fetchImpl).not.toHaveBeenCalled();
		expect(readIndexLines(path.join(tempDir, "index.csv"))).toEqual([orphanLine]);
		expect(logs.stdout.join("\n")).toContain("without a round file (skipped)");
	});

	it("fails without an API key when there is something to re-embed, leaving the index untouched", async () => {
		const roundFile = "needs-key.json";
		writeRound(roundFile, { userPrompt: NOISY_PROMPT, responseSequence: RESPONSE });
		const legacyLine = encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model");
		writeIndex([legacyLine]);
		const logs = logger();

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out, { apiKey: "" }), stderr: logs.err }),
		).resolves.toBe(1);

		expect(readIndexLines(path.join(tempDir, "index.csv"))).toEqual([legacyLine]);
		expect(logs.stderr.join("\n")).toContain("Embedding API key required");
	});

	it("is idempotent: a second run after migration reports nothing to do", async () => {
		const roundFile = "twice.json";
		writeRound(roundFile, { userPrompt: NOISY_PROMPT, responseSequence: RESPONSE, promptEmbedding: [0.5] });
		writeIndex([encodeVectorIndexLine([9, 9], `${roundFile}:prompt`, "old-model")]);
		const fetchImpl = embeddingFetch([
			[3, 4],
			[0, 5],
		]);

		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logger().out), fetchImpl, stderr: logger().err }),
		).resolves.toBe(0);

		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		const logs = logger();
		await expect(
			runPromptEmbeddingsMigration({ ...baseOptions(logs.out), fetchImpl: secondFetch, stderr: logs.err }),
		).resolves.toBe(0);

		expect(secondFetch).not.toHaveBeenCalled();
		expect(logs.stdout.join("\n")).toContain("All :prompt rows are current. Nothing to migrate.");
	});
});

describe("isMainModule", () => {
	it("detects the script entry point", () => {
		expect(isMainModule("file:///tmp/migrate-prompt-embeddings.ts", "/tmp/migrate-prompt-embeddings.ts")).toBe(true);
		expect(isMainModule("file:///tmp/migrate-prompt-embeddings.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/migrate-prompt-embeddings.ts", "")).toBe(false);
	});
});
