import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRoundFilePath } from "./hash.ts";
import { parsePiSessionJsonl } from "./pi-session.ts";
import {
	buildAgentEndEmbeddingTexts,
	buildAgentEndRoundFile,
	buildPromptEmbeddingInput,
	embeddingMaxTokensToResponseBytes,
} from "./round-capture.ts";
import type { ToolCallDetail } from "./round-data.ts";
import {
	backfillMissingRounds,
	buildBackfillCandidates,
	embedRecoveredRounds,
	extractCheckpointSummary,
	findMissingRounds,
	indexRecoveredRounds,
	isBackfillStartReason,
	isSessionTailClosed,
	listBackfillCandidates,
	planStartupEmbedding,
	type RecoveredRoundLike,
} from "./session-backfill.ts";

function writeSessionFile(dir: string, lines: object[]): string {
	const file = path.join(dir, "session.jsonl");
	fs.writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	return file;
}

function userMsg(text: string, id = "u1", timestamp = 1000) {
	return { type: "message", id, message: { role: "user", content: [{ type: "text", text }], timestamp } };
}

function assistantMsg(text: string) {
	return { type: "message", message: { role: "assistant", content: [{ type: "text", text }] } };
}

describe("session-backfill", () => {
	let tmp: string;
	beforeEach(() => {
		tmp = fs.mkdtempSync(path.join(os.tmpdir(), "backfill-"));
	});
	afterEach(() => {
		fs.rmSync(tmp, { recursive: true, force: true });
	});

	it("recovers a round missing from the rounds dir (simulated process death)", () => {
		const sessionFile = writeSessionFile(tmp, [userMsg("hello world"), assistantMsg("hi there")]);
		const roundsDir = path.join(tmp, "rounds");
		const expected = createRoundFilePath("hello world", "hi there", []);

		const { missing, scanned } = findMissingRounds(sessionFile, roundsDir);
		expect(scanned).toBe(1);
		expect(missing).toHaveLength(1);
		expect(missing[0].fileName).toBe(expected);

		const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		expect(outcome.recoveredFiles).toEqual([expected]);
		expect(outcome.scanned).toBe(1);

		const written = JSON.parse(fs.readFileSync(path.join(roundsDir, expected), "utf-8"));
		expect(written.userPrompt).toBe("hello world");
		expect(written.responseSequence).toBe("hi there");
		expect(written.promptEmbedding).toBeUndefined();
		expect(written.recovered).toBe(true);
	});

	it("is idempotent — existing round files are skipped", () => {
		const sessionFile = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
		const roundsDir = path.join(tmp, "rounds");
		backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		const first = fs.readFileSync(path.join(roundsDir, createRoundFilePath("q", "a", [])), "utf-8");

		const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		// Startup-cost early exit: with every round on disk, the last-round check
		// short-circuits before the full scan.
		expect(outcome.recoveredFiles).toEqual([]);
		expect(outcome.scanned).toBe(0);
		expect(outcome.skippedComplete).toBe(true);
		expect(fs.readFileSync(path.join(roundsDir, createRoundFilePath("q", "a", [])), "utf-8")).toBe(first);
	});

	it("recovers multiple lost rounds with tool calls and skips promptless rounds", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("first question", "u1"),
			assistantMsg("first answer"),
			userMsg("second question", "u2"),
			{
				type: "message",
				message: {
					role: "assistant",
					content: [{ type: "toolCall", name: "read", arguments: { p: "x" }, id: "t1" }],
				},
			},
			{
				type: "message",
				message: {
					role: "toolResult",
					toolName: "read",
					toolCallId: "t1",
					content: [{ type: "text", text: "file content" }],
				},
			},
			assistantMsg("second answer"),
		]);
		const roundsDir = path.join(tmp, "rounds");
		const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		expect(outcome.recoveredFiles).toHaveLength(2);
		const toolRound = JSON.parse(fs.readFileSync(path.join(roundsDir, outcome.recoveredFiles[1]), "utf-8"));
		expect(toolRound.toolCalls).toHaveLength(1);
		expect(toolRound.toolCalls[0].result_summary).toContain("file content");
		expect(toolRound.toolCallCount).toBe(1);
	});

	it("tool result text preserves trailing whitespace like the live capture path (hash parity)", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("q", "u1"),
			{
				type: "message",
				message: {
					role: "assistant",
					content: [{ type: "toolCall", name: "bash", arguments: {}, id: "t1" }],
				},
			},
			{
				type: "message",
				message: {
					role: "toolResult",
					toolName: "bash",
					toolCallId: "t1",
					content: [{ type: "text", text: "output line\n" }],
				},
			},
			assistantMsg("a"),
		]);
		const roundsDir = path.join(tmp, "rounds");
		backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		const files = fs.readdirSync(roundsDir);
		expect(files).toHaveLength(1);
		const round = JSON.parse(fs.readFileSync(path.join(roundsDir, files[0]), "utf-8"));
		// The live path (round-capture.ts extractText) does not trim; the backfill
		// path must match or the content hash diverges and rounds get re-recovered.
		expect(round.toolCalls[0].result_full).toBe("output line\n");
	});

	it("rounds with empty user prompt are not counted as scanned", () => {
		const sessionFile = writeSessionFile(tmp, [userMsg("   ", "u1"), assistantMsg("response")]);
		const roundsDir = path.join(tmp, "rounds");
		const { missing, scanned } = findMissingRounds(sessionFile, roundsDir);
		expect(scanned).toBe(0);
		expect(missing).toHaveLength(0);
	});

	it("respects fsImpl injection for existsSync", () => {
		const sessionFile = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
		const roundsDir = path.join(tmp, "rounds");
		// Pretend every file exists — nothing should be recovered.
		const outcome = backfillMissingRounds(sessionFile, roundsDir, {
			existsSync: () => true,
			mkdirSync: fs.mkdirSync,
			writeFileSync: fs.writeFileSync,
			statSync: fs.statSync,
		});
		expect(outcome.recoveredFiles).toEqual([]);
	});

	it("defers a live source session whose tail round is missing (retried later) — F2", () => {
		const sessionFile = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
		const roundsDir = path.join(tmp, "rounds");
		// Fresh file — mtime is now, inside the default live window; tail is missing.
		const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBe(1);
		expect(outcome.recoveredFiles).toEqual([]);
		expect(fs.existsSync(roundsDir)).toBe(false);
	});

	it("defers only the live file and recovers from stale files in the same run — F2", () => {
		const roundsDir = path.join(tmp, "rounds");
		const stale = writeSessionFile(tmp, [userMsg("old q"), assistantMsg("old a")]);
		fs.utimesSync(stale, new Date(0), new Date(0));
		const live = path.join(tmp, "live.jsonl");
		fs.writeFileSync(live, `${[userMsg("new q"), assistantMsg("new a")].map((l) => JSON.stringify(l)).join("\n")}\n`);
		const outcome = backfillMissingRounds([live, stale], roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBe(1);
		expect(outcome.recoveredFiles).toEqual([createRoundFilePath("old q", "old a", [])]);
	});

	it("recovers a live-looking file whose tail is already on disk (complete beats live) — F2", () => {
		const roundsDir = path.join(tmp, "rounds");
		const complete = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
		const rounds = backfillMissingRounds(complete, roundsDir, undefined, { liveWindowMs: 0 });
		expect(rounds.recoveredFiles).toEqual([createRoundFilePath("q", "a", [])]);
		// F2 round 3: liveness now gates BEFORE the parse — a live-looking file
		// defers unconditionally (review round 3 finding F2). Completeness is
		// discovered via the early exit once the file goes stale.
		const outcome = backfillMissingRounds(complete, roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBe(1);
		expect(outcome.recoveredFiles).toEqual([]);
		// Once stale, the early exit reports skippedComplete without re-writing.
		// Pin the mtime so the 1ms live window can't flake on clock resolution.
		fs.utimesSync(complete, new Date(0), new Date(0));
		const stale = backfillMissingRounds(complete, roundsDir, undefined, { nowMs: Date.now(), liveWindowMs: 1 });
		expect(stale.skippedComplete).toBe(true);
		expect(stale.recoveredFiles).toEqual([]);
	});

	it("defers a live file before parsing it at all — F2 round 3", () => {
		const roundsDir = path.join(tmp, "rounds");
		const live = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
		// Garbage content would previously abort the whole run once parsed; with
		// the liveness gate ahead of the parse it is simply deferred.
		fs.writeFileSync(live, "not json at all\n");
		const outcome = backfillMissingRounds(live, roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBe(1);
		expect(outcome.recoveredFiles).toEqual([]);
	});

	it("defers a stale file whose tail turn is still open (toolUse stopReason) — F2 round 3", () => {
		const roundsDir = path.join(tmp, "rounds");
		const sessionFile = writeSessionFile(tmp, [
			userMsg("q"),
			{
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: "partial" }], stopReason: "toolUse" },
			},
		]);
		fs.utimesSync(sessionFile, new Date(0), new Date(0)); // stale mtime — closedness must still gate
		const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBe(1);
		expect(outcome.recoveredFiles).toEqual([]);
		expect(fs.existsSync(roundsDir)).toBe(false);
	});

	it("recovers a stale file whose tail turn ended (terminal stopReason) — F2 round 3", () => {
		const roundsDir = path.join(tmp, "rounds");
		const sessionFile = writeSessionFile(tmp, [
			userMsg("q"),
			{
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: "done" }], stopReason: "stop" },
			},
		]);
		fs.utimesSync(sessionFile, new Date(0), new Date(0));
		const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBeUndefined();
		expect(outcome.recoveredFiles).toEqual([createRoundFilePath("q", "done", [])]);
	});

	it("isSessionTailClosed: no assistant message → closed (nothing in flight); toolUse → open; stop → closed", () => {
		expect(isSessionTailClosed(JSON.stringify(userMsg("q")))).toBe(true);
		expect(isSessionTailClosed([userMsg("q"), assistantMsg("a")].map((l) => JSON.stringify(l)).join("\n"))).toBe(
			true,
		);
		expect(
			isSessionTailClosed(
				[userMsg("q"), { type: "message", message: { role: "assistant", content: [], stopReason: "toolUse" } }]
					.map((l) => JSON.stringify(l))
					.join("\n"),
			),
		).toBe(false);
		expect(isSessionTailClosed("{truncated json")).toBe(true); // corrupt lines skipped; nothing in flight
	});

	it("backfills a stale source session whose mtime is older than the live window — F3", () => {
		const sessionFile = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
		const roundsDir = path.join(tmp, "rounds");
		fs.utimesSync(sessionFile, new Date(0), new Date(0));
		const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBeUndefined();
		expect(outcome.recoveredFiles).toEqual([createRoundFilePath("q", "a", [])]);
	});

	describe("embedRecoveredRounds (F4)", () => {
		it("embeds prompt, clipped response, and combined text; stores combined as promptEmbedding", async () => {
			const sessionFile = writeSessionFile(tmp, [
				userMsg(
					"please tell me about the long war and its many causes and consequences for the border provinces and their people",
				),
				assistantMsg("a long answer"),
			]);
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
			const fileName = outcome.recoveredFiles[0];
			const rows: Array<[string, number[]]> = [];
			const result = await embedRecoveredRounds(outcome.recoveredFiles, roundsDir, {
				embed: (text) => Promise.resolve([text.length, 1]),
				appendIndexRow: (label, vec) => rows.push([label, vec]),
				writeRoundEmbedding: (name, vec) => {
					const p = path.join(roundsDir, name);
					const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
					existing.promptEmbedding = vec;
					fs.writeFileSync(p, JSON.stringify(existing, null, 2));
				},
			});
			expect(result.embedded).toEqual([fileName]);
			expect(result.errors).toEqual([]);
			// :prompt and :response rows appended; combined vector stored on the round
			expect(rows.map(([label]) => label)).toEqual([`${fileName}:prompt`, `${fileName}:response`]);
			const written = JSON.parse(fs.readFileSync(path.join(roundsDir, fileName), "utf-8"));
			expect(written.promptEmbedding).toBeDefined();
			// live scale convention: rows are normalized, the stored promptEmbedding
			// is the raw combined vector (embed returns [len, 3], so rows have magnitude √2)
			const responseRowVec = rows.find(([label]) => label === `${fileName}:response`)![1];
			const promptRowVec = rows.find(([label]) => label === `${fileName}:prompt`)![1];
			const rowMag = Math.sqrt(promptRowVec.reduce((s: number, x: number) => s + x * x, 0));
			expect(rowMag).toBeCloseTo(1, 5);
			expect(written.promptEmbedding).not.toEqual(responseRowVec);
		});

		it("skips rounds that already have a promptEmbedding, and reports per-round errors", async () => {
			const sessionFile = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
			const fileName = outcome.recoveredFiles[0];
			// Pre-embed the round: second queue pass must skip it.
			const first = await embedRecoveredRounds(outcome.recoveredFiles, roundsDir, {
				embed: () => Promise.resolve([1]),
				appendIndexRow: () => {},
				writeRoundEmbedding: (name, vec) => {
					const p = path.join(roundsDir, name);
					const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
					existing.promptEmbedding = vec;
					fs.writeFileSync(p, JSON.stringify(existing, null, 2));
				},
			});
			expect(first.embedded).toEqual([fileName]);
			const second = await embedRecoveredRounds(outcome.recoveredFiles, roundsDir, {
				embed: () => Promise.resolve([1]),
				appendIndexRow: () => {},
				writeRoundEmbedding: () => {},
			});
			expect(second.embedded).toEqual([]);
		});

		it("F4 parity: prompt goes through buildPromptEmbeddingInput cleanup and the :prompt row carries the hash stamp", async () => {
			const sessionFile = writeSessionFile(tmp, [
				userMsg("explain this\n```python\n" + "x = 1\n".repeat(200) + "```"),
				// F1 (PR !131): a prompt-only session tail is no longer recovered —
				// the fixture needs a response to produce a fileable round.
				assistantMsg("the answer"),
			]);
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
			const fileName = outcome.recoveredFiles[0];
			const embedded: string[] = [];
			const rowHashes = new Map<string, string | undefined>();
			const result = await embedRecoveredRounds(
				outcome.recoveredFiles,
				roundsDir,
				{
					embed: (text) => {
						embedded.push(text);
						return Promise.resolve([text.length, 3]);
					},
					appendIndexRow: (label, _vec, hash) => rowHashes.set(label, hash),
					writeRoundEmbedding: () => {},
				},
				// live-parity derivation (same convention as semblr.ts agent_end),
				// with a small clip so the collapsed fence shrinks the input
				{ promptNoiseOptions: { fenceMaxChars: 4, jsonMaxChars: 0, repeatMaxChars: 0 }, promptMaxTokens: 8000 },
			);
			expect(result.embedded).toEqual([fileName]);
			// the prompt embedding input is the cleaned prompt, not the raw round text
			const raw = JSON.parse(fs.readFileSync(path.join(roundsDir, fileName), "utf-8")).userPrompt as string;
			const { text: expected, hash } = buildPromptEmbeddingInput(
				raw,
				{ fenceMaxChars: 4, jsonMaxChars: 0, repeatMaxChars: 0 },
				8000,
			);
			expect(embedded[0]).toBe(expected);
			expect(expected.length).toBeLessThan(raw.length);
			// the :prompt row is hash-stamped (same domain as live rows); :response is not
			expect(rowHashes.get(`${fileName}:prompt`)).toBe(hash);
			expect(rowHashes.get(`${fileName}:response`)).toBeUndefined();
		});

		it("F6 parity: a short recovered prompt drops the :prompt row and stores the normalized response vector (shared-core policy)", async () => {
			const sessionFile = writeSessionFile(tmp, [userMsg("continue"), assistantMsg("the walls hold for now")]);
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
			const fileName = outcome.recoveredFiles[0];
			const rows: Array<[string, number[]]> = [];
			const embedded: string[] = [];
			const result = await embedRecoveredRounds(outcome.recoveredFiles, roundsDir, {
				embed: (text) => {
					embedded.push(text);
					return Promise.resolve([text.length, 1]);
				},
				appendIndexRow: (label, vec) => rows.push([label, vec]),
				writeRoundEmbedding: (name, vec) => {
					const p = path.join(roundsDir, name);
					const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
					existing.promptEmbedding = vec;
					fs.writeFileSync(p, JSON.stringify(existing, null, 2));
				},
			});
			expect(result.embedded).toEqual([fileName]);
			// the prompt embedding is dropped entirely — same policy as live agent_end
			expect(embedded).toHaveLength(1);
			expect(rows.map(([label]) => label)).toEqual([`${fileName}:response`]);
			// scale convention: the round stores the normalized response vector
			const written = JSON.parse(fs.readFileSync(path.join(roundsDir, fileName), "utf-8"));
			const mag = Math.sqrt(written.promptEmbedding.reduce((s: number, x: number) => s + x * x, 0));
			expect(mag).toBeCloseTo(1, 5);
		});

		it("F3 parity: response clip honors the caller-supplied maxResponseBytes budget (non-default embeddingMaxTokens)", async () => {
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(
				writeSessionFile(tmp, [userMsg("q"), assistantMsg("x".repeat(1000))]),
				roundsDir,
				undefined,
				{ liveWindowMs: 0 },
			);
			const fileName = outcome.recoveredFiles[0];
			// non-default config: embeddingMaxTokens 100 → 300-byte response budget
			const budget = embeddingMaxTokensToResponseBytes(100);
			const responseInputs: string[] = [];
			const result = await embedRecoveredRounds(
				outcome.recoveredFiles,
				roundsDir,
				{
					embed: (text) => {
						responseInputs.push(text);
						return Promise.resolve([text.length, 1]);
					},
					appendIndexRow: () => {},
					writeRoundEmbedding: () => {},
				},
				{ maxResponseBytes: budget },
			);
			expect(result.embedded).toEqual([fileName]);
			// the :response embedding input equals what the live agent_end path would
			// clip to with the same configured budget (live/recovery parity)
			const raw = JSON.parse(fs.readFileSync(path.join(roundsDir, fileName), "utf-8")).responseSequence as string;
			const { clippedResponse } = buildAgentEndEmbeddingTexts("", raw, budget);
			expect(Buffer.byteLength(clippedResponse, "utf-8")).toBe(budget);
			expect(responseInputs).toContain(clippedResponse);
			// and it is strictly shorter than the default-budget clip, proving the
			// non-default budget was honored rather than the 8000-token default
			const { clippedResponse: defaultClip } = buildAgentEndEmbeddingTexts("", raw);
			expect(clippedResponse.length).toBeLessThan(defaultClip.length);
		});

		it("F5: re-run after a crash between appends and embedding write does not duplicate rows", async () => {
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(
				writeSessionFile(tmp, [userMsg("crash q", "u2"), assistantMsg("crash answer")]),
				roundsDir,
				undefined,
				{ liveWindowMs: 0 },
			);
			const crashFile = outcome.recoveredFiles[0];
			const preexisting = new Set([`${crashFile}:prompt`, `${crashFile}:response`]);
			const rows: string[] = [];
			const recovered = await embedRecoveredRounds([crashFile], roundsDir, {
				embed: (text) => Promise.resolve([text.length, 2]),
				appendIndexRow: (label) => rows.push(label),
				hasIndexRow: (label) => preexisting.has(label),
				writeRoundEmbedding: (name, vec) => {
					const p = path.join(roundsDir, name);
					const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
					existing.promptEmbedding = vec;
					fs.writeFileSync(p, JSON.stringify(existing, null, 2));
				},
			});
			expect(recovered.embedded).toEqual([crashFile]);
			expect(recovered.errors).toEqual([]);
			// guarded labels were skipped — no duplicate rows appended
			expect(rows).toEqual([]);
			expect(JSON.parse(fs.readFileSync(path.join(roundsDir, crashFile), "utf-8")).promptEmbedding).toBeDefined();
		});

		// A round that fails to read reports an error without aborting the queue
		it("reports per-round errors without aborting the queue", async () => {
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(
				writeSessionFile(tmp, [userMsg("err q", "u3"), assistantMsg("err answer")]),
				roundsDir,
				undefined,
				{ liveWindowMs: 0 },
			);
			const fileName = outcome.recoveredFiles[0];
			// pre-embed the round so it is skipped on this pass
			const pre = path.join(roundsDir, fileName);
			const preRound = JSON.parse(fs.readFileSync(pre, "utf-8"));
			preRound.promptEmbedding = [1];
			fs.writeFileSync(pre, JSON.stringify(preRound, null, 2));
			const failing = await embedRecoveredRounds([fileName, "missing-round.json"], roundsDir, {
				embed: () => Promise.resolve([1]),
				appendIndexRow: () => {},
				writeRoundEmbedding: () => {},
			});
			// already-embedded round is skipped, missing round is reported
			expect(failing.embedded).toEqual([]);
			expect(failing.errors).toHaveLength(1);
			expect(failing.errors[0]).toContain("missing-round.json");
		});
	});

	it("early-exits when the last fileable round already exists (startup cost note)", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("first q"),
			assistantMsg("first a"),
			userMsg("second q", "u2"),
			assistantMsg("second a"),
		]);
		const roundsDir = path.join(tmp, "rounds");
		// Only the LAST round's file exists — earlier rounds are "missing" but the
		// write-first persist order guarantees they are present in reality.
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(path.join(roundsDir, createRoundFilePath("second q", "second a", [])), "{}");
		const { missing, scanned, skippedComplete } = findMissingRounds(sessionFile, roundsDir);
		expect(skippedComplete).toBe(true);
		expect(missing).toEqual([]);
		expect(scanned).toBe(0);
	});

	it("F1: early exit recognizes a live-derived filename for a marker round (PR !131)", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("marker q"),
			{
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: "marker a\nround_needs_followup" }] },
			},
		]);
		const roundsDir = path.join(tmp, "rounds");
		// Filename as the LIVE agent_end write would derive it (marker stripped
		// before hashing). Pre-fix, the early exit hashed the raw marker text
		// and missed this file.
		fs.mkdirSync(roundsDir, { recursive: true });
		const live = buildAgentEndRoundFile("marker q", ["marker a\nround_needs_followup"], undefined, []);
		fs.writeFileSync(path.join(roundsDir, live!.fileName), "{}");
		const { missing, scanned, skippedComplete } = findMissingRounds(sessionFile, roundsDir);
		expect(skippedComplete).toBe(true);
		expect(missing).toEqual([]);
		expect(scanned).toBe(0);
	});

	it("F1: recovered marker rounds are stored with cleaned text, needsFollowup, and the live filename (PR !131)", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("marker q"),
			{
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: "marker a\nround_needs_followup" }] },
			},
		]);
		const roundsDir = path.join(tmp, "rounds");
		// liveWindowMs: 0 — the fixture session file was just written and would
		// otherwise be classified as a live source (F3 gating).
		const outcome = backfillMissingRounds(sessionFile, roundsDir, fs, { liveWindowMs: 0 });
		const live = buildAgentEndRoundFile("marker q", ["marker a\nround_needs_followup"], undefined, []);
		expect(outcome.recoveredFiles).toEqual([live!.fileName]);
		const stored = JSON.parse(fs.readFileSync(path.join(roundsDir, live!.fileName), "utf-8")) as {
			responseSequence: string;
			needsFollowup: boolean;
		};
		expect(stored.responseSequence).toBe("marker a");
		expect(stored.needsFollowup).toBe(true);
	});

	it("does not early-exit when the last fileable round is missing", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("first q"),
			assistantMsg("first a"),
			userMsg("second q", "u2"),
			assistantMsg("second a"),
		]);
		const roundsDir = path.join(tmp, "rounds");
		// Only an EARLIER round's file exists — full scan must still run.
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(path.join(roundsDir, createRoundFilePath("first q", "first a", [])), "{}");
		const { missing, scanned, skippedComplete } = findMissingRounds(sessionFile, roundsDir);
		expect(skippedComplete).toBeUndefined();
		expect(scanned).toBe(2);
		expect(missing).toHaveLength(1);
		expect(missing[0].fileName).toBe(createRoundFilePath("second q", "second a", []));
	});
});

describe("planStartupEmbedding (issue #133)", () => {
	type Round = { promptEmbedding?: number[] };
	const embedded = (n: number): [string, Round][] =>
		Array.from({ length: n }, (_, i) => [`r${i}.json`, { promptEmbedding: [0.1] }]);

	it("keeps embedding inline when the unembedded count is at or below the threshold", () => {
		const files = new Map<string, Round>([...embedded(10), ["rX.json", { promptEmbedding: [0.2] }]]);
		const plan = planStartupEmbedding([...files.keys()], (f) => files.get(f) ?? null);
		expect(plan).toEqual({ mode: "inline", pendingCount: 0 });
	});

	it("defers when the unembedded count exceeds the threshold, reporting the pending count", () => {
		const files = new Map<string, Round>([
			...embedded(3),
			...Array.from({ length: 11 }, (_, i) => [`p${i}.json`, {}] as const),
		]);
		const plan = planStartupEmbedding([...files.keys()], (f) => files.get(f) ?? null);
		// Message built from this count: `🧠 11 rounds pending embedding backfill — run just index`
		expect(plan).toEqual({ mode: "defer", pendingCount: 11 });
	});

	it("treats unreadable round files as pending", () => {
		const files = new Map<string, { promptEmbedding: number[] }>([["a.json", { promptEmbedding: [1] }]]);
		const plan = planStartupEmbedding(["a.json", "missing.json"], (f) => files.get(f) ?? null);
		expect(plan).toEqual({ mode: "inline", pendingCount: 1 });
	});
});

describe("listBackfillCandidates", () => {
	let tmp: string;
	beforeEach(() => {
		tmp = fs.mkdtempSync(path.join(os.tmpdir(), "prev-session-"));
	});
	afterEach(() => {
		fs.rmSync(tmp, { recursive: true, force: true });
	});

	it("returns all other .jsonl files in the session dir, newest first", () => {
		const older = path.join(tmp, "a.jsonl");
		const newer = path.join(tmp, "b.jsonl");
		fs.writeFileSync(older, "{}");
		fs.writeFileSync(newer, "{}");
		fs.utimesSync(older, new Date(1000), new Date(1000));
		fs.utimesSync(newer, new Date(2000), new Date(2000));
		const current = path.join(tmp, "c.jsonl");
		fs.writeFileSync(current, "{}");
		expect(listBackfillCandidates(tmp, current)).toEqual([newer, older]);
	});

	it("excludes the current session file", () => {
		const current = path.join(tmp, "c.jsonl");
		fs.writeFileSync(current, "{}");
		expect(listBackfillCandidates(tmp, current)).toEqual([]);
	});

	it("returns [] for a missing session dir", () => {
		expect(listBackfillCandidates(path.join(tmp, "nope"), path.join(tmp, "cur.jsonl"))).toEqual([]);
	});
});

describe("isBackfillStartReason (F5)", () => {
	it("gates backfill on every session_start reason pi emits", () => {
		for (const reason of ["startup", "resume", "reload", "new", "fork"]) {
			expect(isBackfillStartReason(reason), reason).toBe(true);
		}
	});

	it("rejects unknown reasons", () => {
		for (const reason of ["", "shutdown", "compact", "resumed"]) {
			expect(isBackfillStartReason(reason), reason).toBe(false);
		}
	});
});

describe("buildBackfillCandidates (F5)", () => {
	let tmp: string;
	beforeEach(() => {
		tmp = fs.mkdtempSync(path.join(os.tmpdir(), "backfill-candidates-"));
	});
	afterEach(() => {
		fs.rmSync(tmp, { recursive: true, force: true });
	});

	it("pins previousSessionFile first when the dir scan misses it", () => {
		const scanned = path.join(tmp, "a.jsonl");
		fs.writeFileSync(scanned, "{}");
		const other = fs.mkdtempSync(path.join(os.tmpdir(), "backfill-external-"));
		const external = path.join(other, "prev.jsonl");
		fs.writeFileSync(external, "{}");
		try {
			expect(buildBackfillCandidates(tmp, path.join(tmp, "cur.jsonl"), external)).toEqual([external, scanned]);
		} finally {
			fs.rmSync(other, { recursive: true, force: true });
		}
	});

	it("does not duplicate when the dir scan already includes previousSessionFile", () => {
		const prev = path.join(tmp, "prev.jsonl");
		fs.writeFileSync(prev, "{}");
		expect(buildBackfillCandidates(tmp, path.join(tmp, "cur.jsonl"), prev)).toEqual([prev]);
	});

	it("returns just the dir scan when no previousSessionFile (startup/reload)", () => {
		const scanned = path.join(tmp, "a.jsonl");
		fs.writeFileSync(scanned, "{}");
		expect(buildBackfillCandidates(tmp, path.join(tmp, "cur.jsonl"))).toEqual([scanned]);
	});

	it("excludes the current session file even when it is previousSessionFile (resume)", () => {
		const current = path.join(tmp, "cur.jsonl");
		fs.writeFileSync(current, "{}");
		expect(buildBackfillCandidates(tmp, current, current)).toEqual([]);
	});
});

describe("indexRecoveredRounds", () => {
	const toolCall: ToolCallDetail = { index: 0, id: "t1", name: "bash", arguments: "{}", result_summary: "" };

	it("upserts bm25 for every readable round and appends tool rows only for rounds with toolCalls", () => {
		const rounds: Record<string, RecoveredRoundLike | null> = {
			"a.json": { toolCalls: [toolCall] },
			"b.json": { toolCalls: [] },
			"c.json": {},
		};
		const bm25: string[] = [];
		const toolAppends: Array<{ fileName: string; toolCalls: readonly ToolCallDetail[] }> = [];
		const report = indexRecoveredRounds(Object.keys(rounds), {
			readRoundData: (f) => rounds[f] ?? null,
			upsertBm25: (f) => bm25.push(f),
			flushBm25: () => {},
			appendToolRows: (f, tcs) => toolAppends.push({ fileName: f, toolCalls: tcs }),
		});
		expect(report.errors).toEqual([]);
		expect(report.bm25Indexed).toBe(3);
		expect(report.toolIndexed).toBe(1);
		expect(toolAppends).toEqual([{ fileName: "a.json", toolCalls: [toolCall] }]);
	});

	it("records unreadable round files as errors and skips indexing them", () => {
		const bm25: string[] = [];
		const report = indexRecoveredRounds(["bad.json"], {
			readRoundData: () => null,
			upsertBm25: (f) => bm25.push(f),
			flushBm25: () => {},
			appendToolRows: () => {},
		});
		expect(bm25).toEqual([]);
		expect(report.bm25Indexed).toBe(0);
		expect(report.errors).toEqual(["bad.json: unreadable round file"]);
	});

	it("guards each round independently — a throwing index call costs only that round", () => {
		const bm25: string[] = [];
		const report = indexRecoveredRounds(["fail.json", "ok.json"], {
			readRoundData: (f) => (f === "fail.json" ? { toolCalls: [] } : { toolCalls: [] }),
			upsertBm25: (f) => {
				if (f === "fail.json") throw new Error("bm25 boom");
				bm25.push(f);
			},
			flushBm25: () => {},
			appendToolRows: () => {},
		});
		expect(bm25).toEqual(["ok.json"]);
		expect(report.bm25Indexed).toBe(1);
		expect(report.errors).toEqual(["fail.json: bm25 boom"]);
	});

	it("calls the flush dep exactly once for an N-round batch (issue #137)", () => {
		const bm25Upserts: string[] = [];
		const writes: string[] = [];
		const report = indexRecoveredRounds(["a.json", "b.json", "c.json"], {
			readRoundData: (f) => (f === "c.json" ? null : { toolCalls: [] }),
			upsertBm25: (f) => bm25Upserts.push(f),
			flushBm25: () => writes.push("flush"),
			appendToolRows: () => {},
		});
		expect(bm25Upserts).toEqual(["a.json", "b.json"]);
		expect(writes).toEqual(["flush"]);
		expect(report.errors).toEqual(["c.json: unreadable round file"]);
	});

	it("lands a failing batch flush in errors without corrupting the report", () => {
		const report = indexRecoveredRounds(["a.json"], {
			readRoundData: () => ({ toolCalls: [] }),
			upsertBm25: () => {},
			flushBm25: () => {
				throw new Error("flush boom");
			},
			appendToolRows: () => {},
		});
		expect(report.bm25Indexed).toBe(1);
		expect(report.errors).toEqual(["bm25 flush: flush boom"]);
	});

	it("does not flush when no round was bm25-indexed", () => {
		const writes: string[] = [];
		const report = indexRecoveredRounds(["bad.json"], {
			readRoundData: () => null,
			upsertBm25: () => {},
			flushBm25: () => writes.push("flush"),
			appendToolRows: () => {},
		});
		expect(writes).toEqual([]);
		expect(report.bm25Indexed).toBe(0);
	});

	it("is synchronous — all indexing effects have landed when it returns (issue #133: must-not-lose contract)", () => {
		const bm25: string[] = [];
		const toolAppends: string[] = [];
		const report = indexRecoveredRounds(["a.json", "b.json"], {
			readRoundData: () => ({ toolCalls: [toolCall] }),
			upsertBm25: (f) => bm25.push(f),
			flushBm25: () => {},
			appendToolRows: (f) => toolAppends.push(f),
		});
		// Not a promise: the returned report is final the instant the call returns.
		expect(report).not.toBeInstanceOf(Promise);
		// Every effect is complete by return time — the caller's deferral of the
		// embedding burst must never delay these indexes.
		expect(bm25).toEqual(["a.json", "b.json"]);
		expect(toolAppends).toEqual(["a.json", "b.json"]);
		expect(report.bm25Indexed).toBe(2);
		expect(report.toolIndexed).toBe(2);
		expect(report.errors).toEqual([]);
	});

	it("propagates appendToolRows failures into errors without blocking later rounds", () => {
		const toolAppends: string[] = [];
		const report = indexRecoveredRounds(["a.json", "b.json"], {
			readRoundData: () => ({ toolCalls: [toolCall] }),
			upsertBm25: () => {},
			flushBm25: () => {},
			appendToolRows: (f) => {
				if (f === "a.json") throw new Error("tool index boom");
				toolAppends.push(f);
			},
		});
		expect(toolAppends).toEqual(["b.json"]);
		expect(report.toolIndexed).toBe(1);
		expect(report.errors).toEqual(["a.json: tool index boom"]);
	});
});

describe("extractCheckpointSummary (F7)", () => {
	const acceptedArgs = JSON.stringify({
		currentTask: "task",
		progressMade: ["a"],
		currentState: ["b"],
		nextSteps: ["c"],
		keyFindings: ["d"],
	});
	const acceptedResult =
		"Checkpoint recorded. Your progress summary has been saved. You may now stop — do not start new work.";

	function tc(name: string, args: string, result: string): ToolCallDetail {
		return { index: 0, id: "t1", name, arguments: args, result_summary: result.slice(0, 300) };
	}

	it("recovers the summary from an accepted semblr_checkpoint call", () => {
		const summary = extractCheckpointSummary([tc("semblr_checkpoint", acceptedArgs, acceptedResult)]);
		expect(summary).toEqual({
			currentTask: "task",
			progressMade: ["a"],
			currentState: ["b"],
			nextSteps: ["c"],
			keyFindings: ["d"],
		});
	});

	it("takes the LAST accepted call (live overwrite semantics)", () => {
		const last = extractCheckpointSummary([
			tc("semblr_checkpoint", JSON.stringify({ currentTask: "old", progressMade: [] }), acceptedResult),
			tc("semblr_checkpoint", acceptedArgs, acceptedResult),
		]);
		expect(last?.currentTask).toBe("task");
	});

	it("ignores rejected calls (no context warning was active)", () => {
		expect(
			extractCheckpointSummary([tc("semblr_checkpoint", acceptedArgs, "No context size warning is active.")]),
		).toBeNull();
	});

	it("ignores malformed arguments and unrelated tool calls", () => {
		expect(extractCheckpointSummary([tc("read", "not json{", acceptedResult)])).toBeNull();
		expect(
			extractCheckpointSummary([tc("semblr_checkpoint", '{"currentTask":1,"progressMade":[]}', acceptedResult)]),
		).toBeNull();
		expect(extractCheckpointSummary([])).toBeNull();
	});

	it("backfill writes the recovered summary into the round file", () => {
		const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "backfill-f7-"));
		try {
			const sessionFile = writeSessionFile(tmpDir, [
				userMsg("big task"),
				{
					type: "message",
					message: {
						role: "assistant",
						content: [
							{ type: "toolCall", name: "semblr_checkpoint", arguments: JSON.parse(acceptedArgs), id: "c1" },
						],
					},
				},
				{
					type: "message",
					message: {
						role: "toolResult",
						toolName: "semblr_checkpoint",
						toolCallId: "c1",
						content: [{ type: "text", text: acceptedResult }],
					},
				},
				assistantMsg("done"),
			]);
			const roundsDir = path.join(tmpDir, "rounds");
			const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
			const written = JSON.parse(fs.readFileSync(path.join(roundsDir, outcome.recoveredFiles[0]), "utf-8"));
			expect(written.summary.currentTask).toBe("task");
			expect(written.summary.keyFindings).toEqual(["d"]);
		} finally {
			fs.rmSync(tmpDir, { recursive: true, force: true });
		}
	});

	it("recovered summary embeds as a :summary row like the live path", async () => {
		const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "backfill-f7-embed-"));
		try {
			const sessionFile = writeSessionFile(tmpDir, [
				userMsg("big task"),
				{
					type: "message",
					message: {
						role: "assistant",
						content: [
							{ type: "toolCall", name: "semblr_checkpoint", arguments: JSON.parse(acceptedArgs), id: "c1" },
						],
					},
				},
				{
					type: "message",
					message: {
						role: "toolResult",
						toolName: "semblr_checkpoint",
						toolCallId: "c1",
						content: [{ type: "text", text: acceptedResult }],
					},
				},
				assistantMsg("done"),
			]);
			const roundsDir = path.join(tmpDir, "rounds");
			const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
			const fileName = outcome.recoveredFiles[0];
			const labels: string[] = [];
			const inputs: string[] = [];
			await embedRecoveredRounds(outcome.recoveredFiles, roundsDir, {
				embed: (text) => {
					inputs.push(text);
					return Promise.resolve([text.length, 1]);
				},
				appendIndexRow: (label) => labels.push(label),
				writeRoundEmbedding: () => {},
			});
			expect(labels).toContain(`${fileName}:summary`);
			expect(inputs.some((t) => t.startsWith("Current Task: task"))).toBe(true);
		} finally {
			fs.rmSync(tmpDir, { recursive: true, force: true });
		}
	});
});

describe("prompt-only round skip (PR !131 F1)", () => {
	let tmp: string;
	beforeEach(() => {
		tmp = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-prompt-only-"));
	});
	afterEach(() => {
		fs.rmSync(tmp, { recursive: true, force: true });
	});

	it("does not recover a prompt-only tail round (EOF flush of a just-prompted live session)", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("first q"),
			assistantMsg("first a"),
			userMsg("second q", "u2"), // prompt with no assistant reply yet
		]);
		const roundsDir = path.join(tmp, "rounds");
		const outcome = backfillMissingRounds(sessionFile, roundsDir, fs, { liveWindowMs: 0 });
		expect(outcome.recoveredFiles).toEqual([createRoundFilePath("first q", "first a", [])]);
	});

	it("skips prompt-only rounds in the full scan but still recovers the real ones", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("only q", "u0"), // never answered — prompt-only, not fileable
			userMsg("real q", "u1"),
			assistantMsg("real a"),
		]);
		const roundsDir = path.join(tmp, "rounds");
		const { missing, scanned } = findMissingRounds(sessionFile, roundsDir);
		expect(scanned).toBe(1);
		expect(missing).toHaveLength(1);
		expect(missing[0].fileName).toBe(createRoundFilePath("real q", "real a", []));
	});

	it("early exit skips a prompt-only tail and still recognizes the earlier real round", () => {
		const sessionFile = writeSessionFile(tmp, [
			userMsg("real q"),
			assistantMsg("real a"),
			userMsg("tail q", "u2"), // prompt-only tail
		]);
		const roundsDir = path.join(tmp, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(path.join(roundsDir, createRoundFilePath("real q", "real a", [])), "{}");
		const { missing, scanned, skippedComplete } = findMissingRounds(sessionFile, roundsDir);
		expect(skippedComplete).toBe(true);
		expect(missing).toEqual([]);
		expect(scanned).toBe(0);
	});
});
