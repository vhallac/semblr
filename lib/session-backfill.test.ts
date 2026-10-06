import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRoundFilePath } from "./hash.ts";
import { buildAgentEndRoundFile, buildPromptEmbeddingInput } from "./round-capture.ts";
import type { ToolCallDetail } from "./round-data.ts";
import {
	backfillMissingRounds,
	embedRecoveredRounds,
	findMissingRounds,
	indexRecoveredRounds,
	listBackfillCandidates,
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
		// File still looks live, but the tail round is on disk → fully backed up.
		const outcome = backfillMissingRounds(complete, roundsDir, undefined, { nowMs: Date.now() });
		expect(outcome.deferredLive).toBeUndefined();
		expect(outcome.skippedComplete).toBe(true);
		expect(outcome.recoveredFiles).toEqual([]);
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
			const sessionFile = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a long answer")]);
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
			]);
			const roundsDir = path.join(tmp, "rounds");
			const outcome = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
			const fileName = outcome.recoveredFiles[0];
			const embedded: string[] = [];
			const rowHashes = new Map<string, string | undefined>();
			const result = await embedRecoveredRounds(outcome.recoveredFiles, roundsDir, {
				embed: (text) => {
					embedded.push(text);
					return Promise.resolve([text.length, 3]);
				},
				appendIndexRow: (label, _vec, hash) => rowHashes.set(label, hash),
				// live-parity derivation (same convention as semblr.ts agent_end),
				// with a small clip so the collapsed fence shrinks the input
				preparePrompt: (userPrompt) =>
					buildPromptEmbeddingInput(userPrompt, { fenceMaxChars: 4, jsonMaxChars: 0, repeatMaxChars: 0 }, 8000),
				writeRoundEmbedding: () => {},
			});
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
			appendToolRows: () => {},
		});
		expect(bm25).toEqual(["ok.json"]);
		expect(report.bm25Indexed).toBe(1);
		expect(report.errors).toEqual(["fail.json: bm25 boom"]);
	});

	it("propagates appendToolRows failures into errors without blocking later rounds", () => {
		const toolAppends: string[] = [];
		const report = indexRecoveredRounds(["a.json", "b.json"], {
			readRoundData: () => ({ toolCalls: [toolCall] }),
			upsertBm25: () => {},
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
