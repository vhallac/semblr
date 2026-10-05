import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRoundFilePath } from "./hash.ts";
import { backfillMissingRounds, findMissingRounds, findPreviousSessionFile } from "./session-backfill.ts";

function writeSessionFile(dir: string, lines: object[]): string {
	const file = path.join(dir, "session.jsonl");
	fs.writeFileSync(file, lines.map((l) => JSON.stringify(l)).join("\n") + "\n");
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

		const outcome = backfillMissingRounds(sessionFile, roundsDir);
		expect(outcome.recoveredFiles).toEqual([expected]);
		expect(outcome.scanned).toBe(1);

		const written = JSON.parse(fs.readFileSync(path.join(roundsDir, expected), "utf-8"));
		expect(written.userPrompt).toBe("hello world");
		expect(written.responseSequence).toBe("hi there");
		expect(written.promptEmbedding).toBeUndefined();
	});

	it("is idempotent — existing round files are skipped", () => {
		const sessionFile = writeSessionFile(tmp, [userMsg("q"), assistantMsg("a")]);
		const roundsDir = path.join(tmp, "rounds");
		backfillMissingRounds(sessionFile, roundsDir);
		const first = fs.readFileSync(path.join(roundsDir, createRoundFilePath("q", "a", [])), "utf-8");

		const outcome = backfillMissingRounds(sessionFile, roundsDir);
		expect(outcome.recoveredFiles).toEqual([]);
		expect(outcome.scanned).toBe(1);
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
		const outcome = backfillMissingRounds(sessionFile, roundsDir);
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
		});
		expect(outcome.recoveredFiles).toEqual([]);
	});
});

describe("findPreviousSessionFile", () => {
	let tmp: string;
	beforeEach(() => {
		tmp = fs.mkdtempSync(path.join(os.tmpdir(), "prev-session-"));
	});
	afterEach(() => {
		fs.rmSync(tmp, { recursive: true, force: true });
	});

	it("returns the most recent other .jsonl in the session dir", () => {
		const older = path.join(tmp, "a.jsonl");
		const newer = path.join(tmp, "b.jsonl");
		fs.writeFileSync(older, "{}");
		fs.writeFileSync(newer, "{}");
		fs.utimesSync(older, new Date(1000), new Date(1000));
		fs.utimesSync(newer, new Date(2000), new Date(2000));
		const current = path.join(tmp, "c.jsonl");
		fs.writeFileSync(current, "{}");
		expect(findPreviousSessionFile(tmp, current)).toBe(newer);
	});

	it("excludes the current session file", () => {
		const current = path.join(tmp, "c.jsonl");
		fs.writeFileSync(current, "{}");
		expect(findPreviousSessionFile(tmp, current)).toBeNull();
	});

	it("returns null for a missing session dir", () => {
		expect(findPreviousSessionFile(path.join(tmp, "nope"), path.join(tmp, "cur.jsonl"))).toBeNull();
	});
});
