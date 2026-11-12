import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { backfillMissingRounds } from "./session-backfill.ts";

// R1.F1 (PR !160 review): the session_start handler's startup backfill block
// (previous-session recovery + issue #136 pending scan + inline embed or
// defer) is extracted into lib/startup-backfill.ts. These tests drive the
// REAL handler captured from the extension factory via a fake pi/ctx against
// a real temp rounds dir, so a wiring mutation (deleting the
// runStartupBackfill call, dropping the skippedComplete scan trigger, or
// changing the pinned defer message) fails the suite.

let agentDir = "";
let roundsDir = "";
let sessionStartHandler: ((event: unknown, ctx: unknown) => Promise<unknown>) | undefined;

function writeSessionFile(dir: string, lines: object[]): string {
	const file = path.join(dir, "session.jsonl");
	fs.writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	return file;
}

function userMsg(text: string, id: string, timestamp = 1000) {
	return { type: "message", id, message: { role: "user", content: [{ type: "text", text }], timestamp } };
}

function assistantMsg(text: string) {
	return { type: "message", message: { role: "assistant", content: [{ type: "text", text }] } };
}

beforeAll(async () => {
	// ROUNDS_DIR is resolved from the env at module load — pin it to the temp
	// dir before importing the extension.
	agentDir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "semblr-startup-wiring-")));
	roundsDir = path.join(agentDir, "rounds");
	fs.mkdirSync(roundsDir, { recursive: true });
	process.env.SEMBLR_ROUNDS_DIR = roundsDir;
	process.env.SEMBLR_HOST_ID = "startup-wiring-test";
	const mod = await import("../src/semblr.ts");
	const captured = new Map<string, unknown>();
	mod.default({
		on: (name: string, handler: unknown) => captured.set(name, handler),
		registerTool: () => undefined,
		registerCommand: () => undefined,
	} as never);
	sessionStartHandler = captured.get("session_start") as typeof sessionStartHandler;
});

beforeEach(() => {
	for (const entry of fs.readdirSync(roundsDir)) {
		fs.rmSync(path.join(roundsDir, entry), { recursive: true, force: true });
	}
});

// Runs the REAL session_start handler with a fake ctx: recording ui and a
// session manager pointed at the given session file.
async function runSessionStart(sessionFile: string, currentFile = sessionFile): Promise<string[]> {
	const statuses: string[] = [];
	await sessionStartHandler!(
		{ reason: "startup" },
		{
			ui: { setStatus: (_source: string, message: string) => statuses.push(message) },
			sessionManager: { getSessionFile: () => currentFile },
			modelRegistry: undefined,
		},
	);
	return statuses;
}

describe("session_start handler wiring (R1.F1, issue #136)", () => {
	// claim: issue-136-EXT-3a/EXT-3a/test_startup_handler_scan_finds_nothing_pending_startup_unchanged
	// claim text: the scan finds nothing pending → startup is unchanged (no
	// embedding or deferral message occurs on the skippedComplete path).
	// (every round file has current-model rows and a marker — the temp rounds
	// dir starts empty, so the skippedComplete-path scan counts 0 pending)
	it("startup handler scan finds nothing pending — startup is unchanged (EXT-3a)", async () => {
		const sessionDir = path.join(agentDir, "sessions-empty");
		fs.mkdirSync(sessionDir, { recursive: true });
		const sessionFile = writeSessionFile(sessionDir, [
			userMsg("an ordinary complete exchange", "u1"),
			assistantMsg("an ordinary complete answer"),
		]);
		const statuses = await runSessionStart(sessionFile);
		expect(statuses.some((m) => m.includes("pending embedding backfill"))).toBe(false);
		expect(statuses.some((m) => m.includes("embedded") && m.includes("pending round"))).toBe(false);
	});

	// claim: issue-136-acceptance/acceptance/test_startup_handler_run2_surfaces_stranded_rounds_with_pinned_defer_message
	// claim text: a startup after a crash-stranding run takes the skippedComplete
	// early exit, still scans the round directory, and surfaces the stranded
	// rounds with the pinned message (defer arm; the inline arm is realized by
	// the crash-stranding-harness tests).
	it("startup handler run 2 surfaces stranded rounds with the pinned defer message (acceptance)", async () => {
		const lines: object[] = [];
		for (let i = 0; i < 11; i++) {
			lines.push(
				userMsg(`please discuss border province dispute number ${i} at length`, `u${i}`),
				assistantMsg(`answer number ${i}`),
			);
		}
		const sessionDir = path.join(agentDir, "sessions-stranded");
		fs.mkdirSync(sessionDir, { recursive: true });
		const sessionFile = writeSessionFile(sessionDir, lines);
		// Run 1: recover, then crash before embedding — round files on disk,
		// no rows, no markers.
		const run1 = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		expect(run1.recoveredFiles).toHaveLength(11);
		// Age the session file beyond the live window (5 min) so the handler's
		// dir scan treats it as a closed previous session.
		const old = new Date(Date.now() - 6 * 60 * 1000);
		fs.utimesSync(sessionFile, old, old);
		// Run 2: the real session_start handler — the dir scan finds the closed
		// session (current.jsonl is excluded), takes the skippedComplete early
		// exit, the scan runs, the burst defers with the pinned message.
		const statuses = await runSessionStart(sessionFile, path.join(sessionDir, "current.jsonl"));
		expect(statuses).toContainEqual("\u{1f9e0} 11 rounds pending embedding backfill \u2014 run just index");
	});
});
