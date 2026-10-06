/**
 * session-backfill.ts — Recover rounds lost to process death (issue #130).
 *
 * When pi dies mid-run (kill / power loss), no agent_end fires and semblr
 * never writes the round file — but the pi session JSONL still contains the
 * full user prompt and assistant response. On the next session_start with a
 * `previousSessionFile`, parse that file and write round files for any round
 * whose content-hash file does not yet exist in the rounds directory.
 *
 * Idempotent: existing round files (matched by content-hash filename) are
 * skipped, so re-running or normal agent_end dedup stays consistent.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { parsePiSessionJsonl, reconstructPiSessionRounds } from "./pi-session.ts";
import { buildAgentEndRoundData } from "./round-capture.ts";
import type { RoundData, ToolCallDetail } from "./round-data.ts";

export interface BackfillOutcome {
	/** Round files that were written by this backfill run. */
	recoveredFiles: string[];
	/** Number of fileable rounds found in the session file. */
	scanned: number;
	/** True when the source file looked live and backfill was skipped (F3). */
	skippedLive?: boolean;
}

/**
 * A source file modified within this window is considered live (still being
 * written by a running pi process) and backfilling from it is skipped (F3,
 * PR #131): a live session's tail round is incomplete and will be written by
 * the live process itself. Tests can shrink the window for determinism.
 */
export const LIVE_SESSION_WINDOW_MS = 5 * 60 * 1000;

export interface BackfillWrite {
	fileName: string;
	roundData: RoundData;
}

/**
 * Locate the previous session file when pi does not provide one (fresh launch:
 * session_start fires with reason "startup" and no previousSessionFile). The
 * session directory is per-cwd, so the most recent other .jsonl file in the
 * same directory as the current session file is the previous session.
 */
export function findPreviousSessionFile(
	sessionDir: string,
	currentSessionFile: string,
	fsImpl: Pick<typeof fs, "readdirSync" | "statSync"> = fs,
): string | null {
	let candidates: string[];
	try {
		candidates = fsImpl.readdirSync(sessionDir).filter((f) => f.endsWith(".jsonl"));
	} catch {
		return null;
	}
	let best: { file: string; mtime: number } | null = null;
	for (const name of candidates) {
		const file = path.join(sessionDir, name);
		if (path.resolve(file) === path.resolve(currentSessionFile)) continue;
		let mtime: number;
		try {
			mtime = fsImpl.statSync(file).mtimeMs;
		} catch {
			continue;
		}
		if (!best || mtime > best.mtime) best = { file, mtime };
	}
	return best?.file ?? null;
}

/**
 * Parse a session JSONL and return the round files missing from roundsDir.
 * Exported for testing / inspection without side effects.
 */
export function findMissingRounds(
	sessionFile: string,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync"> = fs,
): { missing: BackfillWrite[]; scanned: number } {
	const raw = fs.readFileSync(sessionFile, "utf-8");
	const reconstructed = reconstructPiSessionRounds(parsePiSessionJsonl(raw));
	const missing: BackfillWrite[] = [];
	let scanned = 0;
	for (const { roundFile, round } of reconstructed) {
		// Rounds without a user prompt cannot be filed (filename derives from it).
		if (!round.userPrompt) continue;
		scanned++;
		if (fsImpl.existsSync(path.join(roundsDir, roundFile))) continue;
		const roundData = buildAgentEndRoundData({
			userPrompt: round.userPrompt,
			responseText: round.responseSequence,
			turnIndex: round.turnIndex,
			toolCallCount: round.toolCallCount,
			toolCallNames: round.toolCallNames,
			toolCalls: round.toolCalls as ToolCallDetail[],
			responseSegments: round.responseSegments,
			parentId: null,
			userTimestamp: round.userTimestamp,
		}) as unknown as RoundData;
		missing.push({ fileName: roundFile, roundData });
	}
	return { missing, scanned };
}

/**
 * Recover lost rounds from a previous session file. Writes any missing round
 * file into roundsDir and returns what was recovered. Sources modified within
 * the live window are treated as live sessions and skipped (F3).
 */
export function backfillMissingRounds(
	sessionFile: string,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "statSync"> = fs,
	opts: { liveWindowMs?: number; nowMs?: number } = {},
): BackfillOutcome {
	const window = opts.liveWindowMs ?? LIVE_SESSION_WINDOW_MS;
	try {
		// Truncate sub-ms precision — a just-written file must not look like a
		// future timestamp and be misclassified as live.
		const ageMs = (opts.nowMs ?? Date.now()) - Math.floor(fsImpl.statSync(sessionFile).mtimeMs);
		if (ageMs < window) {
			return { recoveredFiles: [], scanned: 0, skippedLive: true };
		}
	} catch {
		// Unreadable mtime: fall through and attempt the backfill.
	}
	const { missing, scanned } = findMissingRounds(sessionFile, roundsDir, fsImpl);
	const recovered: string[] = [];
	for (const { fileName, roundData } of missing) {
		fsImpl.mkdirSync(roundsDir, { recursive: true });
		fsImpl.writeFileSync(path.join(roundsDir, fileName), JSON.stringify(roundData, null, 2));
		recovered.push(fileName);
	}
	return { recoveredFiles: recovered, scanned };
}
