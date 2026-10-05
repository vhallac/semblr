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
}

export interface BackfillWrite {
	fileName: string;
	roundData: RoundData;
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
 * file into roundsDir and returns what was recovered.
 */
export function backfillMissingRounds(
	sessionFile: string,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync"> = fs,
): BackfillOutcome {
	const { missing, scanned } = findMissingRounds(sessionFile, roundsDir, fsImpl);
	const recovered: string[] = [];
	for (const { fileName, roundData } of missing) {
		fsImpl.mkdirSync(roundsDir, { recursive: true });
		fsImpl.writeFileSync(path.join(roundsDir, fileName), JSON.stringify(roundData, null, 2));
		recovered.push(fileName);
	}
	return { recoveredFiles: recovered, scanned };
}
