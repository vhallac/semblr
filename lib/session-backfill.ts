/**
 * session-backfill.ts — Recover rounds lost to process death (issue #130).
 *
 * When pi dies mid-run (kill / power loss), no agent_end fires and semblr
 * never writes the round file — but the pi session JSONL still contains the
 * full user prompt and assistant response. On the next backfill-triggering
 * session_start (see `isBackfillStartReason`) with a
 * `previousSessionFile`, parse that file and write round files for any round
 * whose content-hash file does not yet exist in the rounds directory.
 *
 * Idempotent: existing round files (matched by content-hash filename) are
 * skipped, so re-running or normal agent_end dedup stays consistent.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { type EmbedRoundDeps, embedRound } from "./embed-round.ts";
import { parsePiSessionJsonl, reconstructPiSessionRounds } from "./pi-session.ts";
import type { PromptNoiseOptions } from "./round-capture.ts";
import { buildAgentEndRoundData, deriveRoundFile, isPromptOnlyRound } from "./round-capture.ts";
import {
	buildCheckpointSummaryText,
	type CheckpointSummary,
	type RoundData,
	type ToolCallDetail,
} from "./round-data.ts";

/**
 * F7 (PR !131): recover a checkpoint summary from the round's parsed tool
 * calls. The `semblr_checkpoint` call arguments carry the full structured
 * summary, and the "Checkpoint recorded." result text confirms the live
 * path accepted it (warning was active). The live path's
 * `contextWarningIssued` state is process-local and unrecoverable, so the
 * acceptance marker in the result is the authoritative evidence here. The
 * LAST accepted call wins, matching the live overwrite semantics.
 */
export function extractCheckpointSummary(toolCalls: readonly ToolCallDetail[]): CheckpointSummary | null {
	for (let i = toolCalls.length - 1; i >= 0; i--) {
		const tc = toolCalls[i];
		if (!tc.name.includes("semblr_checkpoint")) continue;
		if (!tc.result_summary.includes("Checkpoint recorded")) continue;
		try {
			const parsed = JSON.parse(tc.arguments) as Partial<CheckpointSummary>;
			if (typeof parsed.currentTask !== "string" || !Array.isArray(parsed.progressMade)) continue;
			return {
				currentTask: parsed.currentTask,
				progressMade: parsed.progressMade,
				currentState: Array.isArray(parsed.currentState) ? parsed.currentState : [],
				nextSteps: Array.isArray(parsed.nextSteps) ? parsed.nextSteps : [],
				keyFindings: Array.isArray(parsed.keyFindings) ? parsed.keyFindings : [],
			};
		} catch {}
	}
	return null;
}

export interface BackfillOutcome {
	/** Round files that were written by this backfill run. */
	recoveredFiles: string[];
	/** Number of fileable rounds found across scanned session files. */
	scanned: number;
	/** True when at least one source session was fully backed up (early exit). */
	skippedComplete?: boolean;
	/** Number of source sessions whose tail round is missing but which look live; recovery is deferred to a later startup (F2). */
	deferredLive?: number;
}

/**
 * A source file modified within this window is considered live (still being
 * written by a running pi process). A live session whose tail round is missing
 * gets its recovery DEFERRED, not skipped outright (F2, PR !131): because the
 * candidate scan covers all prior session files on every startup, the file is
 * retried later once the session has ended. Tests can shrink the window for
 * determinism.
 */
export const LIVE_SESSION_WINDOW_MS = 5 * 60 * 1000;

export interface BackfillWrite {
	fileName: string;
	roundData: RoundData;
}

/**
 * All prior session files in the session dir, newest first, excluding the
 * current session (F2, PR !131): the candidate scan covers every previous
 * session so a file deferred as live on one startup is retried on the next.
 */
export function listBackfillCandidates(
	sessionDir: string,
	currentSessionFile: string,
	fsImpl: Pick<typeof fs, "readdirSync" | "statSync"> = fs,
): string[] {
	let names: string[];
	try {
		names = fsImpl.readdirSync(sessionDir).filter((f) => f.endsWith(".jsonl"));
	} catch {
		return [];
	}
	const candidates: { file: string; mtime: number }[] = [];
	for (const name of names) {
		const file = path.join(sessionDir, name);
		if (path.resolve(file) === path.resolve(currentSessionFile)) continue;
		try {
			candidates.push({ file, mtime: fsImpl.statSync(file).mtimeMs });
		} catch {}
	}
	return candidates.sort((a, b) => b.mtime - a.mtime).map((c) => c.file);
}

/**
 * F5 (PR !131): session_start reasons that trigger backfill. `startup` and
 * `resume` scan the session dir; `new`/`fork`/`reload` are included so a
 * source deferred as live on one start is retried on the next transition,
 * with pi's `previousSessionFile` (present for new/resume/fork) pinned as
 * the first candidate. Reload carries no previousSessionFile but still gets
 * the dir scan — a file live at startup may be closed by the time /reload
 * restarts the extension.
 */
export function isBackfillStartReason(reason: string): boolean {
	return reason === "startup" || reason === "resume" || reason === "reload" || reason === "new" || reason === "fork";
}

/**
 * F5 (PR !131): build the backfill candidate list for a session_start —
 * newest-first dir scan (current session excluded) with pi's
 * `previousSessionFile` pinned first when pi supplied one and the scan
 * missed it (e.g. outside the cwd's session dir).
 */
export function buildBackfillCandidates(
	sessionDir: string,
	currentSessionFile: string,
	previousSessionFile?: string,
	fsImpl: Pick<typeof fs, "readdirSync" | "statSync"> = fs,
): string[] {
	const candidates = listBackfillCandidates(sessionDir, currentSessionFile, fsImpl);
	if (
		previousSessionFile &&
		path.resolve(previousSessionFile) !== path.resolve(currentSessionFile) &&
		!candidates.some((c) => path.resolve(c) === path.resolve(previousSessionFile))
	) {
		candidates.unshift(previousSessionFile);
	}
	return candidates;
}

/**
 * F2 (PR !131 round 3): a session's tail is closed only when its last
 * assistant message ended the turn (stopReason other than "toolUse"). A tail
 * still waiting on tool results is mid-round — its last round is incomplete
 * and must not be recovered, no matter how old the file is (an mtime window
 * alone misclassifies a paused-but-open session). A file with no assistant
 * message at all has no turn in flight (the process died before responding)
 * → treated as closed; the mtime gate covers the just-prompted live case.
 */
export function isSessionTailClosed(rawJsonl: string): boolean {
	const lines = rawJsonl.split("\n");
	for (let i = lines.length - 1; i >= 0; i--) {
		const line = lines[i].trim();
		if (!line) continue;
		let entry: { type?: string; message?: { role?: string; stopReason?: string } };
		try {
			entry = JSON.parse(line);
		} catch {
			continue; // tolerate a truncated/corrupt trailing line, same as the parser
		}
		if (entry?.type !== "message") continue;
		const msg = entry.message;
		if (msg?.role !== "assistant") continue;
		return msg.stopReason !== "toolUse";
	}
	return true;
}

/**
 * Parse a session JSONL and return the round files missing from roundsDir.
 * Exported for testing / inspection without side effects.
 */
export function findMissingRounds(
	sessionFile: string,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync"> = fs,
): { missing: BackfillWrite[]; scanned: number; skippedComplete?: boolean } {
	const parsed = parsePiSessionJsonl(fs.readFileSync(sessionFile, "utf-8"));
	return collectMissingRounds(parsed, roundsDir, fsImpl);
}

/**
 * Startup-cost early exit (PR #131 note): rounds are persisted in order at
 * each agent_end, with the round file written before any fallible post-write
 * step — so a mid-round death loses only the round in progress. If the LAST
 * fileable round's file already exists, all earlier rounds are on disk too;
 * skip per-round reconstruction and hashing entirely. Doubles as the F2
 * tail-completeness check.
 */
function collectMissingRounds(
	parsed: ReturnType<typeof parsePiSessionJsonl>,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync">,
): { missing: BackfillWrite[]; scanned: number; skippedComplete?: boolean } {
	for (let i = parsed.length - 1; i >= 0; i--) {
		if (!parsed[i].userPrompt) continue;
		// F1 (PR !131): prompt-only tails (e.g. the parser's EOF flush of a
		// just-prompted live session) are never written by the live path, so they
		// must not anchor the early exit — skip them like unprompted rounds.
		if (isPromptOnlyRound(parsed[i].responseSequence, parsed[i].toolCalls)) continue;
		// F1 (PR !131): hash via the shared derivation so the early exit agrees
		// with the live write's filename for followup-marker rounds.
		const lastFile = deriveRoundFile(parsed[i].userPrompt, parsed[i].responseSequence, parsed[i].toolCalls).fileName;
		if (fsImpl.existsSync(path.join(roundsDir, lastFile))) {
			return { missing: [], scanned: 0, skippedComplete: true };
		}
		break;
	}
	const reconstructed = reconstructPiSessionRounds(parsed);
	const missing: BackfillWrite[] = [];
	let scanned = 0;
	for (const { roundFile, round } of reconstructed) {
		// Rounds without a user prompt cannot be filed (filename derives from it).
		if (!round.userPrompt) continue;
		// F1 (PR !131): prompt-only rounds (empty response, no tool calls) are
		// skipped by the live save path too — never persist a bogus empty round
		// from the parser's EOF flush.
		if (isPromptOnlyRound(round.responseSequence, round.toolCalls)) continue;
		scanned++;
		if (fsImpl.existsSync(path.join(roundsDir, roundFile))) continue;
		const roundData = {
			...buildAgentEndRoundData({
				userPrompt: round.userPrompt,
				responseText: round.responseSequence,
				turnIndex: round.turnIndex,
				toolCallCount: round.toolCallCount,
				toolCallNames: round.toolCallNames,
				toolCalls: round.toolCalls as ToolCallDetail[],
				responseSegments: round.responseSegments,
				parentId: null,
				userTimestamp: round.userTimestamp,
				// F1 (PR !131): recovered copies carry the marker state the live
				// write would have persisted instead of the default false.
				needsFollowup: round.needsFollowup,
				// F7 (PR !131): recover the checkpoint summary from the JSONL's
				// semblr_checkpoint tool call so checkpoint injection works for
				// recovered rounds too.
				summary: extractCheckpointSummary(round.toolCalls as ToolCallDetail[]) ?? undefined,
			}),
			// F4 (PR #131): mark recovered rounds so they are distinguishable from
			// live-saved rounds (second-class retrieval provenance).
			recovered: true,
		} satisfies RoundData;
		missing.push({ fileName: roundFile, roundData });
	}
	return { missing, scanned };
}

/**
 * Recover lost rounds from previous session files. Writes any missing round
 * file into roundsDir and returns what was recovered.
 *
 * F2 (PR !131) gating per source file, replacing the old one-shot mtime skip:
 * - tail round already on disk → session fully backed up, skip (cheap).
 * - tail missing and file looks live (mtime inside the window) → DEFER: a live
 *   session's tail is incomplete and will be written by the live process;
 *   the all-sessions candidate scan retries this file on a later startup.
 * - tail missing and file is stale → recover.
 */
/**
 * A minimal round-file shape — enough to decide whether tool-index rows exist.
 * Real round files are wider (RoundData); extra fields are irrelevant here.
 */
export interface RecoveredRoundLike {
	toolCalls?: ToolCallDetail[];
}

export interface IndexRecoveredRoundsDeps {
	/** Read a recovered round file; null/throwing means the file is unusable. */
	readRoundData: (fileName: string) => RecoveredRoundLike | null;
	/** Upsert the round into the bm25 index. */
	upsertBm25: (fileName: string, roundData: RecoveredRoundLike) => void;
	/** Append tool-index rows for the round's tool calls. */
	appendToolRows: (fileName: string, toolCalls: readonly ToolCallDetail[]) => void;
}

export interface RecoveredIndexReport {
	/** Rounds successfully upserted into the bm25 index. */
	bm25Indexed: number;
	/** Rounds whose tool calls were appended to the tool index. */
	toolIndexed: number;
	/** Per-file error messages; empty when everything indexed. */
	errors: string[];
}

/**
 * Index recovered rounds (F3, PR !131): a backfill must feed every derived
 * index the live agent_end path feeds — bm25 AND the tool fulltext index.
 * Each round is guarded independently so one corrupt file or index failure
 * costs only that derived index, never the other rounds.
 */
export function indexRecoveredRounds(
	fileNames: readonly string[],
	deps: IndexRecoveredRoundsDeps,
): RecoveredIndexReport {
	const report: RecoveredIndexReport = { bm25Indexed: 0, toolIndexed: 0, errors: [] };
	for (const fileName of fileNames) {
		try {
			const roundData = deps.readRoundData(fileName);
			if (!roundData) {
				report.errors.push(`${fileName}: unreadable round file`);
				continue;
			}
			deps.upsertBm25(fileName, roundData);
			report.bm25Indexed++;
			if (roundData.toolCalls && roundData.toolCalls.length > 0) {
				deps.appendToolRows(fileName, roundData.toolCalls);
				report.toolIndexed++;
			}
		} catch (err) {
			report.errors.push(`${fileName}: ${(err as Error).message}`);
		}
	}
	return report;
}

export function backfillMissingRounds(
	sessionFiles: string | string[],
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "statSync"> = fs,
	opts: { liveWindowMs?: number; nowMs?: number } = {},
): BackfillOutcome {
	const files = Array.isArray(sessionFiles) ? sessionFiles : [sessionFiles];
	const window = opts.liveWindowMs ?? LIVE_SESSION_WINDOW_MS;
	const now = opts.nowMs ?? Date.now();
	const recovered: string[] = [];
	let scanned = 0;
	let skippedComplete = false;
	let deferredLive = 0;
	for (const sessionFile of files) {
		let live = false;
		try {
			// Truncate sub-ms precision — a just-written file must not look like a
			// future timestamp and be misclassified as live.
			live = now - Math.floor(fsImpl.statSync(sessionFile).mtimeMs) < window;
		} catch {
			// Unreadable mtime: treat as stale and attempt the backfill.
		}
		// F2: defer before paying for the parse at all — a live file's tail is
		// incomplete by definition, so parsing it is wasted startup work.
		if (live) {
			deferredLive++;
			continue;
		}
		const raw = fs.readFileSync(sessionFile, "utf-8");
		// F2: mtime staleness is not closedness — the tail must have ended its
		// turn (terminal stopReason) before any of its rounds are recoverable.
		if (!isSessionTailClosed(raw)) {
			deferredLive++;
			continue;
		}
		const parsed = parsePiSessionJsonl(raw);
		const {
			missing,
			scanned: fileScanned,
			skippedComplete: complete,
		} = collectMissingRounds(parsed, roundsDir, fsImpl);
		if (complete) {
			skippedComplete = true;
			continue;
		}
		scanned += fileScanned;
		for (const { fileName, roundData } of missing) {
			fsImpl.mkdirSync(roundsDir, { recursive: true });
			fsImpl.writeFileSync(path.join(roundsDir, fileName), JSON.stringify(roundData, null, 2));
			recovered.push(fileName);
		}
	}
	return {
		recoveredFiles: recovered,
		scanned,
		...(skippedComplete ? { skippedComplete: true } : {}),
		...(deferredLive > 0 ? { deferredLive } : {}),
	};
}

/**
 * Injectable embedding + index-update surface for recovered-round embedding
 * (F4). The shape is the shared core's `EmbedRoundDeps`; the embedding policy
 * itself lives in `embedRound` (lib/embed-round.ts).
 */
export interface RecoveredEmbedDeps extends EmbedRoundDeps {}

export async function embedRecoveredRounds(
	fileNames: string[],
	roundsDir: string,
	deps: RecoveredEmbedDeps,
	opts: { maxResponseBytes?: number; promptNoiseOptions?: PromptNoiseOptions; promptMaxTokens?: number } = {},
): Promise<{ embedded: string[]; errors: string[] }> {
	const embedded: string[] = [];
	const errors: string[] = [];
	for (const fileName of fileNames) {
		try {
			const round = JSON.parse(fs.readFileSync(path.join(roundsDir, fileName), "utf-8")) as RoundData;
			if (round.promptEmbedding) continue;
			await embedRound(
				{
					fileName,
					userPrompt: round.userPrompt ?? "",
					responseText: round.responseSequence ?? "",
					// F7 (PR !131): recovered checkpoint summaries embed as a `:summary`
					// row, same as the live path.
					checkpointSummaryText: round.summary ? buildCheckpointSummaryText(round.summary) : null,
					maxResponseBytes: opts.maxResponseBytes,
					promptNoiseOptions: opts.promptNoiseOptions,
					promptMaxTokens: opts.promptMaxTokens,
				},
				deps,
			);
			embedded.push(fileName);
		} catch (err) {
			errors.push(`${fileName}: ${(err as Error).message}`);
		}
	}
	return { embedded, errors };
}
