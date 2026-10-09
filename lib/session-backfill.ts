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
import { canonicalDuplicateKey, type DuplicateKeyRound } from "./hash.ts";
import { indexRoundFileFromPath, type VectorIndexEntry } from "./index-io.ts";
import { parsePiSessionJsonl, reconstructPiSessionRounds } from "./pi-session.ts";
import type { PromptNoiseOptions } from "./round-capture.ts";
import { buildAgentEndRoundData, deriveRoundFile, isPromptOnlyRound } from "./round-capture.ts";
import {
	buildCheckpointSummaryText,
	type CheckpointSummary,
	type RoundData,
	type ToolCallDetail,
} from "./round-data.ts";
import { loadScanCutoff } from "./scan-register.ts";

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
	opts: { stateDir?: string; cutoffMs?: number } = {},
): string[] {
	let candidates = listBackfillCandidates(sessionDir, currentSessionFile, fsImpl);
	// Unit-002 scan register: files at-or-below the last-full-scan mtime were
	// already covered by a complete scan, so opening them again is wasted work.
	// The tail-completeness check downstream stays authoritative — the cutoff
	// only prunes which files are OPENED, it cannot mask a missing round because
	// the register is only advanced after zero-recovery full scans (unit-003).
	// Corrupt/missing register → null → no filtering (full-scan fallback).
	if (opts.cutoffMs === undefined && opts.stateDir !== undefined) {
		opts = { ...opts, cutoffMs: loadScanCutoff(opts.stateDir, sessionDir) ?? undefined };
	}
	if (opts.cutoffMs !== undefined) {
		const cutoff = opts.cutoffMs;
		candidates = candidates.filter((file) => {
			try {
				return fsImpl.statSync(file).mtimeMs > cutoff;
			} catch {
				return true; // unreadable mtime: keep the candidate, stay safe
			}
		});
	}
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
 * Unit-001 tail-only completeness check: bytes read from the END of a session
 * file to decide completeness without a full read+parse. A session's last
 * round is at EOF, so a bounded tail window suffices; larger files fall back
 * to the full read when the window is inconclusive.
 */
export const TAIL_PEEK_BYTES = 256 * 1024;

/**
 * The tail-read fs surface is Partial so pre-existing minimal fsImpl injections
 * (e.g. existsSync-only stubs) keep type-checking; a missing method makes the
 * tail read throw → null → full-read fall-through, preserving old behavior.
 */
export type TailReadFs = Partial<Pick<typeof fs, "statSync" | "openSync" | "readSync" | "closeSync">>;

function tailFs(fsImpl: TailReadFs): Pick<typeof fs, "statSync" | "openSync" | "readSync" | "closeSync"> {
	return {
		statSync: fsImpl.statSync ?? fs.statSync,
		openSync: fsImpl.openSync ?? fs.openSync,
		readSync: fsImpl.readSync ?? fs.readSync,
		closeSync: fsImpl.closeSync ?? fs.closeSync,
	};
}

/**
 * Read the last `maxBytes` of a file. Returns the text plus whether the read
 * started mid-file (so the first returned chunk may be a partial line that
 * must be dropped). Null when the file cannot be stat'd/opened/read.
 */
export function readTailText(
	sessionFile: string,
	fsImpl: TailReadFs = fs,
	maxBytes = TAIL_PEEK_BYTES,
): { text: string; truncated: boolean } | null {
	const f = tailFs(fsImpl);
	try {
		const size = f.statSync(sessionFile).size;
		const start = Math.max(0, size - maxBytes);
		const fd = f.openSync(sessionFile, "r");
		try {
			const buf = Buffer.alloc(size - start);
			let read = 0;
			while (read < buf.length) {
				const n = f.readSync(fd, buf, read, buf.length - read, start + read);
				if (n <= 0) break;
				read += n;
			}
			return { text: buf.toString("utf-8", 0, read), truncated: start > 0 };
		} finally {
			f.closeSync(fd);
		}
	} catch {
		return null;
	}
}

/**
 * Tail closedness with an explicit "no evidence" answer: like
 * isSessionTailClosed, but returns null when the text contains NO assistant
 * message at all — a tail window truncated mid tool-loop proves nothing about
 * the file's real tail, so the caller must fall back to the full read.
 */
function tailClosedFromText(text: string): boolean | null {
	for (let i = text.split("\n").length - 1; i >= 0; i--) {
		const line = text.split("\n")[i].trim();
		if (!line) continue;
		let entry: { type?: string; message?: { role?: string; stopReason?: string } };
		try {
			entry = JSON.parse(line);
		} catch {
			continue;
		}
		if (entry?.type !== "message") continue;
		const msg = entry.message;
		if (msg?.role !== "assistant") continue;
		return msg.stopReason !== "toolUse";
	}
	return null;
}

/**
 * Unit-001: decide completeness from a bounded tail read only — no full-file
 * read or parse. Returns:
 * - true  → the tail round's file is on disk; the session is fully backed up.
 * - false → the tail turn is still open (stopReason toolUse); the caller must
 *   defer as live, exactly as the full isSessionTailClosed path would.
 * - null  → inconclusive (unreadable file, window without an assistant
 *   message, or no fileable tail round in the window); fall through to the
 *   full read+parse recovery path.
 */
export function tailRoundComplete(
	sessionFile: string,
	roundsDir: string,
	fsImpl: TailReadFs & Pick<typeof fs, "existsSync"> = fs,
): boolean | null {
	const tail = readTailText(sessionFile, fsImpl);
	if (tail === null) return null;
	// A read that started mid-file opens on a partial line; drop it. The rest
	// of the lines are complete (a UTF-8 continuation byte is never 0x0A).
	const text = tail.truncated ? tail.text.slice(tail.text.indexOf("\n") + 1) : tail.text;
	const closed = tailClosedFromText(text);
	if (closed !== true) return closed; // null → inconclusive; false → defer as live
	const parsed = parsePiSessionJsonl(text);
	for (let i = parsed.length - 1; i >= 0; i--) {
		const round = parsed[i];
		if (!round.userPrompt) continue;
		if (isPromptOnlyRound(round.responseSequence, round.toolCalls)) continue;
		// Same derivation as the full-path early exit in collectMissingRounds,
		// so a tail-window hit agrees byte-for-byte with the live write's name.
		const lastFile = deriveRoundFile(round.userPrompt, round.responseSequence, round.toolCalls).fileName;
		// Only TRUE short-circuits. A missing tail file is NOT "open tail" —
		// the round may be recoverable, so the caller must run the full path.
		return fsImpl.existsSync(path.join(roundsDir, lastFile)) ? true : null;
	}
	return null;
}

/**
 * Parse a session JSONL and return the round files missing from roundsDir.
 * Exported for testing / inspection without side effects.
 */
export function findMissingRounds(
	sessionFile: string,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readdirSync" | "readFileSync"> = fs,
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
/**
 * Build the set of whitespace-insensitive content keys (canonicalDuplicateKey)
 * for every round file currently on disk. Exported for testing. Unreadable or
 * unparseable files are tolerated (skipped) — they cannot match a candidate.
 */
export function loadDiskCanonicalKeys(
	roundsDir: string,
	fsImpl: Pick<typeof fs, "readdirSync" | "readFileSync"> = fs,
): Set<string> {
	const keys = new Set<string>();
	let names: string[];
	try {
		names = fsImpl.readdirSync(roundsDir);
	} catch {
		return keys;
	}
	for (const name of names) {
		if (!name.endsWith(".json")) continue;
		try {
			const round = JSON.parse(fsImpl.readFileSync(path.join(roundsDir, name), "utf-8")) as DuplicateKeyRound;
			keys.add(canonicalDuplicateKey(round));
		} catch {}
	}
	return keys;
}

function collectMissingRounds(
	parsed: ReturnType<typeof parsePiSessionJsonl>,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readdirSync" | "readFileSync">,
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
	// Lazily built on the first exact-hash miss: content keys of every round
	// file on disk (see the legacy-drift comment below).
	let existingCanonicalKeys: Set<string> | null = null;
	for (const { roundFile, round } of reconstructed) {
		// Rounds without a user prompt cannot be filed (filename derives from it).
		if (!round.userPrompt) continue;
		// F1 (PR !131): prompt-only rounds (empty response, no tool calls) are
		// skipped by the live save path too — never persist a bogus empty round
		// from the parser's EOF flush.
		if (isPromptOnlyRound(round.responseSequence, round.toolCalls)) continue;
		scanned++;
		if (fsImpl.existsSync(path.join(roundsDir, roundFile))) continue;
		// Legacy serialization drift: rounds stored by historical semblr code may
		// hash differently from today's parser (trailing-newline handling in tool
		// results, absent tool-call ids) even though the content is the same — and
		// those historical filenames cannot be recomputed from today's parse. So
		// on the first exact-hash miss, compare by whitespace-insensitive CONTENT
		// (same canonicalization as scripts/prune-recovery-duplicates.ts) against
		// every round file on disk; a content match means the round is already
		// stored under a legacy name. Without this, every startup backfill
		// re-recreates duplicates of legacy rounds, faster than pruning removes
		// them. The disk scan is paid only when an exact miss occurs.
		if (existingCanonicalKeys === null) existingCanonicalKeys = loadDiskCanonicalKeys(roundsDir, fsImpl);
		if (existingCanonicalKeys.has(canonicalDuplicateKey(round))) continue;
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
	/** Memory-only upsert of the round into the cached bm25 index (no file write). */
	upsertBm25: (fileName: string, roundData: RecoveredRoundLike) => void;
	/** Commit the batch: write the bm25 index file exactly once after the loop. */
	flushBm25: () => void;
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
	if (report.bm25Indexed > 0) {
		try {
			deps.flushBm25();
		} catch (err) {
			report.errors.push(`bm25 flush: ${(err as Error).message}`);
		}
	}
	return report;
}

export function backfillMissingRounds(
	sessionFiles: string | string[],
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readdirSync" | "readFileSync" | "mkdirSync" | "writeFileSync" | "statSync"> &
		TailReadFs = fs,
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
		// Unit-001: try the bounded tail read first. A complete tail skips the
		// full read+parse entirely; an open tail defers exactly as the full
		// closedness path would; anything inconclusive falls through below.
		const tail = tailRoundComplete(sessionFile, roundsDir, fsImpl);
		if (tail === true) {
			skippedComplete = true;
			continue;
		}
		if (tail === false) {
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

/**
 * Issue #133: at most this many unembedded recovered rounds are embedded
 * inline during session_start; above the threshold the embedding burst is
 * deferred to `just index` (scripts/digest-all.ts) so startup never pays a
 * large OpenRouter embedding bill.
 */
export const STARTUP_EMBED_INLINE_MAX = 10;

export interface StartupEmbedPlan {
	mode: "inline" | "defer";
	pendingCount: number;
}

/**
 * Issue #140 (D1): is this round covered by vector-index rows under the
 * current embedding model? A round is covered when it has at least one index
 * row and none of its rows were written by a different model. Legacy rows
 * without a model column are treated as current (issue #62), matching the
 * `just index` sweep predicate (`existingRounds.has(key) &&
 * !modelMismatchedRounds.has(key)`).
 *
 * Keyed by round-file basename so the start-up counter and the sweep agree on
 * what "this round is indexed" means: index rows, not the promptEmbedding
 * marker.
 */
export function buildCurrentModelRowPredicate(
	entries: readonly VectorIndexEntry[],
	currentModel: string,
): (fileName: string) => boolean {
	const covered = new Set<string>();
	const mismatched = new Set<string>();
	for (const entry of entries) {
		const roundFile = path.basename(indexRoundFileFromPath(entry.filePath));
		if (entry.model !== undefined && entry.model !== currentModel) mismatched.add(roundFile);
		else covered.add(roundFile);
	}
	return (fileName: string) => covered.has(fileName) && !mismatched.has(fileName);
}

/**
 * Issue #133: the status message shown when startup defers the embedding
 * burst. Extracted so the exact wording is pinned by a test.
 */
export function startupEmbedStatusMessage(plan: StartupEmbedPlan): string {
	return `🧠 ${plan.pendingCount} rounds pending embedding backfill — run just index`;
}

/**
 * Issue #133: decide how startup should handle embedding for recovered
 * rounds. A round is "pending" when its round file is missing/unreadable or
 * is not covered by current-model vector-index rows. Pure — no I/O beyond the
 * injected reader/predicate.
 *
 * Issue #140 (D1): the pending count keys on index rows, not the
 * `promptEmbedding` marker. A recovered round may already have current-model
 * rows but no marker (rows written by `just index`/recovery without the
 * marker, e.g. after a crash); counting it pending inflated the startup count
 * and made startup and the `just index` sweep disagree. When
 * `hasCurrentModelRows` is omitted the marker is the only evidence available
 * (a missing/unreadable round is always pending), preserving the previous
 * behaviour for callers with no index loaded.
 */
export function planStartupEmbedding(
	fileNames: readonly string[],
	readRoundData: (fileName: string) => { promptEmbedding?: unknown } | null,
	hasCurrentModelRows?: (fileName: string) => boolean,
): StartupEmbedPlan {
	let pendingCount = 0;
	for (const fileName of fileNames) {
		if (hasCurrentModelRows?.(fileName)) continue;
		const round = readRoundData(fileName);
		if (!round?.promptEmbedding) pendingCount++;
	}
	return {
		mode: pendingCount > STARTUP_EMBED_INLINE_MAX ? "defer" : "inline",
		pendingCount,
	};
}

export async function embedRecoveredRounds(
	fileNames: string[],
	roundsDir: string,
	deps: RecoveredEmbedDeps,
	opts: {
		maxResponseBytes?: number;
		promptNoiseOptions?: PromptNoiseOptions;
		promptMaxTokens?: number;
		hasCurrentModelRows?: (fileName: string) => boolean;
	} = {},
): Promise<{ embedded: string[]; errors: string[] }> {
	const embedded: string[] = [];
	const errors: string[] = [];
	for (const fileName of fileNames) {
		try {
			// Issue #140 (D1): a round covered by current-model index rows is not
			// re-embedded, even when its promptEmbedding marker is missing — the
			// same predicate the start-up count uses (planStartupEmbedding), so the
			// count and the pass cannot disagree. A row written without a marker
			// (e.g. a crash between appends and the marker write) must not spend an
			// embedding call. Omitted by callers with no index loaded, preserving
			// marker-only behaviour.
			if (opts.hasCurrentModelRows?.(fileName)) continue;
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
