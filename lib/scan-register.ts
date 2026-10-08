/**
 * scan-register.ts — per-session-dir record of the last fully-scanned mtime.
 *
 * Startup backfill (issue #133) otherwise re-opens every prior session JSONL
 * on each session_start. This register stores, per session directory, the max
 * source mtime seen by the last full scan; on the next startup, files with
 * mtime <= cutoff are never opened (the round files for them are already on
 * disk). All timestamps are EPOCH MILLISECONDS ONLY (Date.now()/mtimeMs) —
 * UTC-invariant by definition; no local-time string formatting anywhere.
 *
 * A corrupt or missing register degrades to "no cutoff" (full scan), which is
 * the pre-register behavior.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

export interface ScanRegisterFs
	extends Pick<typeof fs, "mkdirSync" | "readFileSync" | "writeFileSync" | "existsSync"> {}

/** Register file for one session dir, keyed by a hash of its resolved path. */
export function scanRegisterPath(stateDir: string, sessionDir: string): string {
	const key = crypto.createHash("sha256").update(path.resolve(sessionDir)).digest("hex").slice(0, 16);
	return path.join(stateDir, `scan-register-${key}.json`);
}

/**
 * Read the cutoff (epoch ms) recorded for this session dir. Null when the
 * register is missing or corrupt — the caller then runs a full scan.
 */
export function loadScanCutoff(stateDir: string, sessionDir: string, fsImpl: ScanRegisterFs = fs): number | null {
	try {
		const raw = fsImpl.readFileSync(scanRegisterPath(stateDir, sessionDir), "utf-8");
		const parsed: unknown = JSON.parse(raw);
		if (
			typeof parsed !== "object" ||
			parsed === null ||
			typeof (parsed as { lastFullScanMtime?: unknown }).lastFullScanMtime !== "number"
		) {
			return null;
		}
		const mtime = (parsed as { lastFullScanMtime: number }).lastFullScanMtime;
		return Number.isFinite(mtime) ? mtime : null;
	} catch {
		return null;
	}
}

/**
 * Max mtime (epoch ms) over the given files; null when none are statable.
 * Files that can't be stat are skipped (they can't set a safe cutoff).
 */
export function computeMaxMtime(files: readonly string[], fsImpl: Pick<typeof fs, "statSync"> = fs): number | null {
	let max: number | null = null;
	for (const file of files) {
		try {
			const mtime = fsImpl.statSync(file).mtimeMs;
			if (Number.isFinite(mtime) && (max === null || mtime > max)) max = mtime;
		} catch {}
	}
	return max;
}

/** Persist the cutoff for this session dir. Best-effort: failures are ignored. */
export function saveScanCutoff(
	stateDir: string,
	sessionDir: string,
	lastFullScanMtime: number,
	fsImpl: ScanRegisterFs = fs,
): void {
	try {
		fsImpl.mkdirSync(stateDir, { recursive: true });
		fsImpl.writeFileSync(
			scanRegisterPath(stateDir, sessionDir),
			JSON.stringify({ lastFullScanMtime: Math.floor(lastFullScanMtime) }, null, 2),
		);
	} catch {}
}
