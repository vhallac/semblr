import * as fs from "node:fs";
import { indexRoundFileFromPath, splitVectorIndexMetadata } from "./index-io.ts";

export interface IndexEntry {
	filePath: string;
	vector: number[];
	model?: string;
	/** Optional 4th CSV column: embedding-input hash stamp (see lib/index-io.ts). */
	embeddingInputHash?: string;
}

export function loadIndexFromPath(
	indexPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): IndexEntry[] {
	if (!fsImpl.existsSync(indexPath)) return [];
	const raw = fsImpl.readFileSync(indexPath, "utf-8").trim();
	if (!raw) return [];
	return raw.split("\n").map((line) => {
		const firstComma = line.indexOf(",");
		const b64 = line.slice(0, firstComma);
		const decoded = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
		// Metadata parsed right-to-left so the optional 4th column (embeddingInputHash)
		// does not corrupt the model field (search groups entries by model).
		const { filePath, model, embeddingInputHash } = splitVectorIndexMetadata(line.slice(firstComma + 1));
		const entry: IndexEntry = { filePath, vector: Array.isArray(decoded) ? decoded : [] };
		if (model !== undefined) entry.model = model;
		if (embeddingInputHash !== undefined) entry.embeddingInputHash = embeddingInputHash;
		return entry;
	});
}

export function loadSessionStartIndex(
	indexPath: string,
	deps: { existsSync?: (filePath: string) => boolean; loadIndex?: () => IndexEntry[] } = {},
): IndexEntry[] {
	const existsSync = deps.existsSync ?? fs.existsSync;
	const load = deps.loadIndex ?? (() => loadIndexFromPath(indexPath));
	return existsSync(indexPath) ? load() : [];
}

export function countUniqueIndexedRounds(index: readonly { filePath: string }[]): number {
	return new Set(index.map((e) => indexRoundFileFromPath(e.filePath))).size;
}

export function buildSessionStartStatus(index: readonly { filePath: string }[]): string {
	return `🧠 semblr loaded — ${countUniqueIndexedRounds(index)} rounds indexed`;
}

export type IndexStorageFs = Pick<
	typeof fs,
	| "appendFileSync"
	| "closeSync"
	| "existsSync"
	| "mkdirSync"
	| "openSync"
	| "readFileSync"
	| "renameSync"
	| "statSync"
	| "unlinkSync"
	| "writeFileSync"
>;

export interface AcquireIndexLockDeps {
	fsImpl?: IndexStorageFs;
	lockRetries?: number;
	lockBackoffMs?: number;
	now?: () => number;
	staleLockMs?: number;
	wait?: (ms: number) => void;
}

export interface IndexLockHandle {
	/** Path of the acquired lockfile (`<targetPath>.lock`). */
	lockPath: string;
	/** Close the lockfile fd and unlink the lockfile (best-effort). */
	release: () => void;
}

/**
 * Acquire `<targetPath>.lock` using the shared index-writer protocol: exclusive
 * (wx) create, retry with exponential backoff, stale-lock takeover (default 10s).
 * All index writers — runtime appends and wholesale rewrites (migration) — must
 * go through this so they serialize on the same lockfile.
 *
 * Returns null when the lock cannot be acquired after all retries; the caller
 * decides the failure policy: the runtime append falls back to an
 * unsynchronized single-line append, a wholesale rewrite must fail fast
 * instead (replacing the whole file unsynchronized could drop concurrent rows).
 */
export function acquireIndexLock(targetPath: string, deps: AcquireIndexLockDeps = {}): IndexLockHandle | null {
	const fsImpl = deps.fsImpl ?? fs;
	const lockRetries = deps.lockRetries ?? 15;
	const lockBackoffMs = deps.lockBackoffMs ?? 50;
	const now = deps.now ?? Date.now;
	const staleLockMs = deps.staleLockMs ?? 10_000;
	const wait = deps.wait ?? ((ms: number) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms));
	const lockPath = `${targetPath}.lock`;

	const handle = (fd: number): IndexLockHandle => ({
		lockPath,
		release: () => {
			fsImpl.closeSync(fd);
			try {
				fsImpl.unlinkSync(lockPath);
			} catch {}
		},
	});

	for (let attempt = 0; attempt < lockRetries; attempt++) {
		try {
			return handle(fsImpl.openSync(lockPath, "wx"));
		} catch {
			try {
				const stat = fsImpl.statSync(lockPath);
				if (now() - stat.mtimeMs > staleLockMs) {
					fsImpl.unlinkSync(lockPath);
					return handle(fsImpl.openSync(lockPath, "wx"));
				}
			} catch {
				/* lock disappeared or unreadable — retry below */
			}

			if (attempt < lockRetries - 1) {
				wait(lockBackoffMs * 2 ** attempt);
			}
		}
	}
	return null;
}

export interface LockedAppendDeps extends AcquireIndexLockDeps {
	processId?: number;
}

/**
 * Append `line` to `targetPath` using a lockfile-based read-modify-write so
 * concurrent writers (multiple pi sessions) don't clobber each other. Falls
 * back to a plain (unsynchronized) append if the lock can't be acquired
 * after all retries.
 */
export function appendLineWithLock(targetPath: string, dir: string, line: string, deps: LockedAppendDeps = {}): void {
	const fsImpl = deps.fsImpl ?? fs;
	const processId = deps.processId ?? process.pid;
	fsImpl.mkdirSync(dir, { recursive: true });

	const lock = acquireIndexLock(targetPath, deps);
	if (lock === null) {
		// Lock exhausted: fall back to an unsynchronized append of this single
		// line. Wholesale rewrites (migration) must fail fast instead.
		try {
			fsImpl.appendFileSync(targetPath, line);
		} catch {}
		return;
	}

	try {
		const existing = fsImpl.existsSync(targetPath) ? fsImpl.readFileSync(targetPath, "utf-8") : "";
		const newContent = existing + line;
		const tmp = `${targetPath}.tmp.${processId}`;
		fsImpl.writeFileSync(tmp, newContent);
		fsImpl.renameSync(tmp, targetPath);
	} finally {
		lock.release();
	}
}

export interface AppendIndexDeps {
	fsImpl?: IndexStorageFs;
	lockRetries?: number;
	lockBackoffMs?: number;
	now?: () => number;
	processId?: number;
	staleLockMs?: number;
	wait?: (ms: number) => void;
}

export function appendToIndexPath(
	indexPath: string,
	roundsDir: string,
	filePath: string,
	vector: number[],
	deps: AppendIndexDeps = {},
	model?: string,
	embeddingInputHash?: string,
) {
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	const parts = [b64, filePath];
	if (model !== undefined) parts.push(model);
	if (embeddingInputHash !== undefined) parts.push(embeddingInputHash);
	appendLineWithLock(indexPath, roundsDir, `${parts.join(",")}\n`, deps);
}
