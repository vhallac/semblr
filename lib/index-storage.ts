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

export interface LockedAppendDeps {
	fsImpl?: IndexStorageFs;
	lockRetries?: number;
	lockBackoffMs?: number;
	now?: () => number;
	processId?: number;
	staleLockMs?: number;
	wait?: (ms: number) => void;
}

/**
 * Append `line` to `targetPath` using a lockfile-based read-modify-write so
 * concurrent writers (multiple pi sessions) don't clobber each other. Falls
 * back to a plain (unsynchronized) append if the lock can't be acquired
 * after all retries.
 */
export function appendLineWithLock(targetPath: string, dir: string, line: string, deps: LockedAppendDeps = {}): void {
	const fsImpl = deps.fsImpl ?? fs;
	const lockRetries = deps.lockRetries ?? 15;
	const lockBackoffMs = deps.lockBackoffMs ?? 50;
	const now = deps.now ?? Date.now;
	const staleLockMs = deps.staleLockMs ?? 10_000;
	const wait = deps.wait ?? ((ms: number) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms));
	const processId = deps.processId ?? process.pid;
	fsImpl.mkdirSync(dir, { recursive: true });

	const lockPath = `${targetPath}.lock`;

	let lockFd: number | null = null;
	for (let attempt = 0; attempt < lockRetries; attempt++) {
		try {
			lockFd = fsImpl.openSync(lockPath, "wx");
			break;
		} catch {
			try {
				const stat = fsImpl.statSync(lockPath);
				if (now() - stat.mtimeMs > staleLockMs) {
					fsImpl.unlinkSync(lockPath);
					lockFd = fsImpl.openSync(lockPath, "wx");
					break;
				}
			} catch {
				/* lock disappeared or unreadable — retry below */
			}

			if (attempt < lockRetries - 1) {
				const waitMs = lockBackoffMs * 2 ** attempt;
				wait(waitMs);
			} else {
				try {
					fsImpl.appendFileSync(targetPath, line);
				} catch {}
				return;
			}
		}
	}

	try {
		const existing = fsImpl.existsSync(targetPath) ? fsImpl.readFileSync(targetPath, "utf-8") : "";
		const newContent = existing + line;
		const tmp = `${targetPath}.tmp.${processId}`;
		fsImpl.writeFileSync(tmp, newContent);
		fsImpl.renameSync(tmp, targetPath);
	} finally {
		if (lockFd !== null) {
			fsImpl.closeSync(lockFd);
			try {
				fsImpl.unlinkSync(lockPath);
			} catch {}
		}
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
