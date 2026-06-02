import * as fs from "node:fs";

export interface IndexEntry {
	filePath: string;
	vector: number[];
}

export function loadIndexFromPath(
	indexPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): IndexEntry[] {
	if (!fsImpl.existsSync(indexPath)) return [];
	const raw = fsImpl.readFileSync(indexPath, "utf-8").trim();
	if (!raw) return [];
	return raw.split("\n").map((line) => {
		const comma = line.indexOf(",");
		const b64 = line.slice(0, comma);
		const filePath = line.slice(comma + 1);
		const decoded = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
		return { filePath, vector: Array.isArray(decoded) ? decoded : [] };
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
	return new Set(index.map((e) => e.filePath.replace(/(:prompt|:response|:round)$/, ""))).size;
}

export function buildSessionStartStatus(index: readonly { filePath: string }[]): string {
	return `🧠 semblr loaded — ${countUniqueIndexedRounds(index)} rounds indexed`;
}

type IndexStorageFs = Pick<
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
) {
	const fsImpl = deps.fsImpl ?? fs;
	const lockRetries = deps.lockRetries ?? 15;
	const lockBackoffMs = deps.lockBackoffMs ?? 50;
	const now = deps.now ?? Date.now;
	const staleLockMs = deps.staleLockMs ?? 10_000;
	const wait = deps.wait ?? ((ms: number) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms));
	const processId = deps.processId ?? process.pid;
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	const line = `${b64},${filePath}\n`;
	fsImpl.mkdirSync(roundsDir, { recursive: true });

	const lockPath = `${indexPath}.lock`;

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
					fsImpl.appendFileSync(indexPath, line);
				} catch {}
				return;
			}
		}
	}

	try {
		const existing = fsImpl.existsSync(indexPath) ? fsImpl.readFileSync(indexPath, "utf-8") : "";
		const newContent = existing + line;
		const tmp = `${indexPath}.tmp.${processId}`;
		fsImpl.writeFileSync(tmp, newContent);
		fsImpl.renameSync(tmp, indexPath);
	} finally {
		if (lockFd !== null) {
			fsImpl.closeSync(lockFd);
			try {
				fsImpl.unlinkSync(lockPath);
			} catch {}
		}
	}
}
