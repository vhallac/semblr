import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { atomicWriteFileSync } from "./atomic-write.ts";

/**
 * Host identity (issue #149, OI-6).
 *
 * The index shard of a host is named after its host id. The id is resolved
 * in this order:
 *
 * 1. a configured id (env var `SEMBLR_HOST_ID` or the `hostId` pi setting),
 * 2. the host-local host id file `<agentDir>/semblr/host-id`,
 * 3. `os.hostname()`.
 *
 * The machine name is persisted to the host-local file on first use, so a
 * renamed machine (or a new container id) keeps writing to the same shard
 * instead of orphaning it as a permanent foreign shard. The file is
 * host-local by design: the rounds directory is synced between hosts, so a
 * marker file inside it would give every host the same id.
 */

/** Filename of the host-local host id file (under `<agentDir>/semblr/`). */
export const HOST_ID_FILENAME = "host-id";

/**
 * fs surface needed to read and persist the host id file. The write methods
 * are optional: without them the host id is read but not persisted.
 */
export interface HostIdFs {
	existsSync(filePath: string): boolean;
	readFileSync(filePath: string, encoding: "utf-8"): string;
	writeFileSync?(filePath: string, data: string): void;
	mkdirSync?(dirPath: string, options: { recursive: boolean }): void;
	renameSync?(oldPath: string, newPath: string): void;
}

/**
 * Path of the host-local host id file: `<agentDir>/semblr/host-id`.
 * The agent dir is host-local (default `~/.pi/agent`), never the synced
 * rounds directory.
 */
export function hostIdFilePath(agentDir: string): string {
	return path.join(agentDir, "semblr", HOST_ID_FILENAME);
}

/**
 * A usable host id: non-empty, not a path traversal, and free of path
 * separators and control characters (the id becomes a filename component).
 * Returns `undefined` for an unusable value.
 */
export function sanitizeHostId(value: string | undefined | null): string | undefined {
	if (value === undefined || value === null) return undefined;
	const trimmed = value.trim();
	if (!trimmed || trimmed === "." || trimmed === "..") return undefined;
	// biome-ignore lint/suspicious/noControlCharactersInRegex: control characters are rejected on purpose
	if (/[/\\\u0000-\u001f]/.test(trimmed)) return undefined;
	return trimmed;
}

export interface ResolveHostIdDeps {
	/** Host id configured through the env var or the pi setting. */
	configuredHostId?: string;
	/** Agent dir — the host id file lives under `<agentDir>/semblr/`. */
	agentDir: string;
	/** Machine-name fallback; defaults to `os.hostname()`. */
	hostname?: string;
	/**
	 * fs seam. When it cannot write (a read-only test seam), the resolved id
	 * is still returned — it is simply not persisted.
	 */
	fsImpl?: HostIdFs;
	/** Warning sink for a rejected configured id and a failed persist. */
	warn?: (message: string) => void;
}

export interface ResolvedHostId {
	/** The resolved host id — the shard name component. */
	hostId: string;
	/** Where the id came from. */
	source: "configured" | "file" | "hostname";
	/** True when this call wrote the host id file. */
	persisted: boolean;
}

/**
 * Resolve the host id and persist the machine-name fallback on first use.
 *
 * A configured id wins and is never written to the file (the configuration
 * is the source of truth). The saved file wins over the machine name, so a
 * renamed machine keeps its shard. The machine name is saved when no id is
 * configured and no file exists yet.
 */
export function resolveHostId(deps: ResolveHostIdDeps): ResolvedHostId {
	const warn = deps.warn ?? console.warn;
	const fsImpl = deps.fsImpl;

	const configured = sanitizeHostId(deps.configuredHostId);
	if (configured === undefined && deps.configuredHostId !== undefined) {
		warn(`Invalid Semblr host id ${JSON.stringify(deps.configuredHostId)} — using the machine name instead.`);
	}
	if (configured !== undefined) return { hostId: configured, source: "configured", persisted: false };

	const filePath = hostIdFilePath(deps.agentDir);
	if (fsImpl?.existsSync(filePath)) {
		try {
			const saved = sanitizeHostId(fsImpl.readFileSync(filePath, "utf-8"));
			if (saved !== undefined) return { hostId: saved, source: "file", persisted: false };
		} catch (err) {
			warn(`Failed to read the host id file ${filePath}: ${err instanceof Error ? err.message : String(err)}`);
		}
	}

	const hostname = deps.hostname ?? os.hostname();
	const persisted = persistHostId(filePath, hostname, fsImpl, warn);
	return { hostId: hostname, source: "hostname", persisted };
}

/**
 * Persist the host id to the host-local file (atomic tmp+rename, decision
 * 015). A failed write is reported, not fatal: the id still names the shard
 * for this run.
 */
function persistHostId(
	filePath: string,
	hostId: string,
	fsImpl: HostIdFs | undefined,
	warn: (message: string) => void,
): boolean {
	if (!fsImpl?.writeFileSync || !fsImpl.mkdirSync || !fsImpl.renameSync) return false;
	try {
		fsImpl.mkdirSync(path.dirname(filePath), { recursive: true });
		atomicWriteFileSync(filePath, `${hostId}\n`, {
			writeFileSync: fsImpl.writeFileSync,
			renameSync: fsImpl.renameSync,
		});
		return true;
	} catch (err) {
		warn(`Failed to save the host id to ${filePath}: ${err instanceof Error ? err.message : String(err)}`);
		return false;
	}
}
