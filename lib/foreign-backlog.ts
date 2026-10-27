import * as fs from "node:fs";
import * as path from "node:path";
import { indexEntryFilename, readIndexLines } from "./index-io.ts";
import { discoverIndexShards, FOREIGN_BACKLOG_FILENAME } from "./index-shards.ts";
import { acquireIndexLock } from "./index-storage.ts";

/**
 * The outstanding foreign-shard backlog (issue #149, OI-8).
 *
 * A wholesale rewrite touches the host's own shard only (decision 025), so
 * the rows a foreign shard holds are updated when *that* host runs the same
 * script. Until then the work is outstanding — the one-line notice printed at
 * the end of the rewrite is transient (it scrolls away), so the outstanding
 * half of the migration is recorded in a small state file next to the shards.
 *
 * - One entry per (script, foreign shard): the shard name, its row count, the
 *   round files it references, and when it was recorded.
 * - The host that owns a shard clears its own entry when it runs the same
 *   script (the rewrite it was waiting for).
 * - The file is written atomically (tmp+rename, decision 015) under the
 *   writer lockfile, so a concurrent host cannot lose an entry.
 */

/** fs seam for the backlog state file. */
export type ForeignBacklogFs = typeof fs;

/** One outstanding foreign shard, as recorded in the backlog. */
export interface ForeignBacklogEntry {
	/** The rewrite script that left the foreign shard untouched. */
	script: string;
	/** The foreign shard file name (`index.beta`). */
	shard: string;
	/** How many rows the foreign shard still holds. */
	rows: number;
	/** The round files the foreign shard references. */
	roundFiles: string[];
	/** ISO timestamp of the record. */
	recordedAt: string;
}

/** A foreign shard as seen by the recording host. */
export interface ForeignShardDescription {
	/** The shard file name (`index.beta`). */
	shard: string;
	/** How many rows the shard holds. */
	rows: number;
	/** The round files the shard references. */
	roundFiles: string[];
}

/** Absolute path of the backlog state file inside a shard root. */
export function foreignBacklogPath(shardRoot: string): string {
	return path.join(shardRoot, FOREIGN_BACKLOG_FILENAME);
}

function isBacklogEntry(value: unknown): value is ForeignBacklogEntry {
	if (typeof value !== "object" || value === null) return false;
	const entry = value as Partial<ForeignBacklogEntry>;
	return (
		typeof entry.script === "string" &&
		typeof entry.shard === "string" &&
		typeof entry.rows === "number" &&
		Array.isArray(entry.roundFiles) &&
		entry.roundFiles.every((file) => typeof file === "string") &&
		typeof entry.recordedAt === "string"
	);
}

/**
 * Read the recorded backlog. A missing or corrupt file reads as an empty
 * backlog — the backlog is advisory state, and a broken file must never stop
 * a rewrite.
 */
export function readForeignBacklog(shardRoot: string, fsImpl: ForeignBacklogFs = fs): ForeignBacklogEntry[] {
	const file = foreignBacklogPath(shardRoot);
	if (!fsImpl.existsSync(file)) return [];
	try {
		const parsed = JSON.parse(fsImpl.readFileSync(file, "utf-8")) as { entries?: unknown };
		if (!parsed || !Array.isArray(parsed.entries)) return [];
		return parsed.entries.filter(isBacklogEntry);
	} catch {
		return [];
	}
}

/** Write the backlog atomically (tmp+rename, decision 015). */
export function writeForeignBacklog(
	shardRoot: string,
	entries: ForeignBacklogEntry[],
	fsImpl: ForeignBacklogFs = fs,
): void {
	const file = foreignBacklogPath(shardRoot);
	const payload = `${JSON.stringify({ version: 1, entries }, null, 2)}\n`;
	const tmp = `${file}.tmp.${process.pid}`;
	try {
		fsImpl.writeFileSync(tmp, payload);
		fsImpl.renameSync(tmp, file);
	} catch (err) {
		// The publish failed: the old content is intact and the tmp file is
		// litter — remove it best-effort (decision 015).
		try {
			fsImpl.unlinkSync(tmp);
		} catch {}
		throw err;
	}
}

/**
 * Describe the foreign shards of a shard root: their row counts and the round
 * files they reference. `excludedPaths` holds the paths that are not foreign
 * (the host's own shard, and the legacy `index.csv` the host can still claim).
 */
export function describeForeignShards(
	shardRoot: string,
	excludedPaths: readonly string[],
	fsImpl: ForeignBacklogFs = fs,
): ForeignShardDescription[] {
	const excluded = new Set(excludedPaths);
	return discoverIndexShards(shardRoot, fsImpl)
		.filter((shardPath) => !excluded.has(shardPath))
		.map((shardPath) => {
			const lines = readIndexLines(shardPath, fsImpl);
			const roundFiles = new Set<string>();
			for (const line of lines) {
				const entryFile = indexEntryFilename(line);
				if (entryFile !== null) roundFiles.add(path.basename(entryFile));
			}
			return {
				shard: path.basename(shardPath),
				rows: lines.length,
				roundFiles: [...roundFiles].sort(),
			};
		});
}

function withBacklogLock<T>(shardRoot: string, fsImpl: ForeignBacklogFs, run: () => T): T {
	const lock = acquireIndexLock(foreignBacklogPath(shardRoot), { fsImpl });
	if (lock === null) {
		throw new Error(
			`Could not acquire the backlog lock (${foreignBacklogPath(shardRoot)}.lock) after all retries — the backlog was NOT written.`,
		);
	}
	try {
		return run();
	} finally {
		lock.release();
	}
}

/**
 * Record the outstanding foreign shards of a rewrite run. The entries of the
 * given script are replaced wholesale: the run knows the full set of foreign
 * shards it left untouched, so a shard that disappeared (or was adopted) does
 * not keep a stale entry.
 */
export function recordForeignBacklog(
	shardRoot: string,
	script: string,
	shards: readonly ForeignShardDescription[],
	fsImpl: ForeignBacklogFs = fs,
	now: Date = new Date(),
): void {
	withBacklogLock(shardRoot, fsImpl, () => {
		const kept = readForeignBacklog(shardRoot, fsImpl).filter((entry) => entry.script !== script);
		const recordedAt = now.toISOString();
		const added: ForeignBacklogEntry[] = shards.map((shard) => ({
			script,
			shard: shard.shard,
			rows: shard.rows,
			roundFiles: shard.roundFiles,
			recordedAt,
		}));
		writeForeignBacklog(shardRoot, [...kept, ...added], fsImpl);
	});
}

/**
 * Clear the backlog entries of `shards` for `script` — the host that owns a
 * shard clears its own entry when it runs the rewrite it was waiting for.
 * Returns true when an entry was removed.
 */
export function clearForeignBacklog(
	shardRoot: string,
	script: string,
	shards: readonly string[],
	fsImpl: ForeignBacklogFs = fs,
): boolean {
	return withBacklogLock(shardRoot, fsImpl, () => {
		const existing = readForeignBacklog(shardRoot, fsImpl);
		const kept = existing.filter((entry) => !(entry.script === script && shards.includes(entry.shard)));
		if (kept.length === existing.length) return false;
		writeForeignBacklog(shardRoot, kept, fsImpl);
		return true;
	});
}
