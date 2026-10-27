import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

/**
 * Per-host index shards (issue #149): the round index is split into one file
 * per host (`index.<hostname>`) instead of a single shared `index.csv`, so
 * synced multi-host stores never write-conflict on one artifact.
 *
 * - Writes go only to the host's own shard; reads iterate every shard.
 * - Legacy adoption (Q1, option b): a host that finds a headerless
 *   `index.csv` and no shard of its own renames it to `index.<hostname>`
 *   and takes ownership (atomic same-directory rename, decision 015).
 * - No shard headers (Q2, option 3): the hostname lives in the filename;
 *   shard files stay byte-compatible with the legacy index format.
 * - The legacy `index.csv` is part of the read shard set until it is
 *   adopted (renamed away), so rows appended between adoption checks are
 *   never skipped by readers.
 */

/** Legacy headerless index filename, adopted by rename on first write. */
export const LEGACY_INDEX_FILENAME = "index.csv";

/** Fulltext tool-call index, sibling artifact — never a shard. */
const TOOL_INDEX_FILENAME = "index-tools.fulltext.csv";

/** BM25 index, sibling artifact — never a shard. */
const BM25_INDEX_FILENAME = "index.bm25.json";

/**
 * Outstanding foreign-shard backlog (issue #149, OI-8) — a state file, never
 * a shard. It lives in the shard root next to the shards and is excluded
 * from shard discovery, like the other sibling artifacts.
 */
export const FOREIGN_BACKLOG_FILENAME = "index.backlog.json";

/**
 * Migration backup artifacts — never a shard. Covers the legacy
 * `index.csv.bak.<ts>` form and the `index.<host>.bak.<ts>` /
 * `index.csv.bak-<ts>` variants the migration scripts write.
 */
const BACKUP_PATTERN = /^index\..*\.bak(?:[.-]|$)/;

/** Lockfile protocol artifact (`<shard>.lock`) — never a shard. */
const LOCK_PATTERN = /^index\..+\.lock$/;

/** Atomic-write temp artifact (`<shard>.tmp.<pid>`) — never a shard. */
const TMP_PATTERN = /^index\..+\.tmp\./;

/**
 * Classify a rounds-directory filename as an index shard. Shards are
 * `index.<name>` files (`index.csv` = legacy, included for read tolerance);
 * sibling artifacts, migration backups, lockfiles, and atomic-write temps
 * are explicitly excluded (issue #149 extension 3.a).
 */
export function isIndexShardFilename(filename: string): boolean {
	if (!filename.startsWith("index.")) return false;
	if (filename === TOOL_INDEX_FILENAME || filename === BM25_INDEX_FILENAME) return false;
	if (filename === FOREIGN_BACKLOG_FILENAME) return false;
	if (BACKUP_PATTERN.test(filename) || LOCK_PATTERN.test(filename) || TMP_PATTERN.test(filename)) return false;
	return true;
}

/**
 * fs surface needed for shard discovery and legacy adoption. Only the
 * narrow slice both operations use — so storage-side fs implementations
 * (which lack `readdirSync`) can be injected without adaptation.
 */
export interface IndexShardsFs {
	existsSync(filePath: string): boolean;
	readdirSync(filePath: string): string[];
	renameSync(oldPath: string, newPath: string): void;
}

/** fs surface needed for legacy adoption alone (no directory listing). */
export interface AdoptShardFs {
	existsSync(filePath: string): boolean;
	renameSync(oldPath: string, newPath: string): void;
}

/**
 * The shard file names of a shard set (basenames), in the given order.
 * Used by scripts that report which shards they read (issue #149).
 */
export function shardNames(shards: readonly string[]): string[] {
	return shards.map((shardPath) => path.basename(shardPath));
}

/**
 * One-line, human-readable label of a shard set: the shard file names joined
 * by commas, or "none (empty shard set)" when the set is empty (issue #149).
 */
export function describeShardSet(shards: readonly string[]): string {
	const names = shardNames(shards);
	return names.length === 0 ? "none (empty shard set)" : names.join(", ");
}

/**
 * Absolute path of the host's own shard file (`index.<hostname>`) inside
 * `roundsDir`. Hostname is injectable for tests; it is only ever used as a
 * filename component, never as a CSV column (assumption A-2).
 */
export function ownShardPath(roundsDir: string, hostname: string = os.hostname()): string {
	return path.join(roundsDir, `index.${hostname}`);
}

/**
 * Discover the shard set of `roundsDir`: every index shard file present,
 * sorted for deterministic merge order. Returns an empty list for a missing
 * or empty rounds directory (issue #149 extension 1.b — reads tolerate an
 * empty shard set). The legacy `index.csv` is included until adopted.
 */
export function discoverIndexShards(roundsDir: string, fsImpl: IndexShardsFs = fs): string[] {
	if (!fsImpl.existsSync(roundsDir)) return [];
	return fsImpl
		.readdirSync(roundsDir)
		.filter(isIndexShardFilename)
		.sort()
		.map((filename) => path.join(roundsDir, filename));
}

export interface AdoptOwnShardDeps {
	fsImpl?: AdoptShardFs;
	hostname?: string;
}

export interface AdoptOwnShardResult {
	/** Path the host must append to from now on. */
	shardPath: string;
	/** True when this call performed the legacy adoption rename. */
	adoptedLegacy: boolean;
}

/**
 * Resolve the host's own shard, adopting a legacy headerless `index.csv` on
 * first write (issue #149 Q1 option b): when the legacy file exists and the
 * host has no shard of its own, it is renamed to `index.<hostname>` — atomic
 * same-directory rename (decision 015). A failed rename (the legacy file was
 * claimed by another host between the check and the rename) is a no-op: the
 * later host simply finds nothing to claim. When the host's own shard already
 * exists, the legacy file is left untouched — renaming would clobber the
 * shard with POSIX rename-overwrite, and the orphan stays readable via the
 * legacy-tolerant shard set.
 */
export function adoptOwnShard(roundsDir: string, deps: AdoptOwnShardDeps = {}): AdoptOwnShardResult {
	const fsImpl = deps.fsImpl ?? fs;
	const shardPath = ownShardPath(roundsDir, deps.hostname);
	const legacyPath = path.join(roundsDir, LEGACY_INDEX_FILENAME);
	if (!fsImpl.existsSync(legacyPath) || fsImpl.existsSync(shardPath)) {
		return { shardPath, adoptedLegacy: false };
	}
	try {
		fsImpl.renameSync(legacyPath, shardPath);
		return { shardPath, adoptedLegacy: true };
	} catch {
		// Lost the adoption race: the legacy file is gone (claimed by another
		// host) — proceed with the host's own (possibly new) shard.
		return { shardPath, adoptedLegacy: false };
	}
}
