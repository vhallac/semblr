import * as fs from "node:fs";
import * as path from "node:path";
import {
	createEmbeddingModelRegistry,
	type EmbeddingClientConfig,
	type EmbeddingModelRegistry,
	getApiKey,
} from "./embedding-client.ts";
import {
	clearForeignBacklog,
	describeForeignShards,
	type ForeignShardDescription,
	recordForeignBacklog,
} from "./foreign-backlog.ts";
import {
	forEachShardLine,
	indexEntryFilename,
	loadShardVectorIndex,
	migrateIndexEntryLine,
	readIndexLines,
	type VectorIndexEntry,
	writeIndexLinesAtomic,
} from "./index-io.ts";
import { adoptOwnShard, discoverIndexShards, LEGACY_INDEX_FILENAME, ownShardPath } from "./index-shards.ts";
import { type AcquireIndexLockDeps, acquireIndexLock, appendIndexRowToShards } from "./index-storage.ts";
import { loadSemblrConfig, type SemblrConfig, type SemblrConfigDeps } from "./semblr-config.ts";

export interface ScriptConfigOptions {
	config?: SemblrConfig;
	configDeps?: SemblrConfigDeps;
}

export interface ScriptEmbeddingOptions extends ScriptConfigOptions {
	apiKey?: string;
	modelRegistry?: EmbeddingModelRegistry;
}

export function resolveScriptConfig(options: ScriptConfigOptions = {}): SemblrConfig {
	return options.config ?? loadSemblrConfig(options.configDeps);
}

export function resolveScriptIndexPath(config: SemblrConfig, roundsDir: string, indexPath?: string): string {
	return indexPath ?? (roundsDir === config.roundsDir ? config.indexPath : path.resolve(roundsDir, "index.csv"));
}

/** fs seam for the script shard context; defaults to node:fs. */
export type ScriptIndexFs = typeof fs;

/**
 * Options for {@link ScriptIndexContext.writeOwnLines}.
 */
export interface WriteOwnLinesOpts {
	/**
	 * The caller already holds the shard append lock (e.g. the
	 * migrate-prompt-embeddings short-lock merge, #107 F2) — skip the
	 * re-acquire, which cannot succeed while the same process holds the
	 * lockfile (PR #151 R1.F1). The caller owns fail-fast on lock exhaustion.
	 */
	lockHeld?: boolean;
}

export interface ScriptIndexContextDeps {
	/** fs seam for shard discovery, adoption, and index writes. */
	fsImpl?: ScriptIndexFs;
	/**
	 * Host id override for tests; defaults to the configured host id
	 * (`config.hostId` — env, pi setting, host-local host id file, or the
	 * machine name; issue #149, OI-6).
	 */
	hostname?: string;
	/**
	 * Name of the rewrite script (e.g. `migrate-model-column`). The
	 * outstanding foreign-shard backlog is keyed by it (issue #149, OI-8):
	 * the entry a host is waiting for is cleared when the host that owns the
	 * shard runs the same script.
	 */
	script?: string;
	/**
	 * Lock-retry knobs for the wholesale-rewrite append lock (PR #151 R1.F1);
	 * defaults to the shared index-writer protocol. Tests shorten the retries.
	 */
	lockDeps?: AcquireIndexLockDeps;
}

/**
 * The shard-aware view of a script's index (issue #149, extension 4.a).
 *
 * - **Reads** see the *merged* view: every row of every discovered shard.
 * - **Writes** go to the host's *own* shard (`index.<hostname>`) — a foreign
 *   shard is never rewritten by this host (D-005), so no row another host
 *   appended can be lost to a rewrite here.
 */
export interface ScriptIndexContext {
	/** Directory holding the index artifacts — the shard root. */
	readonly shardRoot: string;
	/** The resolved index path (legacy `index.csv` location) — reporting only. */
	readonly indexPath: string;
	/** The host's own shard path (`index.<hostname>`). */
	readonly ownShardPath: string;
	/** The discovered shard set, sorted — the merged read view. */
	readonly shards: string[];
	/** Every index line across every shard (merged read view). */
	readMergedLines(): string[];
	/** Parsed vector entries across every shard (corrupt rows are skipped). */
	readMergedEntries(): VectorIndexEntry[];
	/** The host's own shard's lines only. */
	readOwnLines(): string[];
	/**
	 * The rows a wholesale rewrite operates on: the own shard's rows plus the
	 * legacy `index.csv` rows the write would claim by rename. Read-only — the
	 * claim happens in {@link writeOwnLines}.
	 */
	readRewriteRows(): string[];
	/** Foreign shards (never rewritten by this host) with their row counts. */
	foreignShards(): Array<{ shardPath: string; rows: number }>;
	/**
	 * The foreign shards with their row counts and the round files they
	 * reference — the payload of the outstanding backlog entry (OI-8).
	 */
	foreignShardDetails(): ForeignShardDescription[];
	/**
	 * Record the outstanding foreign shards of this run in the backlog state
	 * file (issue #149, OI-8). A no-op when the context has no script name.
	 */
	recordForeignBacklog(): void;
	/**
	 * Clear this host's own backlog entry for the current script — the host
	 * that owns a shard clears its entry by running the rewrite it was
	 * waiting for (issue #149, OI-8).
	 */
	clearOwnBacklog(): void;
	/** Round files a foreign shard still references (the own shard excluded). */
	foreignReferencedFiles(): Set<string>;
	/** Claim a legacy headerless `index.csv` (atomic rename) — returns the own shard path. */
	adopt(): string;
	/** Wholesale rewrite of the own shard (atomic tmp+rename). */
	writeOwnLines(lines: string[], opts?: WriteOwnLinesOpts): void;
	/**
	 * Rewrite the own shard under its append lock. The final read happens
	 * after the lock is acquired, so the transformation sees rows appended
	 * since the caller's detection read and cannot overwrite them.
	 */
	rewriteOwnRows(transform: (rows: string[]) => string[]): void;
	/**
	 * Rename the own shard's rows that reference `oldRoundFile` to
	 * `newRoundFile` (the stale-round-file migration). The rewrite runs under
	 * the shard's append lock and publishes atomically (tmp+rename): a crash
	 * leaves the shard byte-identical, and a session appending concurrently
	 * blocks on the same lockfile and lands after the publish instead of being
	 * overwritten. A foreign shard is never touched.
	 */
	migrateOwnRows(oldRoundFile: string, newRoundFile: string): void;
	/** Append one labeled row to the own shard under the lockfile protocol. */
	appendRow(filePath: string, vector: number[], model?: string, embeddingInputHash?: string): void;
	/** Foreign shards that still hold a row for `filename` (own shard excluded). */
	foreignShardsReferencing(filename: string): string[];
}

/**
 * Resolve the shard set of a script run (issue #149): the shard root is the
 * directory of the resolved index path — the rounds dir in the default
 * configuration and under a custom `--rounds-dir`, the index dir under an
 * explicit `--index-path` (extension 4.a). Reads iterate every discovered
 * shard; writes land in the host's own shard.
 */
export function resolveScriptIndexContext(
	config: SemblrConfig,
	roundsDir: string,
	indexPath?: string,
	deps: ScriptIndexContextDeps = {},
): ScriptIndexContext {
	const fsImpl = deps.fsImpl ?? fs;
	// Issue #149 (OI-6): the own shard is named after the resolved host id,
	// not after the raw machine name, so a renamed host keeps its shard.
	const hostId = deps.hostname ?? config.hostId;
	const resolvedIndexPath = resolveScriptIndexPath(config, roundsDir, indexPath);
	const shardRoot = path.dirname(resolvedIndexPath);
	const own = ownShardPath(shardRoot, hostId);
	const legacyPath = path.join(shardRoot, LEGACY_INDEX_FILENAME);
	const discover = (): string[] => discoverIndexShards(shardRoot, fsImpl);
	// The legacy `index.csv` belongs to this host while it can still claim it
	// (no shard of its own yet): the write renames it into the own shard, so
	// its rows are this host's rows — not a foreign shard's.
	const legacyClaimable = (): boolean => !fsImpl.existsSync(own) && fsImpl.existsSync(legacyPath);
	// The backlog state file is keyed by the script name (OI-8).
	const scriptName = deps.script ?? null;
	// The paths that are not foreign shards: the own shard, and the legacy
	// `index.csv` while this host can still claim it.
	const foreignExcludedPaths = (): string[] => (legacyClaimable() ? [own, legacyPath] : [own]);
	const foreignShardDetails = (): ForeignShardDescription[] =>
		describeForeignShards(shardRoot, foreignExcludedPaths(), fsImpl);
	const readRewriteRows = (): string[] => {
		const ownRows = readIndexLines(own, fsImpl);
		if (legacyClaimable()) {
			return [...ownRows, ...readIndexLines(legacyPath, fsImpl)];
		}
		return ownRows;
	};

	return {
		shardRoot,
		indexPath: resolvedIndexPath,
		ownShardPath: own,
		get shards() {
			return discover();
		},
		readMergedLines: () => {
			const lines: string[] = [];
			forEachShardLine(
				shardRoot,
				(line) => {
					lines.push(line);
				},
				discover(),
				fsImpl,
			);
			return lines;
		},
		readMergedEntries: () => loadShardVectorIndex(shardRoot, { shards: discover(), fsImpl }),
		readOwnLines: () => readIndexLines(own, fsImpl),
		readRewriteRows,
		foreignShards: () =>
			foreignShardDetails().map((shard) => ({
				shardPath: path.join(shardRoot, shard.shard),
				rows: shard.rows,
			})),
		foreignShardDetails: () => foreignShardDetails(),
		recordForeignBacklog: () => {
			if (!scriptName) return;
			try {
				recordForeignBacklog(shardRoot, scriptName, foreignShardDetails(), fsImpl);
			} catch {
				// The backlog is advisory state: a failed record must never
				// break the rewrite it describes.
			}
		},
		clearOwnBacklog: () => {
			if (!scriptName) return;
			try {
				clearForeignBacklog(shardRoot, scriptName, [path.basename(own)], fsImpl);
			} catch {
				// Advisory state — a failed clear must never break the rewrite.
			}
		},
		foreignReferencedFiles: () => {
			const files = new Set<string>();
			for (const shardPath of discover()) {
				if (shardPath === own) continue;
				if (shardPath === legacyPath && legacyClaimable()) continue;
				for (const line of readIndexLines(shardPath, fsImpl)) {
					const entryFile = indexEntryFilename(line);
					if (entryFile !== null) files.add(path.basename(entryFile));
				}
			}
			return files;
		},
		adopt: () => adoptOwnShard(shardRoot, { fsImpl, hostname: hostId }).shardPath,
		writeOwnLines: (lines, opts) => {
			// Nothing to publish and no index artifact of our own (no own
			// shard, no legacy file) — do not create an empty shard file.
			if (lines.length === 0 && !fsImpl.existsSync(own) && !fsImpl.existsSync(legacyPath)) return;
			// Claim a legacy `index.csv` first so its rows are part of the
			// rewritten shard instead of being stranded in a legacy file.
			adoptOwnShard(shardRoot, { fsImpl, hostname: hostId });
			// The append lock serializes this wholesale rewrite against the
			// runtime appends (same `<shard>.lock`): an append landing between
			// the caller's read and the atomic rename would otherwise be
			// dropped. Fail fast when the lock is exhausted — replacing the
			// whole file unsynchronized could drop concurrent rows (PR #151 R1.F1).
			const lock = opts?.lockHeld ? null : acquireIndexLock(own, { ...deps.lockDeps, fsImpl });
			if (!opts?.lockHeld && lock === null) {
				throw new Error(
					`Could not acquire the index lock (${own}.lock) after all retries — the index was NOT written.`,
				);
			}
			try {
				writeIndexLinesAtomic(own, lines, fsImpl);
			} finally {
				lock?.release();
			}
		},
		rewriteOwnRows: (transform) => {
			const hasArtifact = fsImpl.existsSync(own) || fsImpl.existsSync(legacyPath);
			const lock = acquireIndexLock(own, { ...deps.lockDeps, fsImpl });
			if (lock === null) {
				throw new Error(
					`Could not acquire the index lock (${own}.lock) after all retries — the index was NOT written.`,
				);
			}
			try {
				// The transformation runs on the final, lock-protected read.
				// Rows appended after the caller's detection read are therefore
				// present here and are carried into the atomic publish.
				const freshRows = readRewriteRows();
				const rewritten = transform(freshRows);
				if (rewritten.length === 0 && !hasArtifact) return;
				// Claim a legacy `index.csv` after the final read so its rows
				// are part of the rewritten shard instead of being stranded.
				adoptOwnShard(shardRoot, { fsImpl, hostname: hostId });
				writeIndexLinesAtomic(own, rewritten, fsImpl);
			} finally {
				lock.release();
			}
		},
		migrateOwnRows: (oldRoundFile, newRoundFile) => {
			// Nothing to migrate — do not create an empty shard file.
			if (!fsImpl.existsSync(own)) return;
			// The append lock is the same lockfile the runtime appends take
			// (`<shard>.lock`), so the read-modify-write of the migration and a
			// concurrent append serialize: the append waits for the publish
			// instead of racing it (a wholesale rewrite must never write
			// unsynchronized — it could drop the appended rows).
			const lock = acquireIndexLock(own, { fsImpl });
			if (lock === null) {
				throw new Error(
					`Could not acquire the index lock (${own}.lock) after all retries — the index was NOT written.`,
				);
			}
			try {
				const rows = readIndexLines(own, fsImpl).map((line) =>
					migrateIndexEntryLine(line, oldRoundFile, newRoundFile),
				);
				writeIndexLinesAtomic(own, rows, fsImpl);
			} finally {
				lock.release();
			}
		},
		appendRow: (filePath, vector, model, embeddingInputHash) =>
			appendIndexRowToShards(shardRoot, filePath, vector, { fsImpl, hostname: hostId }, model, embeddingInputHash),
		foreignShardsReferencing: (filename) =>
			discover()
				.filter((shardPath) => shardPath !== own)
				.filter((shardPath) =>
					readIndexLines(shardPath, fsImpl).some((line) => {
						const entryFile = indexEntryFilename(line);
						return entryFile !== null && path.basename(entryFile) === filename;
					}),
				),
	};
}

/**
 * One-line notice naming the foreign shards a rewrite left untouched
 * (issue #149, D-005): a wholesale rewrite touches the host's own shard
 * only, so the rows another host holds are updated when that host runs the
 * same script.
 */
export function foreignShardNotice(shards: Array<{ shardPath: string; rows: number }>): string {
	if (shards.length === 0) return "";
	const summary = shards.map((shard) => `${path.basename(shard.shardPath)} (${shard.rows} rows)`).join(", ");
	return `Foreign shards left untouched: ${summary} — run this script on that host to update them.`;
}

/** Log the foreign shards this host left untouched; no-op when there are none. */
export function reportForeignShards(index: ScriptIndexContext, out: Pick<typeof console, "log">): void {
	// Issue #149 (OI-8): the notice is transient, so the outstanding
	// foreign-shard backlog is recorded in the backlog state file. This host
	// clears its own entry first — it just ran the rewrite the foreign hosts
	// were waiting for.
	index.clearOwnBacklog();
	index.recordForeignBacklog();
	const notice = foreignShardNotice(index.foreignShards());
	if (notice) out.log(notice);
}

export function resolveScriptModelRegistry(
	config: SemblrConfig,
	options: Pick<ScriptEmbeddingOptions, "apiKey" | "modelRegistry"> = {},
): EmbeddingModelRegistry {
	return (
		options.modelRegistry ??
		createEmbeddingModelRegistry(config, { agentDir: config.agentDir, apiKey: options.apiKey })
	);
}

export async function resolveScriptApiKey(
	config: SemblrConfig,
	options: ScriptEmbeddingOptions = {},
): Promise<string | null> {
	if (options.apiKey !== undefined) return options.apiKey || null;
	return getApiKey(undefined, { config, modelRegistry: resolveScriptModelRegistry(config, options) });
}

export function scriptEmbeddingConfig(config: SemblrConfig): EmbeddingClientConfig {
	return {
		embeddingProvider: config.embeddingProvider,
		embeddingModel: config.embeddingModel,
		embeddingApiUrl: config.embeddingApiUrl,
		embedTimeoutMs: config.embedTimeoutMs,
		embedMaxRetries: config.embedMaxRetries,
		embedBackoffMs: config.embedBackoffMs,
	};
}
