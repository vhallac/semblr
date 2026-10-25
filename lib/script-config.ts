import * as fs from "node:fs";
import * as path from "node:path";
import {
	createEmbeddingModelRegistry,
	type EmbeddingClientConfig,
	type EmbeddingModelRegistry,
	getApiKey,
} from "./embedding-client.ts";
import {
	forEachShardLine,
	indexEntryFilename,
	loadShardVectorIndex,
	readIndexLines,
	type VectorIndexEntry,
	writeIndexLinesAtomic,
} from "./index-io.ts";
import { adoptOwnShard, discoverIndexShards, LEGACY_INDEX_FILENAME, ownShardPath } from "./index-shards.ts";
import { appendIndexRowToShards } from "./index-storage.ts";
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

export interface ScriptIndexContextDeps {
	/** fs seam for shard discovery, adoption, and index writes. */
	fsImpl?: ScriptIndexFs;
	/** Hostname override for tests; defaults to os.hostname(). */
	hostname?: string;
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
	/** Round files a foreign shard still references (the own shard excluded). */
	foreignReferencedFiles(): Set<string>;
	/** Claim a legacy headerless `index.csv` (atomic rename) — returns the own shard path. */
	adopt(): string;
	/** Wholesale rewrite of the own shard (atomic tmp+rename). */
	writeOwnLines(lines: string[]): void;
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
	const resolvedIndexPath = resolveScriptIndexPath(config, roundsDir, indexPath);
	const shardRoot = path.dirname(resolvedIndexPath);
	const own = ownShardPath(shardRoot, deps.hostname);
	const legacyPath = path.join(shardRoot, LEGACY_INDEX_FILENAME);
	const discover = (): string[] => discoverIndexShards(shardRoot, fsImpl);
	// The legacy `index.csv` belongs to this host while it can still claim it
	// (no shard of its own yet): the write renames it into the own shard, so
	// its rows are this host's rows — not a foreign shard's.
	const legacyClaimable = (): boolean => !fsImpl.existsSync(own) && fsImpl.existsSync(legacyPath);

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
		readRewriteRows: () => {
			const ownRows = readIndexLines(own, fsImpl);
			// The legacy file is claimed by the write (rename) only when the host
			// has no shard of its own; otherwise it stays a separate shard and
			// its rows are not part of this host's rewrite.
			if (legacyClaimable()) {
				return [...ownRows, ...readIndexLines(legacyPath, fsImpl)];
			}
			return ownRows;
		},
		foreignShards: () =>
			discover()
				.filter((shardPath) => shardPath !== own)
				.filter((shardPath) => !(shardPath === legacyPath && legacyClaimable()))
				.map((shardPath) => ({ shardPath, rows: readIndexLines(shardPath, fsImpl).length })),
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
		adopt: () => adoptOwnShard(shardRoot, { fsImpl, hostname: deps.hostname }).shardPath,
		writeOwnLines: (lines) => {
			// Nothing to publish and no index artifact of our own (no own
			// shard, no legacy file) — do not create an empty shard file.
			if (lines.length === 0 && !fsImpl.existsSync(own) && !fsImpl.existsSync(legacyPath)) return;
			// Claim a legacy `index.csv` first so its rows are part of the
			// rewritten shard instead of being stranded in a legacy file.
			adoptOwnShard(shardRoot, { fsImpl, hostname: deps.hostname });
			writeIndexLinesAtomic(own, lines, fsImpl);
		},
		appendRow: (filePath, vector, model, embeddingInputHash) =>
			appendIndexRowToShards(
				shardRoot,
				filePath,
				vector,
				{ fsImpl, hostname: deps.hostname },
				model,
				embeddingInputHash,
			),
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
