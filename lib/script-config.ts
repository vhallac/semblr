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
import { adoptOwnShard, discoverIndexShards, ownShardPath } from "./index-shards.ts";
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
	const discover = (): string[] => discoverIndexShards(shardRoot, fsImpl);

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
		adopt: () => adoptOwnShard(shardRoot, { fsImpl, hostname: deps.hostname }).shardPath,
		writeOwnLines: (lines) => {
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
