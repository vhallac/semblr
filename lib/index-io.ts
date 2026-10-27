import * as fs from "node:fs";
import * as path from "node:path";
import { computeContentHash, type HashToolCallDetail } from "./hash.ts";
import { discoverIndexShards, ownShardPath } from "./index-shards.ts";
import { type AcquireIndexLockDeps, acquireIndexLock, type IndexStorageFs } from "./index-storage.ts";

export interface VectorIndexEntry {
	vector: number[];
	filePath: string;
	model?: string;
	/**
	 * Optional 4th CSV column: sha256-base64url of the exact prompt-side embedding
	 * input text the vector was computed from (issue #106 re-embed migration stamp).
	 * Only meaningful on :prompt rows; consumers must treat it as opaque metadata.
	 *
	 * Provenance contract (#107 F1): a plain stamp asserts the vector was computed
	 * from the exact input it hashes, and is only written when that derivation was
	 * verifiable at write time (post-#107-F3 capture, digest re-embeds, or the
	 * migration's post-F3 re-embed path). Rows captured earlier have no verifiable
	 * capture input (the hook embedded its augmented prompt), so the migration
	 * restamps them with an `assumed-` marker after re-embedding over the round-file
	 * derivation — see `ASSUMED_STAMP_PREFIX`.
	 */
	embeddingInputHash?: string;
}

interface RoundHashContent {
	userPrompt?: string;
	responseSequence?: string;
	toolCalls?: HashToolCallDetail[];
}

export function encodeVectorIndexLine(
	vector: number[],
	filePath: string,
	model?: string,
	embeddingInputHash?: string,
): string {
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	const parts = [b64, filePath];
	if (model !== undefined) parts.push(model);
	if (embeddingInputHash !== undefined) parts.push(embeddingInputHash);
	return parts.join(",");
}

/**
 * Split an index line's metadata columns (everything after the vector column).
 * Metadata columns (filePath, model, embeddingInputHash) contain no commas
 * themselves (round filenames are content hashes; model slugs and hashes are
 * comma-free), so the trailing segments map positionally from the right:
 *   [filePath] | [filePath, model] | [filePath, model, embeddingInputHash]
 */
export function splitVectorIndexMetadata(rest: string): {
	filePath: string;
	model?: string;
	embeddingInputHash?: string;
} {
	const segments = rest.split(",");
	if (segments.length >= 3) {
		return {
			filePath: segments.slice(0, segments.length - 2).join(","),
			model: segments[segments.length - 2],
			embeddingInputHash: segments[segments.length - 1],
		};
	}
	if (segments.length === 2) {
		return { filePath: segments[0], model: segments[1] };
	}
	return { filePath: segments[0] ?? "" };
}

/**
 * Prefix marking an assumed-provenance embedding-input stamp (#107 F1): the vector
 * was computed over the assumed cleaned-raw derivation from round.json, not over the
 * original capture input (which is unverifiable for rows captured before #107 F3 —
 * the capture hook embedded its augmented prompt). The re-embed migration treats a
 * matching `assumed-<hash>` stamp as proof the row was already swept; plain stamps
 * on such rows are untrusted, because a false stamp and a digest-corrected one are
 * byte-identical. The prefix is ASCII and the hash is base64url, so the combined
 * stamp stays CSV-safe.
 */
export const ASSUMED_STAMP_PREFIX = "assumed-";

export function makeAssumedStamp(hash: string): string {
	return `${ASSUMED_STAMP_PREFIX}${hash}`;
}

/** Strip the assumed-provenance prefix; any other stamp passes through unchanged. */
export function bareEmbeddingInputHash(stamp: string): string {
	return stamp.startsWith(ASSUMED_STAMP_PREFIX) ? stamp.slice(ASSUMED_STAMP_PREFIX.length) : stamp;
}

function parseVectorIndexLine(line: string): VectorIndexEntry {
	const firstComma = line.indexOf(",");
	const b64 = line.slice(0, firstComma);
	const vector = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
	// Metadata columns are parsed right-to-left so the optional 4th column
	// (embeddingInputHash) does not corrupt the model or filePath fields.
	const { filePath, model, embeddingInputHash } = splitVectorIndexMetadata(line.slice(firstComma + 1));
	const entry: VectorIndexEntry = { vector, filePath };
	if (model !== undefined) entry.model = model;
	if (embeddingInputHash !== undefined) entry.embeddingInputHash = embeddingInputHash;
	return entry;
}

export function readIndexLines(indexPath: string, fsImpl: IndexReadFs = fs): string[] {
	const lines: string[] = [];
	forEachIndexLine(indexPath, fsImpl, (line) => {
		lines.push(line);
	});
	return lines;
}

/** fs surface needed for chunked reading of the vector index. */
export interface IndexReadFs {
	existsSync(filePath: string): boolean;
	openSync(filePath: string, flags: string): number;
	readSync(fd: number, buffer: Buffer, offset: number, length: number, position: number | null): number;
	closeSync(fd: number): void;
}

/** Default chunk size for index reads: large enough to be fast, well under the string-length limit. */
export const INDEX_CHUNK_BYTES = 8 * 1024 * 1024;

/**
 * Invoke `onLine` for every non-blank line of the vector index using
 * fixed-size chunked reads instead of a single readFileSync: the file can
 * exceed Node's ~512MB max string length, which made wholesale loading fail
 * with ERR_STRING_TOO_LONG.
 */
export function forEachIndexLine(
	indexPath: string,
	fsImpl: IndexReadFs,
	onLine: (line: string) => void,
	chunkSize: number = INDEX_CHUNK_BYTES,
): void {
	if (!fsImpl.existsSync(indexPath)) return;
	const chunk = Buffer.allocUnsafe(chunkSize);
	let carry: Buffer = Buffer.alloc(0);
	const fd = fsImpl.openSync(indexPath, "r");
	try {
		for (;;) {
			const bytesRead = fsImpl.readSync(fd, chunk, 0, chunk.length, null);
			if (bytesRead <= 0) break;
			const data = bytesRead === chunk.length ? Buffer.from(chunk) : Buffer.from(chunk.subarray(0, bytesRead));
			const buf = carry.length > 0 ? Buffer.concat([carry, data]) : data;
			const lastNewline = buf.lastIndexOf(0x0a);
			if (lastNewline === -1) {
				carry = buf;
				continue;
			}
			carry = buf.subarray(lastNewline + 1);
			for (const line of buf.subarray(0, lastNewline).toString("utf-8").split("\n")) {
				const trimmed = line.trim();
				if (trimmed) onLine(trimmed);
			}
		}
		if (carry.length > 0) {
			const trimmed = carry.toString("utf-8").trim();
			if (trimmed) onLine(trimmed);
		}
	} finally {
		fsImpl.closeSync(fd);
	}
}

/**
 * Write index lines atomically (tmp+rename, decision 015): a crash during
 * the write leaves the target byte-identical — only a successful rename
 * publishes the new content. The tmp file is pid-suffixed
 * (`<target>.tmp.<pid>`) so concurrent runs cannot collide, and the tmp name
 * is excluded from shard discovery.
 */
export function writeIndexLinesAtomic(
	indexPath: string,
	entries: string[],
	fsImpl: Pick<typeof fs, "writeFileSync" | "renameSync"> & Partial<Pick<typeof fs, "unlinkSync">> = fs,
): void {
	const tmp = `${indexPath}.tmp.${process.pid}`;
	try {
		fsImpl.writeFileSync(tmp, entries.length > 0 ? `${entries.join("\n")}\n` : "");
		fsImpl.renameSync(tmp, indexPath);
	} catch (err) {
		// The publish failed: the target is still the old content (the rename
		// never happened), and the tmp file is litter — remove it best-effort.
		try {
			fsImpl.unlinkSync?.(tmp);
		} catch {}
		throw err;
	}
}

/**
 * Write the full vector index atomically (tmp+rename, decision 015) from an
 * in-memory entry list. digest-all's batch path (issue #142) buffers all row
 * mutations in memory — appends and model-change replaces — and flushes once
 * after the batch, so a run performs exactly one full index write regardless
 * of the number of re-indexed rounds. The tmp file is pid-suffixed so
 * concurrent runs cannot collide on it.
 */
export function flushVectorIndex(
	indexPath: string,
	entries: VectorIndexEntry[],
	fsImpl: Pick<typeof fs, "writeFileSync" | "renameSync"> = fs,
): void {
	const lines = entries.map((entry) =>
		encodeVectorIndexLine(entry.vector, entry.filePath, entry.model, entry.embeddingInputHash),
	);
	const tmp = `${indexPath}.tmp.${process.pid}`;
	fsImpl.writeFileSync(tmp, lines.length > 0 ? `${lines.join("\n")}\n` : "");
	fsImpl.renameSync(tmp, indexPath);
}

export interface FlushOwnShardDeps extends AcquireIndexLockDeps {
	/** Hostname override for tests; defaults to os.hostname(). */
	hostname?: string;
	/**
	 * fs surface for the flush and the shard append lock it takes (PR #151
	 * R1.F1). Defaults to node:fs.
	 */
	fsImpl?: IndexStorageFs;
	/**
	 * The own shard's rows before the caller's long-running work began. When
	 * provided, the final locked read uses this snapshot to identify rows
	 * appended concurrently and preserves them alongside the desired rows.
	 */
	baselineEntries?: readonly VectorIndexEntry[];
}

/**
 * Flush the host's own shard (issue #149): the rows are written to
 * `index.<hostname>` inside `roundsDir` (tmp+rename, decision 015). A
 * wholesale rewrite is always scoped to one shard — the caller passes only the
 * rows of the host's own shard, and a foreign shard is never rewritten by this
 * host (D-005). Because the lockfile protocol is per path
 * (`<shard>.lock`), the short-lock merge model of decision 005 applies per
 * shard as well.
 *
 * The caller must have resolved the own shard's rows *after* any legacy
 * adoption (the adoption renames `index.csv` to `index.<hostname>`), so the
 * legacy rows are part of the flushed content.
 */
export function flushOwnShardVectorIndex(
	roundsDir: string,
	entries: VectorIndexEntry[],
	deps: FlushOwnShardDeps = {},
): string {
	const shardPath = ownShardPath(roundsDir, deps.hostname);
	// Same append lock the runtime appends take (`<shard>.lock`): a live
	// append landing between the caller's read and the tmp+rename would
	// otherwise be dropped. Fail fast when the lock is exhausted — replacing
	// the whole file unsynchronized could drop concurrent rows (PR #151 R1.F1).
	const lock = acquireIndexLock(shardPath, deps);
	if (lock === null) {
		throw new Error(
			`Could not acquire the index lock (${shardPath}.lock) after all retries — the index was NOT written.`,
		);
	}
	try {
		const desiredLines = entries.map((entry) =>
			encodeVectorIndexLine(entry.vector, entry.filePath, entry.model, entry.embeddingInputHash),
		);
		if (deps.baselineEntries) {
			// The caller's long-running work happened outside the lock. Re-read
			// under it and merge rows appended since the baseline snapshot, using
			// exact-string multiset difference so duplicate live rows are kept.
			const baselineLines = deps.baselineEntries.map((entry) =>
				encodeVectorIndexLine(entry.vector, entry.filePath, entry.model, entry.embeddingInputHash),
			);
			const freshLines = readIndexLines(shardPath, deps.fsImpl ?? fs);
			const baselineCounts = new Map<string, number>();
			for (const line of baselineLines) baselineCounts.set(line, (baselineCounts.get(line) ?? 0) + 1);
			const appendedLines = freshLines.filter((line) => {
				const remaining = baselineCounts.get(line) ?? 0;
				if (remaining === 0) return true;
				baselineCounts.set(line, remaining - 1);
				return false;
			});
			writeIndexLinesAtomic(shardPath, [...desiredLines, ...appendedLines], deps.fsImpl);
		} else {
			flushVectorIndex(shardPath, entries, deps.fsImpl);
		}
	} finally {
		lock.release();
	}
	return shardPath;
}

export function appendVectorIndexEntry(
	indexPath: string,
	vector: number[],
	filePath: string,
	model?: string,
	embeddingInputHash?: string,
): void {
	fs.appendFileSync(indexPath, `${encodeVectorIndexLine(vector, filePath, model, embeddingInputHash)}\n`);
}

/**
 * Options for the merged (shard-set) read APIs: iterating or loading rows
 * across every shard discovered in a rounds directory (issue #149).
 */
export interface ShardedReadOptions {
	/**
	 * Explicit shard set; defaults to {@link discoverIndexShards} over
	 * `indexPath` interpreted as the *rounds directory*. Missing dirs and
	 * empty shard sets yield no lines and no errors.
	 */
	shards?: string[];
	/** Warn sink for skipped corrupt lines; defaults to console.warn. */
	warn?: (message: string) => void;
	/**
	 * fs surface used to read the shards; defaults to node:fs. Injected so a
	 * caller with its own fs seam (the sweep script) reads the same files the
	 * rest of the run sees.
	 */
	fsImpl?: IndexReadFs;
	/**
	 * Deduplicate the merged rows by round + suffix + model (issue #149, OI-4).
	 * Defaults to true: a round embedded by two hosts with the same model
	 * exists as a row in both shards and must be counted once.
	 */
	dedupe?: boolean;
}

/**
 * The merge key of an index row: the round file, the row suffix
 * (`:prompt`/`:response`/…), and the embedding model.
 *
 * The model is part of the key on purpose: rows of *different* models are not
 * duplicates — the search groups the index by model (`getIndexEmbeddingModels`)
 * and the coverage predicate needs the stale-model rows to detect a round that
 * still needs a re-embed (issue #140 D1). Collapsing them would silently drop
 * a searchable model from the index and would mark stale rounds as covered.
 */
export function indexRowMergeKey(filePath: string, model?: string): string {
	const roundFile = path.basename(indexRoundFileFromPath(filePath));
	return `${roundFile}${indexRowSuffix(filePath)} ${model ?? ""}`;
}

/**
 * Drop the duplicate rows of a merged read (issue #149, OI-4): the first row
 * of a merge key wins, in the deterministic shard order the reader used. The
 * shards on disk are untouched — the dedup happens at read time.
 */
export function dedupeIndexEntries<T extends { filePath: string; model?: string }>(entries: readonly T[]): T[] {
	const seen = new Set<string>();
	const deduped: T[] = [];
	for (const entry of entries) {
		const key = indexRowMergeKey(entry.filePath, entry.model);
		if (seen.has(key)) continue;
		seen.add(key);
		deduped.push(entry);
	}
	return deduped;
}

/**
 * Invoke `onLine` for every non-blank line across the shard set of a rounds
 * directory (merged view, issue #149). Each corrupt line — bad base64 or a
 * partial write — is skipped with a warning naming the shard file it came
 * from; healthy lines of the same shard are unaffected.
 *
 * The corrupt-line guard applies only to vector parsing: this helper yields
 * raw lines, so parse errors are the caller's concern unless it delegates to
 * the parsing callbacks below.
 */
export function forEachShardLine(
	roundsDir: string,
	onLine: (line: string, shardPath: string) => void,
	shards?: string[],
	fsImpl: IndexReadFs = fs,
): void {
	for (const shardPath of shards ?? discoverIndexShards(roundsDir)) {
		forEachIndexLine(shardPath, fsImpl, (line) => {
			onLine(line, shardPath);
		});
	}
}

/**
 * Shard-set variant of {@link loadVectorIndex}: parse every row across all
 * shards, skipping corrupt lines (unparseable base64/JSON, partial writes)
 * with a warning naming the shard file.
 */
export function loadShardVectorIndex(roundsDir: string, opts: ShardedReadOptions = {}): VectorIndexEntry[] {
	const warn = opts.warn ?? console.warn;
	const entries: VectorIndexEntry[] = [];
	forEachShardLine(
		roundsDir,
		(line, shardPath) => {
			try {
				entries.push(parseVectorIndexLine(line));
			} catch (reason) {
				warn(
					`Skipping corrupt index row in ${shardPath}: ${reason instanceof Error ? reason.message : String(reason)}`,
				);
			}
		},
		opts.shards,
		opts.fsImpl ?? fs,
	);
	return opts.dedupe === false ? entries : dedupeIndexEntries(entries);
}

export function loadVectorIndex(indexPath: string, fsImpl: IndexReadFs = fs): VectorIndexEntry[] {
	const entries: VectorIndexEntry[] = [];
	forEachIndexLine(indexPath, fsImpl, (line) => {
		entries.push(parseVectorIndexLine(line));
	});
	return entries;
}

export function indexRoundFileFromPath(filePath: string): string {
	return filePath.replace(/(:prompt|:response|:round|:summary)$/, "");
}

/**
 * The index label suffix of a round's row (`:prompt`, `:response`, `:round`,
 * `:summary`), or `""` for a bare round-file row. Shared by the `just index`
 * sweep and the startup coverage predicate so both classify a row by the same
 * suffix (issue #140 D1 / F2, PR !141 review).
 */
export function indexRowSuffix(filePath: string): string {
	const roundFile = path.basename(indexRoundFileFromPath(filePath));
	return path.basename(filePath).slice(roundFile.length);
}

/** Shard-set variant of {@link loadIndexedRoundFiles} (merged view). */
export function loadIndexedRoundFilesSharded(roundsDir: string, opts: ShardedReadOptions = {}): Set<string> {
	return new Set(
		loadShardVectorIndex(roundsDir, opts).map((entry) => path.basename(indexRoundFileFromPath(entry.filePath))),
	);
}

export function loadIndexedRoundFiles(indexPath: string, fsImpl: IndexReadFs = fs): Set<string> {
	return new Set(
		loadVectorIndex(indexPath, fsImpl).map((entry) => path.basename(indexRoundFileFromPath(entry.filePath))),
	);
}

/** Shard-set variant of {@link loadRoundFilesWithDifferentModel} (merged view). */
export function loadRoundFilesWithDifferentModelSharded(
	roundsDir: string,
	currentModel: string,
	opts: ShardedReadOptions = {},
): Set<string> {
	const mismatched = loadShardVectorIndex(roundsDir, opts)
		.filter((entry) => entry.model !== undefined && entry.model !== currentModel)
		.map((entry) => path.basename(indexRoundFileFromPath(entry.filePath)));
	return new Set(mismatched);
}

export function loadRoundFilesWithDifferentModel(
	indexPath: string,
	currentModel: string,
	fsImpl: IndexReadFs = fs,
): Set<string> {
	const mismatched = loadVectorIndex(indexPath, fsImpl)
		.filter((entry) => entry.model !== undefined && entry.model !== currentModel)
		.map((entry) => path.basename(indexRoundFileFromPath(entry.filePath)));
	return new Set(mismatched);
}

export function migrateIndexEntryLine(line: string, oldRoundFile: string, newRoundFile: string): string {
	const firstComma = line.indexOf(",");
	if (firstComma === -1) return line;
	// First metadata segment is "filePath[:suffix]"; trailing model/embeddingInputHash
	// columns (comma-free) are preserved verbatim.
	const segments = line.slice(firstComma + 1).split(",");
	if (!segments[0].startsWith(oldRoundFile)) return line;
	segments[0] = newRoundFile + segments[0].slice(oldRoundFile.length);
	return `${line.slice(0, firstComma + 1)}${segments.join(",")}`;
}

export function indexEntryFilename(line: string): string | null {
	const commaIdx = line.indexOf(",");
	if (commaIdx === -1) return null;
	// First metadata segment is "filePath[:suffix]"; trailing model/hash columns are ignored.
	const filePath = line.slice(commaIdx + 1).split(",")[0];
	const colonIdx = filePath.lastIndexOf(":");
	if (colonIdx === -1) return null;
	return filePath.slice(0, colonIdx);
}

export function readIndexByFilename(indexPath: string): Map<string, string[]> {
	return indexLinesByFilename(readIndexLines(indexPath));
}

/**
 * Group index lines by the round file they reference. Works on an in-memory
 * line set so a caller can group the rows of one shard without re-reading a
 * file (issue #149: a rewrite operates on the host's own shard's rows).
 */
export function indexLinesByFilename(lines: string[]): Map<string, string[]> {
	const index = new Map<string, string[]>();
	for (const line of lines) {
		const filename = indexEntryFilename(line);
		if (!filename) continue;
		const bucket = index.get(filename) ?? [];
		bucket.push(line);
		index.set(filename, bucket);
	}
	return index;
}

export function replaceIndexLineFilename(line: string, newFilename: string): string {
	const commaIdx = line.indexOf(",");
	if (commaIdx === -1) return line;
	// Replace the filename in the first metadata segment ("filePath[:suffix]");
	// trailing model/embeddingInputHash columns are preserved verbatim.
	const segments = line.slice(commaIdx + 1).split(",");
	segments[0] = segments[0].replace(/^[^:]+/, newFilename);
	return line.slice(0, commaIdx + 1) + segments.join(",");
}

export function filterIndexLinesExcludingFilenames(lines: string[], filenames: Set<string>): string[] {
	return lines.filter((line) => {
		const filename = indexEntryFilename(line);
		return !filename || !filenames.has(filename);
	});
}

export function findStaleContentMatches(roundsDir: string, roundFile: string): string[] {
	const files = fs.readdirSync(roundsDir).filter((f) => f.endsWith(".json") && !f.startsWith("index"));
	const matches: string[] = [];
	for (const file of files) {
		if (file === roundFile) continue;
		try {
			const data = JSON.parse(fs.readFileSync(path.join(roundsDir, file), "utf-8")) as RoundHashContent;
			const hash = `${computeContentHash(data.userPrompt ?? "", data.responseSequence ?? "", data.toolCalls)}.json`;
			if (hash === roundFile) matches.push(file);
		} catch {
			// Corrupt round files are ignored during stale-content discovery.
		}
	}
	return matches;
}

/**
 * Materialize the findStaleContentMatches result for every round file in one
 * pass as a map keyed by the *target* (content-hash) filename. Callers that
 * process many rounds in a loop — digest-all's sweep — would otherwise scan and
 * JSON-parse the whole store once per round (issue #139: O(n²)). The per-call
 * `findStaleContentMatches` is unchanged for direct callers.
 *
 * A file whose content hashes to its own name contributes no stale entry; a
 * legacy file whose stored name differs from its content hash is recorded under
 * the hash filename its content belongs to. Corrupt/unreadable files are skipped
 * exactly as in the per-call scan.
 */
export function buildStaleContentMatchMap(
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readdirSync" | "readFileSync"> = fs,
): Map<string, string[]> {
	const matches = new Map<string, string[]>();
	if (!fsImpl.existsSync(roundsDir)) return matches;
	const files = fsImpl.readdirSync(roundsDir).filter((f) => f.endsWith(".json") && !f.startsWith("index"));
	for (const file of files) {
		try {
			const data = JSON.parse(fsImpl.readFileSync(path.join(roundsDir, file), "utf-8")) as RoundHashContent;
			const hash = `${computeContentHash(data.userPrompt ?? "", data.responseSequence ?? "", data.toolCalls)}.json`;
			if (hash === file) continue;
			const list = matches.get(hash);
			if (list) list.push(file);
			else matches.set(hash, [file]);
		} catch {
			// Corrupt round files are ignored during stale-content discovery.
		}
	}
	return matches;
}
