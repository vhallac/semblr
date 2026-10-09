import * as fs from "node:fs";
import * as path from "node:path";
import { computeContentHash, type HashToolCallDetail } from "./hash.ts";

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

export function writeIndexLines(indexPath: string, entries: string[]): void {
	fs.writeFileSync(indexPath, entries.join("\n") + (entries.length > 0 ? "\n" : ""));
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

export function loadIndexedRoundFiles(indexPath: string, fsImpl: IndexReadFs = fs): Set<string> {
	return new Set(
		loadVectorIndex(indexPath, fsImpl).map((entry) => path.basename(indexRoundFileFromPath(entry.filePath))),
	);
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

export function replaceIndexEntriesForRoundFile(
	indexPath: string,
	roundFile: string,
	entries: VectorIndexEntry[],
): void {
	const remaining = readIndexLines(indexPath).filter((line) => {
		const filename = indexEntryFilename(line);
		return !filename || path.basename(filename) !== roundFile;
	});
	const replacement = entries.map((entry) =>
		encodeVectorIndexLine(entry.vector, entry.filePath, entry.model, entry.embeddingInputHash),
	);
	writeIndexLines(indexPath, [...remaining, ...replacement]);
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

export function migrateIndexEntries(indexPath: string, oldRoundFile: string, newRoundFile: string): void {
	if (!fs.existsSync(indexPath)) return;
	const migrated = readIndexLines(indexPath).map((line) => migrateIndexEntryLine(line, oldRoundFile, newRoundFile));
	writeIndexLines(indexPath, migrated);
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
	const index = new Map<string, string[]>();
	for (const line of readIndexLines(indexPath)) {
		const filename = indexEntryFilename(line);
		if (!filename) continue;
		if (!index.has(filename)) index.set(filename, []);
		index.get(filename)?.push(line);
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
