import * as fs from "node:fs";
import * as path from "node:path";
import { atomicWriteFileSync } from "./atomic-write.ts";
import type { RoundData } from "./round-data.ts";

const BM25_VERSION = 1;
const BM25_K1 = 1.2;
const BM25_B = 0.75;

export interface Bm25Document {
	length: number;
	termFrequencies: Record<string, number>;
}

export interface Bm25Index {
	version: number;
	documentCount: number;
	averageDocumentLength: number;
	documents: Record<string, Bm25Document>;
}

/** Legacy BM25 artifact name, adopted by rename on first write (decision 024). */
export const LEGACY_BM25_FILENAME = "index.bm25.json";

/**
 * Legacy artifact path — superseded by `ownBm25IndexPath` (issue #152); kept
 * until every caller routes to the host's own artifact (units 005/006).
 */
export function bm25IndexPathForRoundsDir(roundsDir: string): string {
	return path.join(roundsDir, LEGACY_BM25_FILENAME);
}

/**
 * Path of the host's own BM25 artifact (`index.bm25.<hostId>.json`, issue #152).
 * The host id is resolved by the caller (`resolveHostId`, lib/host-id.ts) and
 * is only ever used as a filename component — the artifact format is unchanged
 * (assumption A-2).
 */
export function ownBm25IndexPath(roundsDir: string, hostId: string): string {
	return path.join(roundsDir, `index.bm25.${hostId}.json`);
}

function stemToken(token: string): string {
	if (token.length > 5 && token.endsWith("ing")) return token.slice(0, -3);
	if (token.length > 4 && token.endsWith("ed")) return token.slice(0, -2);
	if (token.length > 4 && token.endsWith("ies")) return `${token.slice(0, -3)}y`;
	if (token.length > 4 && token.endsWith("es")) return token.slice(0, -2);
	if (token.length > 3 && token.endsWith("s")) return token.slice(0, -1);
	return token;
}

export function tokenizeBm25(text: string): string[] {
	const rawTokens = text.toLowerCase().match(/[a-z0-9_./:-]+/g) ?? [];
	const tokens: string[] = [];
	for (const raw of rawTokens) {
		tokens.push(stemToken(raw));
		for (const part of raw.split(/[./:-]+/)) {
			if (part && part !== raw) tokens.push(stemToken(part));
		}
	}
	return tokens;
}

export function roundTextForBm25(
	round: Pick<RoundData, "userPrompt" | "responseSequence" | "toolCallNames" | "toolCalls">,
): string {
	const toolText = [
		...(round.toolCallNames ?? []),
		...(round.toolCalls ?? []).flatMap((toolCall) => [
			toolCall.name,
			toolCall.arguments,
			toolCall.result_summary,
			toolCall.result_full ?? "",
		]),
	].join("\n");
	return [round.userPrompt, round.responseSequence, toolText].filter(Boolean).join("\n\n");
}

function documentFromText(text: string): Bm25Document {
	const termFrequencies: Record<string, number> = {};
	const tokens = tokenizeBm25(text);
	for (const token of tokens) {
		// Store hygiene (PR !131): `termFrequencies[token] ?? 0` reads through
		// the prototype chain — for the token "constructor" it resolves to the
		// inherited Object.prototype.constructor function, and `function + 1`
		// stores the string "function Object() { [native code] }1". One corrupt
		// doc makes isBm25Documents (the .every across all documents) reject the
		// ENTIRE persisted index, forcing a full rebuild on every load. Use
		// own-property semantics instead.
		termFrequencies[token] = (Object.hasOwn(termFrequencies, token) ? termFrequencies[token] : 0) + 1;
	}
	return { length: tokens.length, termFrequencies };
}

function refreshStats(index: Bm25Index): Bm25Index {
	const docs = Object.values(index.documents);
	const documentCount = docs.length;
	const totalLength = docs.reduce((sum, doc) => sum + doc.length, 0);
	index.version = BM25_VERSION;
	index.documentCount = documentCount;
	index.averageDocumentLength = documentCount > 0 ? totalLength / documentCount : 0;
	return index;
}

function isBm25Document(value: unknown): value is Bm25Document {
	if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
	const document = value as Partial<Bm25Document>;
	if (typeof document.length !== "number" || !Number.isFinite(document.length) || document.length < 0) return false;
	if (
		document.termFrequencies === null ||
		typeof document.termFrequencies !== "object" ||
		Array.isArray(document.termFrequencies)
	) {
		return false;
	}
	return Object.values(document.termFrequencies).every(
		(frequency) => typeof frequency === "number" && Number.isFinite(frequency) && frequency >= 0,
	);
}

function isBm25Documents(value: unknown): value is Record<string, Bm25Document> {
	return (
		value !== null && typeof value === "object" && !Array.isArray(value) && Object.values(value).every(isBm25Document)
	);
}

export function buildBm25Index(rounds: Array<{ fileName: string; text: string }>): Bm25Index {
	const index: Bm25Index = {
		version: BM25_VERSION,
		documentCount: 0,
		averageDocumentLength: 0,
		documents: {},
	};
	for (const round of rounds) {
		index.documents[round.fileName] = documentFromText(round.text);
	}
	return refreshStats(index);
}

export function upsertBm25Round(index: Bm25Index, fileName: string, text: string): Bm25Index {
	index.documents[fileName] = documentFromText(text);
	return refreshStats(index);
}

export function deleteBm25Round(index: Bm25Index, fileName: string): Bm25Index {
	delete index.documents[fileName];
	return refreshStats(index);
}

export function loadBm25Index(
	indexPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): Bm25Index {
	if (!fsImpl.existsSync(indexPath)) return buildBm25Index([]);
	try {
		const parsed = JSON.parse(fsImpl.readFileSync(indexPath, "utf-8")) as Partial<Bm25Index>;
		if (!isBm25Documents(parsed.documents)) return buildBm25Index([]);
		return refreshStats({
			version: parsed.version ?? BM25_VERSION,
			documentCount: parsed.documentCount ?? 0,
			averageDocumentLength: parsed.averageDocumentLength ?? 0,
			documents: parsed.documents,
		});
	} catch {
		return buildBm25Index([]);
	}
}

export function loadOrRebuildBm25Index(
	indexPath: string,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "mkdirSync" | "readFileSync" | "readdirSync" | "writeFileSync"> = fs,
): Bm25Index {
	const roundFileNames = fsImpl.existsSync(roundsDir)
		? fsImpl.readdirSync(roundsDir).filter((file) => file.endsWith(".json") && !file.startsWith("index"))
		: [];
	if (fsImpl.existsSync(indexPath)) {
		try {
			const parsed = JSON.parse(fsImpl.readFileSync(indexPath, "utf-8")) as Partial<Bm25Index>;
			const documents = parsed.documents;
			if (
				isBm25Documents(documents) &&
				roundFileNames.length === Object.keys(documents).length &&
				roundFileNames.every((fileName) => fileName in documents)
			) {
				return loadBm25Index(indexPath, fsImpl);
			}
		} catch {
			// Rebuild invalid sidecars from the source round files below.
		}
	}

	const rounds: Array<{ fileName: string; text: string }> = [];
	if (roundFileNames.length > 0) {
		for (const fileName of roundFileNames) {
			try {
				const round = JSON.parse(fsImpl.readFileSync(path.join(roundsDir, fileName), "utf-8")) as RoundData;
				rounds.push({ fileName, text: roundTextForBm25(round) });
			} catch {
				// Ignore unreadable round files, matching the standalone rebuild command.
			}
		}
	}

	// Deliberately NOT persisted here (issue #137 / PR !138 F1): the write is
	// left to the caller's explicit flush so a recovery batch commits the
	// index file exactly once, even when the load itself rebuilt it. Callers
	// that never flush keep the rebuilt index in memory only — queries are
	// still consistent because the in-memory index matches the round files.
	return buildBm25Index(rounds);
}

/** fs surface needed to flush a BM25 artifact: adoption checks plus the atomic tmp+rename write. */
export type Bm25WriteFs = Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "renameSync">;

/**
 * Adopt a legacy `index.bm25.json` into the host's own artifact path by
 * rename (decision 024): on the first write to a path other than the legacy
 * one, when the legacy artifact exists and the target does not, the legacy
 * file is renamed to the target — an atomic same-directory rename (decision
 * 015). The flushed index is a superset of the legacy content (the caller
 * loaded the legacy artifact before upserting its rounds), so the rename is
 * immediately followed by the publishing write. A host that already owns its
 * artifact leaves the legacy file untouched (POSIX rename overwrites, so
 * renaming would clobber the host's own artifact). A lost adoption race (the
 * legacy file was claimed by another host between the check and the rename)
 * is swallowed: the host proceeds with its own artifact.
 */
function adoptLegacyBm25Index(indexPath: string, fsImpl: Bm25WriteFs): boolean {
	if (path.basename(indexPath) === LEGACY_BM25_FILENAME) return false;
	const legacyPath = path.join(path.dirname(indexPath), LEGACY_BM25_FILENAME);
	if (!fsImpl.existsSync(legacyPath) || fsImpl.existsSync(indexPath)) return false;
	try {
		fsImpl.renameSync(legacyPath, indexPath);
		return true;
	} catch {
		// Lost the adoption race: the legacy file is gone (claimed by another
		// host) — proceed with the host's own artifact.
		return false;
	}
}

/**
 * Flush a BM25 index atomically (decision 015): the content is staged to a
 * sibling tmp file (`<target>.tmp.<pid>`) and renamed over the target, so
 * concurrent readers never see a partial artifact and a crash mid-write
 * leaves the previous artifact byte-identical. In-place `writeFileSync` of
 * the artifact is gone (issue #152; decision 015 already named BM25 writes).
 */
export function writeBm25Index(indexPath: string, index: Bm25Index, fsImpl: Bm25WriteFs = fs): void {
	fsImpl.mkdirSync(path.dirname(indexPath), { recursive: true });
	adoptLegacyBm25Index(indexPath, fsImpl);
	atomicWriteFileSync(indexPath, `${JSON.stringify(refreshStats(index), null, 2)}\n`, fsImpl);
}

export function scoreBm25Query(index: Bm25Index, query: string): Map<string, number> {
	const queryTerms = new Set(tokenizeBm25(query));
	const scores = new Map<string, number>();
	if (queryTerms.size === 0 || index.documentCount === 0 || index.averageDocumentLength === 0) return scores;

	const documentFrequency = new Map<string, number>();
	for (const term of queryTerms) {
		let count = 0;
		for (const doc of Object.values(index.documents)) {
			if ((doc.termFrequencies[term] ?? 0) > 0) count++;
		}
		documentFrequency.set(term, count);
	}

	for (const [fileName, doc] of Object.entries(index.documents)) {
		let score = 0;
		for (const term of queryTerms) {
			const tf = doc.termFrequencies[term] ?? 0;
			if (tf === 0) continue;
			const df = documentFrequency.get(term) ?? 0;
			const idf = Math.log(1 + (index.documentCount - df + 0.5) / (df + 0.5));
			const denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc.length / index.averageDocumentLength));
			score += idf * ((tf * (BM25_K1 + 1)) / denominator);
		}
		if (score > 0) scores.set(fileName, score);
	}

	return scores;
}

export function normalizeBm25Scores(scores: ReadonlyMap<string, number>): Map<string, number> {
	const topScores = Array.from(scores.values())
		.filter((score) => score > 0)
		.sort((a, b) => b - a)
		.slice(0, 10);
	if (topScores.length === 0) return new Map();
	const middle = Math.floor(topScores.length / 2);
	const scale = topScores.length % 2 === 0 ? (topScores[middle - 1] + topScores[middle]) / 2 : topScores[middle] || 1;
	return new Map(
		Array.from(scores.entries())
			.filter(([, score]) => score > 0)
			.map(([fileName, score]) => [fileName, 1 / (1 + Math.exp(-score / scale))]),
	);
}
