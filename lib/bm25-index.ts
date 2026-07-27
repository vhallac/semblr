import * as fs from "node:fs";
import * as path from "node:path";
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

export function bm25IndexPathForRoundsDir(roundsDir: string): string {
	return path.join(roundsDir, "index.bm25.json");
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
		termFrequencies[token] = (termFrequencies[token] ?? 0) + 1;
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

	const index = buildBm25Index(rounds);
	writeBm25Index(indexPath, index, fsImpl);
	return index;
}

export function writeBm25Index(
	indexPath: string,
	index: Bm25Index,
	fsImpl: Pick<typeof fs, "mkdirSync" | "writeFileSync"> = fs,
): void {
	fsImpl.mkdirSync(path.dirname(indexPath), { recursive: true });
	fsImpl.writeFileSync(indexPath, `${JSON.stringify(refreshStats(index), null, 2)}\n`);
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
