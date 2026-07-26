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
		if (!parsed.documents || typeof parsed.documents !== "object") return buildBm25Index([]);
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
	const maxScore = Math.max(0, ...scores.values());
	if (maxScore <= 0) return new Map();
	return new Map(Array.from(scores.entries()).map(([fileName, score]) => [fileName, score / maxScore]));
}
