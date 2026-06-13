import { countWordsInMessageContent, shouldDropEmbedding } from "./context-messages.ts";
import { filterCandidatesAsOf } from "./eval-metrics.ts";
import { indexRoundFileFromPath } from "./index-io.ts";
import type { IndexEntry } from "./index-storage.ts";
import type { RoundData } from "./round-data.ts";
import { collectSearchRoundScores } from "./search-interactions.ts";

export type GoldenDifficulty = "miss" | "hard" | "control";

export interface WeakBaselineEntry {
	query: string;
	labels: string[];
	top5: Array<{ file: string; score: number }>;
	first_hit_rank: number | null;
}

export interface GoldenQueryCandidate {
	query: string;
	prompt: string;
	difficulty: GoldenDifficulty;
	semblrActive: boolean;
}

export interface PoolQuery {
	query: string;
	prompt: string;
	promptEmbedding: number[];
	userTimestamp: number;
	parentId?: string | null;
	expandedRounds: string[];
}

export interface PoolOptions {
	index: readonly IndexEntry[];
	rounds: ReadonlyMap<string, RoundData>;
	seed: number;
	cosineTopN?: number;
	keywordTopN?: number;
	cap?: number;
}

function mulberry32(seed: number): () => number {
	let t = seed >>> 0;
	return () => {
		t += 0x6d2b79f5;
		let r = Math.imul(t ^ (t >>> 15), t | 1);
		r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
		return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
	};
}

function shuffleSeeded<T>(values: readonly T[], seed: number): T[] {
	const rng = mulberry32(seed);
	const out = [...values];
	for (let index = out.length - 1; index > 0; index--) {
		const swapIndex = Math.floor(rng() * (index + 1));
		[out[index], out[swapIndex]] = [out[swapIndex], out[index]];
	}
	return out;
}

export function isFilePathQuery(prompt: string): boolean {
	return /(?:^|\s)(?:[./~]?[-\w]+\/[-\w./]+|[-\w]+\.[A-Za-z0-9]{1,10})(?:\s|$)/.test(prompt);
}

export function classifyDifficulty(
	entry: Pick<WeakBaselineEntry, "first_hit_rank"> & { prompt?: string },
): GoldenDifficulty {
	if (entry.prompt && isFilePathQuery(entry.prompt)) return "hard";
	if (entry.first_hit_rank === 1 || entry.first_hit_rank === 2) return "control";
	if (entry.first_hit_rank === null) return "miss";
	return "hard";
}

export function selectGoldenQueries(
	candidates: readonly GoldenQueryCandidate[],
	opts: { seed: number; count: number },
): GoldenQueryCandidate[] {
	const eligible = shuffleSeeded(
		candidates.filter((candidate) => candidate.semblrActive),
		opts.seed,
	).sort((a, b) => {
		const difficultyOrder = { miss: 0, hard: 1, control: 2 } satisfies Record<GoldenDifficulty, number>;
		const diff = difficultyOrder[a.difficulty] - difficultyOrder[b.difficulty];
		return diff !== 0 ? diff : a.query.localeCompare(b.query);
	});
	const count = Math.min(opts.count, eligible.length);
	const hardTarget = Math.round(count * 0.7);
	const hardPool = eligible.filter((candidate) => candidate.difficulty !== "control");
	const controlPool = eligible.filter((candidate) => candidate.difficulty === "control");
	const chosen = [
		...hardPool.slice(0, Math.min(hardTarget, hardPool.length)),
		...controlPool.slice(0, Math.min(count - Math.min(hardTarget, hardPool.length), controlPool.length)),
	];
	const remainder = eligible.filter((candidate) => !chosen.includes(candidate)).slice(0, count - chosen.length);
	return shuffleSeeded([...chosen, ...remainder], opts.seed + 1);
}

function splitToken(raw: string): string[] {
	return raw
		.replace(/([a-z0-9])([A-Z])/g, "$1 $2")
		.split(/[^a-z0-9]+/i)
		.map((token) => token.toLowerCase())
		.filter(Boolean);
}

function tokenize(value: string): string[] {
	return splitToken(value.replace(/_/g, " "));
}

function lexicalScore(queryTerms: ReadonlySet<string>, text: string): number {
	const candidateTerms = new Set(tokenize(text));
	let score = 0;
	for (const term of queryTerms) {
		if (candidateTerms.has(term)) score++;
	}
	return score;
}

function uniqueInOrder(values: readonly string[]): string[] {
	const seen = new Set<string>();
	return values.filter((value) => {
		if (seen.has(value)) return false;
		seen.add(value);
		return true;
	});
}

function balancedCap(channels: readonly string[][], cap: number): string[] {
	const picked: string[] = [];
	const seen = new Set<string>();
	let added = true;
	for (let offset = 0; picked.length < cap && added; offset++) {
		added = false;
		for (const channel of channels) {
			const candidate = channel[offset];
			if (!candidate || seen.has(candidate)) continue;
			seen.add(candidate);
			picked.push(candidate);
			added = true;
			if (picked.length >= cap) break;
		}
	}
	return picked;
}

export function poolCandidates(query: PoolQuery, opts: PoolOptions): string[] {
	const cosineTopN = opts.cosineTopN ?? 12;
	const keywordTopN = opts.keywordTopN ?? 12;
	const cap = opts.cap ?? 20;
	const filtered = filterCandidatesAsOf(opts.index, opts.rounds, query.query, query.userTimestamp);
	const readRound = (filePath: string) => opts.rounds.get(indexRoundFileFromPath(filePath)) ?? null;
	const cosine = collectSearchRoundScores(filtered.entries, query.promptEmbedding, readRound)
		.slice(0, cosineTopN)
		.map((entry) => entry.fileName)
		.filter((file) => file !== query.query);

	const eligibleRounds = uniqueInOrder(filtered.entries.map((entry) => indexRoundFileFromPath(entry.filePath))).filter(
		(file) => file !== query.query,
	);
	const queryTerms = new Set(tokenize(query.prompt));
	const keyword = eligibleRounds
		.map((file) => {
			const round = opts.rounds.get(file);
			return {
				file,
				score: round ? lexicalScore(queryTerms, `${round.userPrompt}\n${round.responseSequence}`) : 0,
			};
		})
		.filter((entry) => entry.score > 0)
		.sort((a, b) => b.score - a.score || a.file.localeCompare(b.file))
		.slice(0, keywordTopN)
		.map((entry) => entry.file);

	const weak = uniqueInOrder([...(query.parentId ? [query.parentId] : []), ...query.expandedRounds]).filter(
		(file) => file !== query.query,
	);

	const capped = balancedCap([cosine, keyword, weak], cap);
	return shuffleSeeded(capped, opts.seed);
}

export function collectReplayableGoldenUniverse(
	rounds: ReadonlyMap<string, RoundData>,
	queryInfo: ReadonlyMap<string, { expandedRounds: string[]; semblrActive: boolean }>,
	weakEntries: readonly WeakBaselineEntry[],
): Array<GoldenQueryCandidate & PoolQuery> {
	const weakByQuery = new Map(weakEntries.map((entry) => [entry.query, entry]));
	const candidates: Array<GoldenQueryCandidate & PoolQuery> = [];
	for (const [query, round] of rounds) {
		if (shouldDropEmbedding(countWordsInMessageContent(round.userPrompt ?? ""))) continue;
		if (!round.promptEmbedding || typeof round.userTimestamp !== "number") continue;
		const info = queryInfo.get(query);
		if (!info?.semblrActive) continue;
		const weak = weakByQuery.get(query);
		const prompt = round.userPrompt ?? "";
		candidates.push({
			query,
			prompt,
			difficulty: classifyDifficulty({ first_hit_rank: weak?.first_hit_rank ?? null, prompt }),
			semblrActive: info.semblrActive,
			promptEmbedding: round.promptEmbedding,
			userTimestamp: round.userTimestamp,
			parentId: round.parentId,
			expandedRounds: info.expandedRounds,
		});
	}
	return candidates;
}
