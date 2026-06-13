import crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { countWordsInMessageContent, shouldDropEmbedding } from "../lib/context-messages.ts";
import { gatherSessionFiles, readCorpusRounds } from "../lib/eval-corpus.ts";
import { collectWeakLabelExpansionsFromSessionJsonl } from "../lib/eval-labels.ts";
import { aggregateQueryMetrics, evaluateQueryMetrics, filterCandidatesAsOf, noiseRate } from "../lib/eval-metrics.ts";
import { indexRoundFileFromPath } from "../lib/index-io.ts";
import { type IndexEntry, loadIndexFromPath } from "../lib/index-storage.ts";
import type { RoundData } from "../lib/round-data.ts";
import type { ScriptConfigOptions } from "../lib/script-config.ts";
import { collectSearchRoundScores } from "../lib/search-interactions.ts";
import { loadStatsFile } from "../lib/stats.ts";
import type { GoldenLabelsFile } from "./ingest-golden-labels.ts";

function defaultSessionsDir(corpusDir: string): string {
	return path.resolve(corpusDir, "sessions");
}

function parseArgValue(args: readonly string[], name: string): string | null {
	const index = args.indexOf(name);
	return index >= 0 ? (args[index + 1] ?? null) : null;
}

function sha256Json(value: unknown): string {
	return crypto.createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function collectWeakLabelsByQuery(sessionsDir: string, fsImpl: typeof fs = fs): Map<string, string[]> {
	const combined = new Map<string, string[]>();
	for (const sessionFile of gatherSessionFiles(sessionsDir, fsImpl)) {
		const labels = collectWeakLabelExpansionsFromSessionJsonl(fsImpl.readFileSync(sessionFile, "utf-8"));
		for (const [query, expansions] of labels) {
			combined.set(query, [...(combined.get(query) ?? []), ...expansions]);
		}
	}
	return combined;
}

function dedupe(values: readonly string[]): string[] {
	return [...new Set(values)];
}

export interface EvalRetrievalCounts {
	total_rounds: number;
	replayed: number;
	skipped_short: number;
	skipped_no_embedding: number;
	skipped_no_timestamp: number;
	skipped_no_labels: number;
}

export interface EvalRetrievalPerQuery {
	query: string;
	labels: string[];
	top5: Array<{ file: string; score: number }>;
	first_hit_rank: number | null;
}

export interface EvalRetrievalResult {
	kind: "weak" | "golden";
	config_hash: string;
	corpus: string;
	counts: EvalRetrievalCounts;
	hit_at_5: number;
	recall_at_5: number;
	mrr: number;
	noise_rate: number | null;
	per_query: EvalRetrievalPerQuery[];
}

export interface RunEvalRetrievalOptions extends ScriptConfigOptions {
	args?: string[];
	corpusDir?: string;
	sessionsDir?: string;
	outFile?: string | null;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: typeof fs;
	homedir?: () => string;
	gitRev?: string;
}

function loadGoldenLabels(filePath: string, fsImpl: typeof fs = fs): GoldenLabelsFile {
	return JSON.parse(fsImpl.readFileSync(filePath, "utf-8")) as GoldenLabelsFile;
}

export function evaluateRetrieval(options: {
	corpusDir: string;
	sessionsDir: string;
	index: readonly IndexEntry[];
	rounds: ReadonlyMap<string, RoundData>;
	statsPath: string;
	configHash: string;
	goldenLabels?: GoldenLabelsFile | null;
	fsImpl?: typeof fs;
}): EvalRetrievalResult {
	const fsImpl = options.fsImpl ?? fs;
	const weakExpansions = options.goldenLabels ? null : collectWeakLabelsByQuery(options.sessionsDir, fsImpl);
	const counts: EvalRetrievalCounts = {
		total_rounds: options.rounds.size,
		replayed: 0,
		skipped_short: 0,
		skipped_no_embedding: 0,
		skipped_no_timestamp: 0,
		skipped_no_labels: 0,
	};
	const metricQueries: Array<{ labels: string[]; ranked: string[] }> = [];
	const perQuery: EvalRetrievalPerQuery[] = [];
	const roundEntries = [...options.rounds.entries()].sort(([a], [b]) => a.localeCompare(b));
	const goldenByQuery = new Map((options.goldenLabels?.queries ?? []).map((entry) => [entry.query, entry]));
	const activeRoundEntries = options.goldenLabels
		? roundEntries.filter(([roundFile]) => goldenByQuery.has(roundFile))
		: roundEntries;

	const readRound = (filePath: string): RoundData | null => {
		const roundFile = path.basename(indexRoundFileFromPath(filePath));
		return options.rounds.get(roundFile) ?? null;
	};

	for (const [roundFile, round] of activeRoundEntries) {
		if (shouldDropEmbedding(countWordsInMessageContent(round.userPrompt ?? ""))) {
			counts.skipped_short++;
			continue;
		}
		if (!round.promptEmbedding) {
			counts.skipped_no_embedding++;
			continue;
		}
		if (typeof round.userTimestamp !== "number") {
			counts.skipped_no_timestamp++;
			continue;
		}

		counts.replayed++;
		const filtered = filterCandidatesAsOf(options.index, options.rounds, roundFile, round.userTimestamp);
		const rankedScores = collectSearchRoundScores(filtered.entries, round.promptEmbedding, readRound).slice(0, 5);
		const ranked = rankedScores.map((entry) => entry.fileName);
		const goldenEntry = goldenByQuery.get(roundFile);
		const labels = dedupe(
			goldenEntry
				? goldenEntry.labels
				: [
						...(typeof round.parentId === "string" && round.parentId ? [round.parentId] : []),
						...((weakExpansions?.get(roundFile) ?? []) as string[]),
					],
		).filter((label) => label !== roundFile);
		if (labels.length === 0) {
			counts.skipped_no_labels++;
			continue;
		}

		const metrics = evaluateQueryMetrics({
			labels: goldenEntry?.primary ? [goldenEntry.primary] : labels,
			ranked,
		});
		metricQueries.push({ labels, ranked });
		perQuery.push({
			query: roundFile,
			labels,
			top5: rankedScores.map((entry) => ({ file: entry.fileName, score: entry.bestScore })),
			first_hit_rank: metrics.firstHitRank,
		});
	}

	const aggregates = options.goldenLabels
		? (() => {
				const queryCount = perQuery.length;
				if (queryCount === 0) return { hitAt5: 0, recallAt5: 0, mrr: 0 };
				const totals = perQuery.reduce(
					(acc, query) => {
						const goldenEntry = goldenByQuery.get(query.query);
						const ranked = query.top5.map((entry) => entry.file);
						const recallMetrics = evaluateQueryMetrics({ labels: query.labels, ranked });
						const mrrMetrics = evaluateQueryMetrics({
							labels: goldenEntry?.primary ? [goldenEntry.primary] : query.labels.slice(0, 1),
							ranked,
						});
						acc.hitAt5 += recallMetrics.hitAt5;
						acc.recallAt5 += recallMetrics.recallAt5;
						acc.mrr += mrrMetrics.mrr;
						return acc;
					},
					{ hitAt5: 0, recallAt5: 0, mrr: 0 },
				);
				return {
					hitAt5: totals.hitAt5 / queryCount,
					recallAt5: totals.recallAt5 / queryCount,
					mrr: totals.mrr / queryCount,
				};
			})()
		: aggregateQueryMetrics(metricQueries);
	return {
		kind: options.goldenLabels ? "golden" : "weak",
		config_hash: options.configHash,
		corpus: path.basename(options.corpusDir),
		counts,
		hit_at_5: aggregates.hitAt5,
		recall_at_5: aggregates.recallAt5,
		mrr: aggregates.mrr,
		noise_rate: noiseRate(loadStatsFile(options.statsPath)),
		per_query: perQuery,
	};
}

export async function runEvalRetrieval(options: RunEvalRetrievalOptions = {}): Promise<number> {
	const args = options.args ?? process.argv.slice(2);
	const fsImpl = options.fsImpl ?? fs;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const homedir = options.homedir ?? os.homedir;
	const corpusDir = options.corpusDir ?? parseArgValue(args, "--corpus");
	const outFile = options.outFile ?? parseArgValue(args, "--out");
	const goldenFile = parseArgValue(args, "--golden");

	if (!corpusDir) {
		err.error(
			"Usage: npx tsx scripts/eval-retrieval.ts --corpus <dir> [--sessions <dir>] [--out <file>] [--golden <file>]",
		);
		return 1;
	}

	const resolvedCorpusDir = path.resolve(corpusDir.replace(/^~(?=$|\/)/, homedir()));
	const sessionsDir =
		options.sessionsDir ?? parseArgValue(args, "--sessions") ?? defaultSessionsDir(resolvedCorpusDir);
	const resolvedSessionsDir = path.resolve(sessionsDir.replace(/^~(?=$|\/)/, homedir()));
	const roundsDir = path.join(resolvedCorpusDir, "rounds");
	const indexPath = path.join(roundsDir, "index.csv");
	const statsPath = path.join(resolvedCorpusDir, "chain-read-stats.json");
	const resolvedGoldenFile = goldenFile ? path.resolve(goldenFile.replace(/^~(?=$|\/)/, homedir())) : null;

	if (!fsImpl.existsSync(roundsDir)) {
		err.error(`Corpus rounds directory does not exist: ${roundsDir}`);
		return 1;
	}
	if (!fsImpl.existsSync(indexPath)) {
		err.error(`Corpus index does not exist: ${indexPath}`);
		return 1;
	}
	if (resolvedGoldenFile && !fsImpl.existsSync(resolvedGoldenFile)) {
		err.error(`Golden labels file does not exist: ${resolvedGoldenFile}`);
		return 1;
	}

	const result = evaluateRetrieval({
		corpusDir: resolvedCorpusDir,
		sessionsDir: resolvedSessionsDir,
		index: loadIndexFromPath(indexPath, fsImpl),
		rounds: readCorpusRounds(roundsDir, fsImpl),
		statsPath,
		configHash: sha256Json({
			git_rev: options.gitRev ?? "not_available_without_git",
			cli_args: {
				corpus: resolvedCorpusDir,
				sessions: resolvedSessionsDir,
				golden: resolvedGoldenFile,
			},
			metric_params: {
				top_k: 5,
				min_words: 20,
				labels: resolvedGoldenFile ? "golden" : "parentId+get_round_details",
				mrr_target: resolvedGoldenFile ? "primary_else_first_label" : "first_relevant",
				as_of: "strictly earlier",
			},
		}),
		goldenLabels: resolvedGoldenFile ? loadGoldenLabels(resolvedGoldenFile, fsImpl) : null,
		fsImpl,
	});

	const rendered = `${JSON.stringify(result, null, 2)}\n`;
	if (outFile) {
		fsImpl.mkdirSync(path.dirname(path.resolve(outFile)), { recursive: true });
		fsImpl.writeFileSync(path.resolve(outFile), rendered);
	} else {
		out.log(rendered.trimEnd());
	}
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const exitCode = await runEvalRetrieval();
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((error) => {
		console.error("❌ Error:", error);
		process.exit(1);
	});
}
