import crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { countWordsInMessageContent, shouldDropEmbedding } from "../lib/context-messages.ts";
import { collectWeakLabelExpansionsFromSessionJsonl } from "../lib/eval-labels.ts";
import { aggregateQueryMetrics, evaluateQueryMetrics, filterCandidatesAsOf, noiseRate } from "../lib/eval-metrics.ts";
import { indexRoundFileFromPath } from "../lib/index-io.ts";
import { type IndexEntry, loadIndexFromPath } from "../lib/index-storage.ts";
import type { RoundData } from "../lib/round-data.ts";
import { resolveScriptConfig, type ScriptConfigOptions } from "../lib/script-config.ts";
import { collectSearchRoundScores } from "../lib/search-interactions.ts";
import { loadStatsFile } from "../lib/stats.ts";

function defaultSessionsDir(agentDir: string): string {
	return path.resolve(agentDir, "sessions");
}

function parseArgValue(args: readonly string[], name: string): string | null {
	const index = args.indexOf(name);
	return index >= 0 ? (args[index + 1] ?? null) : null;
}

function sha256Json(value: unknown): string {
	return crypto.createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function gatherSessionFiles(sessionsDir: string, fsImpl: typeof fs = fs): string[] {
	if (!fsImpl.existsSync(sessionsDir)) return [];
	const sessionDirs = fsImpl
		.readdirSync(sessionsDir)
		.filter((entry) => entry.startsWith("--"))
		.map((entry) => path.join(sessionsDir, entry));

	return sessionDirs
		.flatMap((dir) =>
			fsImpl
				.readdirSync(dir)
				.filter((entry) => entry.endsWith(".jsonl"))
				.map((entry) => path.join(dir, entry)),
		)
		.sort();
}

function readCorpusRounds(roundsDir: string, fsImpl: typeof fs = fs): Map<string, RoundData> {
	const roundFiles = fsImpl
		.readdirSync(roundsDir)
		.filter((entry) => entry.endsWith(".json") && entry !== "chain-read-stats.json")
		.sort();

	return new Map(
		roundFiles.map((roundFile) => [
			roundFile,
			JSON.parse(fsImpl.readFileSync(path.join(roundsDir, roundFile), "utf-8")) as RoundData,
		]),
	);
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
	kind: "weak";
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

export function evaluateRetrieval(options: {
	corpusDir: string;
	sessionsDir: string;
	index: readonly IndexEntry[];
	rounds: ReadonlyMap<string, RoundData>;
	statsPath: string;
	configHash: string;
	fsImpl?: typeof fs;
}): EvalRetrievalResult {
	const fsImpl = options.fsImpl ?? fs;
	const weakExpansions = collectWeakLabelsByQuery(options.sessionsDir, fsImpl);
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

	const readRound = (filePath: string): RoundData | null => {
		const roundFile = path.basename(indexRoundFileFromPath(filePath));
		return options.rounds.get(roundFile) ?? null;
	};

	for (const [roundFile, round] of roundEntries) {
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
		const labels = dedupe([
			...(typeof round.parentId === "string" && round.parentId ? [round.parentId] : []),
			...(weakExpansions.get(roundFile) ?? []),
		]).filter((label) => label !== roundFile);
		if (labels.length === 0) {
			counts.skipped_no_labels++;
			continue;
		}

		const ranked = rankedScores.map((entry) => entry.fileName);
		const metrics = evaluateQueryMetrics({ labels, ranked });
		metricQueries.push({ labels, ranked });
		perQuery.push({
			query: roundFile,
			labels,
			top5: rankedScores.map((entry) => ({ file: entry.fileName, score: entry.bestScore })),
			first_hit_rank: metrics.firstHitRank,
		});
	}

	const aggregates = aggregateQueryMetrics(metricQueries);
	return {
		kind: "weak",
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
	const config = resolveScriptConfig(options);
	const fsImpl = options.fsImpl ?? fs;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const homedir = options.homedir ?? os.homedir;
	const corpusDir = options.corpusDir ?? parseArgValue(args, "--corpus");
	const sessionsDir = options.sessionsDir ?? parseArgValue(args, "--sessions") ?? defaultSessionsDir(config.agentDir);
	const outFile = options.outFile ?? parseArgValue(args, "--out");

	if (!corpusDir) {
		err.error("Usage: npx tsx scripts/eval-retrieval.ts --corpus <dir> [--sessions <dir>] [--out <file>]");
		return 1;
	}

	const resolvedCorpusDir = path.resolve(corpusDir.replace(/^~(?=$|\/)/, homedir()));
	const resolvedSessionsDir = path.resolve(sessionsDir.replace(/^~(?=$|\/)/, homedir()));
	const roundsDir = path.join(resolvedCorpusDir, "rounds");
	const indexPath = path.join(roundsDir, "index.csv");
	const statsPath = path.join(resolvedCorpusDir, "chain-read-stats.json");

	if (!fsImpl.existsSync(roundsDir)) {
		err.error(`Corpus rounds directory does not exist: ${roundsDir}`);
		return 1;
	}
	if (!fsImpl.existsSync(indexPath)) {
		err.error(`Corpus index does not exist: ${indexPath}`);
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
			cli_args: { corpus: resolvedCorpusDir, sessions: resolvedSessionsDir },
			metric_params: { top_k: 5, min_words: 20, labels: "parentId+get_round_details", as_of: "strictly earlier" },
		}),
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
