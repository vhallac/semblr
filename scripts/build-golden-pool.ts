import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { gatherSessionFiles, parseArgValue, readCorpusRounds } from "../lib/eval-corpus.ts";
import {
	collectReplayableGoldenUniverse,
	poolCandidates,
	selectGoldenQueries,
	type WeakBaselineEntry,
} from "../lib/eval-golden.ts";
import { collectWeakLabelExpansionsFromSessionJsonl } from "../lib/eval-labels.ts";
import { loadIndexFromPath } from "../lib/index-storage.ts";
import type { RoundData } from "../lib/round-data.ts";

const SEMBLR_ACTIVITY_SIGNALS = [
	"search_interactions",
	"get_round_details",
	"get_tool_details",
	"RECENCY LIST",
	"RELEVANCE LIST",
	"CONTEXT BUILDING REFERENCES",
] as const;

function expandHome(value: string, homedir: () => string): string {
	return value.replace(/^~(?=$|\/)/, homedir());
}

function defaultSessionsDir(corpusDir: string): string {
	return path.resolve(corpusDir, "sessions");
}

function defaultBaselinePath(): string {
	return path.resolve("docs/eval/baseline-weak.json");
}

function defaultPoolPath(): string {
	return path.resolve("docs/eval/golden-pool.local.json");
}

function defaultWorksheetPath(): string {
	return path.resolve("docs/eval/golden-worksheet.local.md");
}

function parseIntArg(args: readonly string[], name: string, fallback: number): number {
	const raw = parseArgValue(args, name);
	if (!raw) return fallback;
	const parsed = Number.parseInt(raw, 10);
	return Number.isFinite(parsed) ? parsed : fallback;
}

function readWeakEntries(filePath: string, fsImpl: typeof fs = fs): WeakBaselineEntry[] {
	const parsed = JSON.parse(fsImpl.readFileSync(filePath, "utf-8")) as { per_query?: WeakBaselineEntry[] };
	return Array.isArray(parsed.per_query) ? parsed.per_query : [];
}

function isSemblrActiveSession(rawSession: string): boolean {
	return SEMBLR_ACTIVITY_SIGNALS.some((signal) => rawSession.includes(signal));
}

function compactExcerpt(text: string): string {
	return text.replace(/\s+/g, " ").trim();
}

export function buildCandidateExcerpt(
	round: Partial<Pick<RoundData, "userPrompt" | "responseSequence">>,
	maxLength = 250,
): string {
	const prompt = compactExcerpt(round.userPrompt ?? "");
	const response = compactExcerpt(round.responseSequence ?? "");
	const combined = response ? `${prompt} ${response}`.trim() : prompt;
	if (combined.length <= maxLength) return combined;
	if (maxLength <= 1) return combined.slice(0, maxLength);
	return `${combined.slice(0, maxLength - 1).trimEnd()}…`;
}

export function collectQueryInfo(
	sessionsDir: string,
	fsImpl: typeof fs = fs,
): Map<string, { expandedRounds: string[]; semblrActive: boolean }> {
	const info = new Map<string, { expandedRounds: string[]; semblrActive: boolean }>();
	for (const sessionFile of gatherSessionFiles(sessionsDir, fsImpl)) {
		const rawSession = fsImpl.readFileSync(sessionFile, "utf-8");
		const labels = collectWeakLabelExpansionsFromSessionJsonl(rawSession);
		const semblrActive = isSemblrActiveSession(rawSession);
		for (const [query, expandedRounds] of labels) {
			info.set(query, { expandedRounds, semblrActive });
		}
	}
	return info;
}

function renderWorksheet(pool: GoldenPoolFile): string {
	const sections = pool.queries.map((query, index) => {
		const candidates = query.candidates
			.map((candidate) => `- [ ] ${candidate.file}\n      excerpt: ${candidate.excerpt}`)
			.join("\n");
		return `## ${index + 1}. ${query.query}\n\nPrompt:\n\n> ${query.prompt.replace(/\n/g, "\n> ")}\n\nDifficulty: ${query.difficulty}\n\nMark all true labels from the pool below by changing [ ] to [x]. Mark one selected round as (primary) on its checkbox line, for example: - [x] candidate.json (primary)\n\n${candidates}`;
	});

	return `# Golden worksheet\n\nSource pool: ${pool.source_pool}\nCorpus: ${pool.corpus}\n\n${sections.join("\n\n")}${sections.length ? "\n" : ""}`;
}

export interface GoldenPoolCandidate {
	file: string;
	excerpt: string;
}

export interface GoldenPoolQuery {
	query: string;
	prompt: string;
	difficulty: "miss" | "hard" | "control";
	candidates: GoldenPoolCandidate[];
}

export interface GoldenPoolFile {
	kind: "golden-pool";
	version: 1;
	corpus: string;
	source_pool: string;
	queries: GoldenPoolQuery[];
}

export interface RunBuildGoldenPoolOptions {
	args?: string[];
	corpusDir?: string;
	sessionsDir?: string;
	baselineFile?: string;
	outFile?: string;
	worksheetFile?: string;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: typeof fs;
	homedir?: () => string;
}

export async function runBuildGoldenPool(options: RunBuildGoldenPoolOptions = {}): Promise<number> {
	const args = options.args ?? process.argv.slice(2);
	const fsImpl = options.fsImpl ?? fs;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const homedir = options.homedir ?? os.homedir;
	const corpusDir = options.corpusDir ?? parseArgValue(args, "--corpus");
	const outFile = options.outFile ?? parseArgValue(args, "--out") ?? defaultPoolPath();
	const worksheetFile = options.worksheetFile ?? parseArgValue(args, "--worksheet") ?? defaultWorksheetPath();
	const baselineFile = options.baselineFile ?? parseArgValue(args, "--baseline") ?? defaultBaselinePath();
	const seed = parseIntArg(args, "--seed", 1);
	const count = parseIntArg(args, "--count", 30);

	if (!corpusDir) {
		err.error(
			"Usage: npx tsx scripts/build-golden-pool.ts --corpus <dir> [--sessions <dir>] [--baseline <file>] [--out <file>] [--worksheet <file>] [--seed <n>] [--count <n>]",
		);
		return 1;
	}

	const resolvedCorpusDir = path.resolve(expandHome(corpusDir, homedir));
	const sessionsDir =
		options.sessionsDir ?? parseArgValue(args, "--sessions") ?? defaultSessionsDir(resolvedCorpusDir);
	const resolvedSessionsDir = path.resolve(expandHome(sessionsDir, homedir));
	const resolvedBaselineFile = path.resolve(expandHome(baselineFile, homedir));
	const resolvedOutFile = path.resolve(expandHome(outFile, homedir));
	const resolvedWorksheetFile = path.resolve(expandHome(worksheetFile, homedir));
	const roundsDir = path.join(resolvedCorpusDir, "rounds");
	const indexPath = path.join(roundsDir, "index.csv");

	if (!fsImpl.existsSync(roundsDir)) {
		err.error(`Corpus rounds directory does not exist: ${roundsDir}`);
		return 1;
	}
	if (!fsImpl.existsSync(indexPath)) {
		err.error(`Corpus index does not exist: ${indexPath}`);
		return 1;
	}
	if (!fsImpl.existsSync(resolvedBaselineFile)) {
		err.error(`Weak baseline does not exist: ${resolvedBaselineFile}`);
		return 1;
	}

	const rounds = readCorpusRounds(roundsDir, fsImpl);
	const index = loadIndexFromPath(indexPath, fsImpl);
	const weakEntries = readWeakEntries(resolvedBaselineFile, fsImpl);
	const queryInfo = collectQueryInfo(resolvedSessionsDir, fsImpl);
	const universe = collectReplayableGoldenUniverse(rounds, queryInfo, weakEntries);
	const universeByQuery = new Map(universe.map((query) => [query.query, query]));
	const chosen = selectGoldenQueries(universe, { seed, count });
	const pool: GoldenPoolFile = {
		kind: "golden-pool",
		version: 1,
		corpus: path.basename(resolvedCorpusDir),
		source_pool: resolvedOutFile,
		queries: chosen.map((candidate, indexOffset) => {
			const query = universeByQuery.get(candidate.query);
			if (!query) throw new Error(`Selected golden query missing from replayable universe: ${candidate.query}`);
			return {
				query: query.query,
				prompt: query.prompt,
				difficulty: query.difficulty,
				candidates: poolCandidates(query, { index, rounds, seed: seed + indexOffset }).map((file) => ({
					file,
					excerpt: buildCandidateExcerpt(rounds.get(file) ?? {}),
				})),
			};
		}),
	};

	fsImpl.mkdirSync(path.dirname(resolvedOutFile), { recursive: true });
	fsImpl.writeFileSync(resolvedOutFile, `${JSON.stringify(pool, null, 2)}\n`);
	fsImpl.mkdirSync(path.dirname(resolvedWorksheetFile), { recursive: true });
	fsImpl.writeFileSync(resolvedWorksheetFile, renderWorksheet(pool));
	out.log(`Golden pool written to ${resolvedOutFile}`);
	out.log(`Golden worksheet written to ${resolvedWorksheetFile}`);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const exitCode = await runBuildGoldenPool();
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((error) => {
		console.error("❌ Error:", error);
		process.exit(1);
	});
}
