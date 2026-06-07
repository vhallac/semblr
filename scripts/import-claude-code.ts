/**
 * import-claude-code.ts — Import Claude Code JSONL history into Semblr.
 *
 * Reads JSONL files under ~/.claude/projects, converts each real user turn into the
 * Semblr round format, and embeds prompt + response into the shared index.
 *
 * Usage:
 *   OPENROUTER_API_KEY="$(pass show ai/openrouter)" npx tsx scripts/import-claude-code.ts
 *
 * Options:
 *   --dry-run              Parse/count only; do not write or embed
 *   --include-sidechains   Include Claude Code subagent/sidechain transcripts
 *   --limit N             Import at most N new rounds (useful for testing)
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { type ClaudeRound, claudeRoundFileName, parseClaudeCodeJsonl } from "../lib/claude-code.ts";
import { type EmbeddingModelRegistry, embedText, normalize } from "../lib/embed.ts";
import { appendVectorIndexEntry, loadIndexedRoundFiles as loadIndexedRoundFilesFromIndex } from "../lib/index-io.ts";
import {
	resolveScriptApiKey,
	resolveScriptConfig,
	resolveScriptIndexPath,
	resolveScriptModelRegistry,
	type ScriptConfigOptions,
	scriptEmbeddingConfig,
} from "../lib/script-config.ts";

const CLAUDE_PROJECTS_DIR = process.env.CLAUDE_PROJECTS_DIR || path.resolve(os.homedir(), ".claude", "projects");
const CONCURRENCY = Number(process.env.SEMBLR_IMPORT_CONCURRENCY || "5");

type Round = ClaudeRound;

function argValue(name: string): string | null {
	const idx = process.argv.indexOf(name);
	return idx >= 0 ? (process.argv[idx + 1] ?? null) : null;
}

// ─────────────────────────────────────────────
// Walk JSONL files
// ─────────────────────────────────────────────

export function walkJsonlFiles(dir: string, deps: { fsImpl?: typeof fs } = {}): string[] {
	const f = deps.fsImpl ?? fs;
	if (!f.existsSync(dir)) return [];
	const out: string[] = [];
	for (const entry of f.readdirSync(dir, { withFileTypes: true })) {
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) out.push(...walkJsonlFiles(full, deps));
		else if (entry.isFile() && entry.name.endsWith(".jsonl")) out.push(full);
	}
	return out.sort();
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface ImportClaudeCodeOptions extends ScriptConfigOptions {
	projectsDir?: string;
	roundsDir?: string;
	indexPath?: string;
	dryRun?: boolean;
	includeSidechains?: boolean;
	limit?: number | null;
	apiKey?: string;
	fetchImpl?: typeof fetch;
	modelRegistry?: EmbeddingModelRegistry;
	concurrency?: number;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: typeof fs;
}

export async function runImportClaudeCode(options: ImportClaudeCodeOptions = {}): Promise<number> {
	const config = resolveScriptConfig(options);
	const projectsDir = options.projectsDir ?? CLAUDE_PROJECTS_DIR;
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const indexPath = resolveScriptIndexPath(config, roundsDir, options.indexPath);
	const modelRegistry = resolveScriptModelRegistry(config, options);
	const embeddingConfig = scriptEmbeddingConfig(config);
	const dryRun = options.dryRun ?? false;
	const includeSidechains = options.includeSidechains ?? false;
	const limit = options.limit ?? null;
	const concurrency = Math.max(1, options.concurrency ?? CONCURRENCY);
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const f = options.fsImpl ?? fs;

	const apiKey = dryRun ? null : await resolveScriptApiKey(config, { ...options, modelRegistry });
	if (!dryRun && !apiKey) {
		err.error("❌ OPENROUTER_API_KEY required");
		return 1;
	}

	const files = walkJsonlFiles(projectsDir, { fsImpl: f });
	out.log(`📂 Found ${files.length} Claude Code JSONL files in ${projectsDir}`);
	out.log(`   sidechains: ${includeSidechains ? "included" : "skipped (use --include-sidechains to include)"}`);

	const allRounds: Round[] = [];
	let parseErrors = 0;
	for (const file of files) {
		try {
			allRounds.push(
				...parseClaudeCodeJsonl(f.readFileSync(file, "utf-8"), {
					filePath: file,
					projectsDir,
					includeSidechains,
				}),
			);
		} catch (e) {
			parseErrors++;
			err.error(`⚠ ${(e as Error).message}`);
		}
	}

	const indexed = loadIndexedRoundFilesFromIndex(indexPath);
	const seen = new Set<string>();
	let duplicates = 0;
	let alreadyIndexed = 0;
	let newRounds = allRounds.filter((round) => {
		const file = claudeRoundFileName(round);
		if (seen.has(file)) {
			duplicates++;
			return false;
		}
		seen.add(file);
		if (indexed.has(file)) {
			alreadyIndexed++;
			return false;
		}
		return true;
	});
	if (limit != null && Number.isFinite(limit)) newRounds = newRounds.slice(0, limit);

	out.log(`📊 Parsed rounds: ${allRounds.length}`);
	out.log(
		`📊 New rounds to import: ${newRounds.length} (${alreadyIndexed} already indexed, ${duplicates} duplicate in source, ${parseErrors} parse errors)`,
	);

	if (dryRun || newRounds.length === 0) return 0;

	f.mkdirSync(roundsDir, { recursive: true });
	let completed = 0;
	let errors = 0;

	async function processRound(round: Round) {
		const file = claudeRoundFileName(round);
		f.writeFileSync(path.resolve(roundsDir, file), JSON.stringify(round, null, 2));
		try {
			const promptVec = await embedText(round.userPrompt.slice(0, config.embeddingMaxTokens), apiKey!, {
				fetchImpl: options.fetchImpl,
				config: embeddingConfig,
				modelRegistry,
			});
			appendVectorIndexEntry(indexPath, normalize(promptVec), `${file}:prompt`, config.embeddingModel);
			const respText = round.responseSequence.slice(0, config.embeddingMaxTokens);
			if (respText) {
				const respVec = await embedText(respText, apiKey!, {
					fetchImpl: options.fetchImpl,
					config: embeddingConfig,
					modelRegistry,
				});
				appendVectorIndexEntry(indexPath, normalize(respVec), `${file}:response`, config.embeddingModel);
			}
			completed++;
			err.error(`  ✅ [${completed}/${newRounds.length}] ${file} ${round.cwd ?? ""}`);
		} catch (e) {
			errors++;
			err.error(`  ❌ [ERROR] ${file}: ${(e as Error).message}`);
		}
	}

	const queue = [...newRounds];
	const workers = Array.from({ length: concurrency }, async () => {
		while (queue.length > 0) await processRound(queue.shift()!);
	});

	const started = Date.now();
	await Promise.all(workers);
	const elapsed = ((Date.now() - started) / 1000).toFixed(1);
	const vectors = f.existsSync(indexPath) ? f.readFileSync(indexPath, "utf-8").split("\n").filter(Boolean).length : 0;
	const roundFiles = f.existsSync(roundsDir)
		? f.readdirSync(roundsDir).filter((rf) => rf.endsWith(".json")).length
		: 0;

	out.log(`\n✅ Done in ${elapsed}s. ${completed} Claude Code rounds imported, ${errors} errors.`);
	out.log(`   Index: ${vectors} vectors at ${indexPath}`);
	out.log(`   Rounds: ${roundFiles} files at ${roundsDir}`);
	return 0;
}

// ─────────────────────────────────────────────
// CLI entry point
// ─────────────────────────────────────────────

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const exitCode = await runImportClaudeCode({
		dryRun: process.argv.includes("--dry-run"),
		includeSidechains: process.argv.includes("--include-sidechains"),
		limit: argValue("--limit") !== null ? Number(argValue("--limit")) : null,
	});
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((e) => {
		console.error("❌ Fatal:", e);
		process.exit(1);
	});
}
