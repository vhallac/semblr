/**
 * digest-all.ts — Bulk-embed all pi session JSONL files into the semblr index.
 *
 * Iterates every session in ~/.pi/agent/sessions/, skips already-indexed rounds,
 * and embeds through the configured Semblr provider/model.
 *
 * Usage:
 *   npx tsx scripts/digest-all.ts
 *
 * Re-runnable: appends new rows and rewrites rows whose explicit model differs
 * from the current configured embedding model.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { type EmbeddingModelRegistry, embedText, normalize } from "../lib/embed.ts";
import { computeContentHash } from "../lib/hash.ts";
import {
	appendVectorIndexEntry,
	findStaleContentMatches as findStaleContentMatchesInDir,
	loadIndexedRoundFiles,
	loadRoundFilesWithDifferentModel,
	migrateIndexEntries as migrateIndexEntriesFile,
	replaceIndexEntriesForRoundFile,
	type VectorIndexEntry,
} from "../lib/index-io.ts";
import { type ParsedPiRound, parsePiSessionJsonl } from "../lib/pi-session.ts";
import {
	resolveScriptApiKey,
	resolveScriptConfig,
	resolveScriptIndexPath,
	resolveScriptModelRegistry,
	type ScriptConfigOptions,
	scriptEmbeddingConfig,
} from "../lib/script-config.ts";
import {
	appendToolIndexRows,
	buildToolIndexRows,
	buildToolIndexRowsFromRoundsDir,
	loadToolIndexedRoundFiles,
	toolIndexPathForRoundsDir,
	writeToolIndexRows,
} from "../lib/search-tools.ts";

// ─────────────────────────────────────────────
// Config (matches digest-session.ts)
// ─────────────────────────────────────────────

function defaultSessionsDir(agentDir: string): string {
	return path.resolve(agentDir, "sessions");
}

// Keep bulk digest single-worker: stale-hash migrations rewrite index.csv and delete
// duplicate files, so concurrent duplicate-content rounds can race and leave stale refs.
const CONCURRENCY = 1;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

type Round = ParsedPiRound & { sessionLabel: string };

// ─────────────────────────────────────────────
// Parse a single JSONL file into rounds
// ─────────────────────────────────────────────

function parseSessionFile(filePath: string, sessionLabel: string, deps: { fsImpl?: typeof fs } = {}): Round[] {
	const f = deps.fsImpl ?? fs;
	return parsePiSessionJsonl(f.readFileSync(filePath, "utf-8"), {
		sessionLabel,
		skipShortFinalResponse: true,
	});
}

// ─────────────────────────────────────────────
// Gather session JSONL files from a directory
// ─────────────────────────────────────────────

function gatherSessionFiles(
	sessionsDir: string,
	deps: { fsImpl?: typeof fs } = {},
): Array<{ filePath: string; label: string }> {
	const f = deps.fsImpl ?? fs;
	if (!f.existsSync(sessionsDir)) return [];

	const sessionDirs = f
		.readdirSync(sessionsDir)
		.filter((d) => d.startsWith("--"))
		.map((d) => path.join(sessionsDir, d));

	const jsonlFiles: Array<{ filePath: string; label: string }> = [];
	for (const dir of sessionDirs) {
		const label = path.basename(dir);
		const files = f.readdirSync(dir).filter((fn) => fn.endsWith(".jsonl"));
		for (const fn of files) {
			jsonlFiles.push({ filePath: path.join(dir, fn), label });
		}
	}

	jsonlFiles.sort((a, b) => a.filePath.localeCompare(b.filePath));
	return jsonlFiles;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface DigestAllOptions extends ScriptConfigOptions {
	sessionsDir?: string;
	roundsDir?: string;
	indexPath?: string;
	apiKey?: string;
	fetchImpl?: typeof fetch;
	modelRegistry?: EmbeddingModelRegistry;
	concurrency?: number;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: typeof fs;
	/** Rebuild the tool-call fulltext index from existing round files only — no re-embedding. */
	toolsOnly?: boolean;
}

export async function runDigestAll(options: DigestAllOptions = {}): Promise<number> {
	const config = resolveScriptConfig(options);
	const sessionsDir = options.sessionsDir ?? defaultSessionsDir(config.agentDir);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const indexPath = resolveScriptIndexPath(config, roundsDir, options.indexPath);
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const f = options.fsImpl ?? fs;

	if (options.toolsOnly) {
		const toolIndexPath = toolIndexPathForRoundsDir(roundsDir);
		const rows = f.existsSync(roundsDir) ? buildToolIndexRowsFromRoundsDir(roundsDir, f) : [];
		writeToolIndexRows(toolIndexPath, rows, f);
		out.log(
			`✅ Rebuilt tool index: ${rows.length} rows across ${new Set(rows.map((r) => r.hash)).size} rounds at ${toolIndexPath}`,
		);
		return 0;
	}

	const modelRegistry = resolveScriptModelRegistry(config, options);
	const embeddingConfig = scriptEmbeddingConfig(config);
	const concurrency = Math.max(1, options.concurrency ?? CONCURRENCY);

	const rawApiKey = await resolveScriptApiKey(config, { ...options, modelRegistry });
	if (!rawApiKey) {
		err.error("❌ OPENROUTER_API_KEY environment variable required");
		return 1;
	}
	const apiKey: string = rawApiKey;

	// Gather all session JSONL files
	const jsonlFiles = gatherSessionFiles(sessionsDir, { fsImpl: f });

	out.log(
		`📂 Found ${jsonlFiles.length} session files across ${new Set(jsonlFiles.map((j) => j.label)).size} directories\n`,
	);

	// Ensure rounds dir
	f.mkdirSync(roundsDir, { recursive: true });

	// Load existing index dedup set and explicit model mismatches.
	// Legacy two-column rows have no model and are treated as current per #62.
	const existingRounds = loadIndexedRoundFiles(indexPath);
	const modelMismatchedRounds = loadRoundFilesWithDifferentModel(indexPath, config.embeddingModel);
	out.log(`📊 Already indexed: ${existingRounds.size} rounds`);
	out.log(`📊 Model-mismatched rounds to re-index: ${modelMismatchedRounds.size}\n`);

	// Parse all sessions into a flat list of rounds (skipping already-indexed)
	const allRounds: Round[] = [];
	let skippedTotal = 0;

	for (const { filePath, label } of jsonlFiles) {
		const rounds = parseSessionFile(filePath, label, { fsImpl: f });
		const newRounds = rounds.filter((t) => {
			const key = `${computeContentHash(t.userPrompt, t.responseSequence, t.toolCalls)}.json`;
			return !existingRounds.has(key) || modelMismatchedRounds.has(key);
		});
		skippedTotal += rounds.length - newRounds.length;
		allRounds.push(...newRounds);
	}

	const totalNew = allRounds.length;
	out.log(`📊 New rounds to embed: ${totalNew} (${skippedTotal} already indexed)\n`);

	if (totalNew === 0) {
		out.log("✨ Nothing to do — all sessions already indexed!");
		return 0;
	}

	// Parallel embedding with concurrency limit
	let completed = 0;
	let errors = 0;

	async function processRound(round: Round): Promise<void> {
		const roundFile = `${computeContentHash(round.userPrompt, round.responseSequence, round.toolCalls)}.json`;
		const roundId = `${round.sessionLabel}/${roundFile}`;

		// Check for stale files whose stored full hash material belongs under this
		// content-hash filename. If found, migrate old index entries before deleting
		// old files so the round remains retrievable even if embedding is unavailable.
		const staleFiles = findStaleContentMatchesInDir(roundsDir, roundFile);
		for (const staleFile of staleFiles) {
			migrateIndexEntriesFile(indexPath, staleFile, roundFile);
		}

		// Write round file before deleting stale copies.
		f.writeFileSync(path.resolve(roundsDir, roundFile), JSON.stringify(round, null, 2));
		for (const staleFile of staleFiles) {
			f.unlinkSync(path.join(roundsDir, staleFile));
			err.error(`  ♻️  Migrated stale: ${staleFile} → ${roundFile}`);
		}

		// Tool-call fulltext index — independent of embedding, so re-run it even if
		// this round only needs re-embedding due to a model change.
		if (round.toolCalls.length > 0) {
			const toolIndexPath = toolIndexPathForRoundsDir(roundsDir);
			const alreadyToolIndexed = loadToolIndexedRoundFiles(toolIndexPath, f);
			if (!alreadyToolIndexed.has(roundFile)) {
				appendToolIndexRows(toolIndexPath, roundsDir, buildToolIndexRows(roundFile, round.toolCalls), { fsImpl: f });
			}
		}

		// Skip embedding if already indexed under the correct hash and current model
		// (must reload after potential migrations above).
		const indexedAfterCleanup = loadIndexedRoundFiles(indexPath);
		const modelMismatchedAfterCleanup = loadRoundFilesWithDifferentModel(indexPath, config.embeddingModel);
		const needsModelReindex = modelMismatchedAfterCleanup.has(roundFile);
		if (indexedAfterCleanup.has(roundFile) && !needsModelReindex) {
			completed++;
			err.error(`  ⏭  [${completed}/${totalNew}] ${roundId} (already indexed)`);
			return;
		}

		try {
			const entries: VectorIndexEntry[] = [];
			const promptVector = await embedText(round.userPrompt.slice(0, config.embeddingMaxTokens), apiKey, {
				fetchImpl: options.fetchImpl,
				config: embeddingConfig,
				modelRegistry,
			});
			entries.push({
				vector: normalize(promptVector),
				filePath: `${roundFile}:prompt`,
				model: config.embeddingModel,
			});

			const respText = round.responseSequence.slice(0, config.embeddingMaxTokens);
			if (respText) {
				const respVector = await embedText(respText, apiKey, {
					fetchImpl: options.fetchImpl,
					config: embeddingConfig,
					modelRegistry,
				});
				entries.push({
					vector: normalize(respVector),
					filePath: `${roundFile}:response`,
					model: config.embeddingModel,
				});
			}

			if (needsModelReindex) {
				replaceIndexEntriesForRoundFile(indexPath, roundFile, entries);
			} else {
				for (const entry of entries) {
					appendVectorIndexEntry(indexPath, entry.vector, entry.filePath, entry.model);
				}
			}

			completed++;
			const pct = ((completed / totalNew) * 100).toFixed(1);
			const action = needsModelReindex ? "re-indexed" : "embedded";
			err.error(`  ✅ [${completed}/${totalNew} ${pct}%] ${roundId} (${action})`);
		} catch (e) {
			errors++;
			err.error(`  ❌ [ERROR] ${roundId}: ${(e as Error).message}`);
		}
	}

	// Run with concurrency limit
	const queue = [...allRounds];
	const workers: Promise<void>[] = [];

	for (let i = 0; i < concurrency; i++) {
		workers.push(
			(async () => {
				while (queue.length > 0) {
					const round = queue.shift();
					if (round) await processRound(round);
				}
			})(),
		);
	}

	const startTime = Date.now();
	await Promise.all(workers);
	const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

	const finalCount = f.existsSync(indexPath)
		? f.readFileSync(indexPath, "utf-8").trim().split("\n").filter(Boolean).length
		: 0;

	out.log(`\n✅ Done in ${elapsed}s. ${completed} rounds embedded, ${errors} errors.`);
	out.log(`   Index: ${finalCount} vectors at ${indexPath}`);
	out.log(
		`   Rounds: ${f.readdirSync(roundsDir).filter((rf) => rf.endsWith(".json") && !rf.startsWith("index")).length} files`,
	);
	return 0;
}

// ─────────────────────────────────────────────
// CLI entry point
// ─────────────────────────────────────────────

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const toolsOnly = process.argv.includes("--tools-only");
	const exitCode = await runDigestAll({ toolsOnly });
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((err) => {
		console.error("❌ Fatal:", err);
		process.exit(1);
	});
}
