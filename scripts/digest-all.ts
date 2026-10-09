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
import {
	bm25IndexPathForRoundsDir,
	deleteBm25Round,
	loadBm25Index,
	roundTextForBm25,
	upsertBm25Round,
	writeBm25Index,
} from "../lib/bm25-index.ts";
import { type EmbeddingModelRegistry, embedText } from "../lib/embed.ts";
import { embedRound } from "../lib/embed-round.ts";
import {
	appendVectorIndexEntry,
	findStaleContentMatches as findStaleContentMatchesInDir,
	loadIndexedRoundFiles,
	loadRoundFilesWithDifferentModel,
	loadVectorIndex,
	migrateIndexEntries as migrateIndexEntriesFile,
	replaceIndexEntriesForRoundFile,
	type VectorIndexEntry,
} from "../lib/index-io.ts";
import { type ParsedPiRound, parsePiSessionJsonl } from "../lib/pi-session.ts";
import { deriveRoundFile, embeddingMaxTokensToResponseBytes } from "../lib/round-capture.ts";
import { buildCheckpointSummaryText, type CheckpointSummary } from "../lib/round-data.ts";
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

type Round = ParsedPiRound & {
	sessionLabel: string;
	/** Checkpoint summary carried by recovered round files; embeds as a `:summary` row (F1 parity). */
	summary?: CheckpointSummary;
	/** Stored embedding marker (F2); written through embedRound's writeRoundEmbedding dep. */
	promptEmbedding?: number[];
};

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
// F2 sweep heal (PR #134 review, option b)
// ─────────────────────────────────────────────

/**
 * Write the `promptEmbedding` marker onto a round file that already has index
 * rows but no marker, without re-embedding. The value is the round's
 * `:response` index-row vector (see the call site for why the combined vector
 * is unavailable and why grouping is therefore unsupported for healed rounds).
 *
 * Atomic tmp+rename, matching the extension's round-file write
 * (src/semblr.ts writeRoundEmbedding). Round files are the durable store; a
 * truncating write here could destroy a round that has no session JSONL left
 * to recover from. Returns true when a marker was written.
 */
function healMissingPromptEmbedding(
	fileName: string,
	roundsDir: string,
	responseVectorByRoundFile: Map<string, number[]>,
	f: typeof fs,
): boolean {
	const roundPath = path.join(roundsDir, fileName);
	let round: Round;
	try {
		round = JSON.parse(f.readFileSync(roundPath, "utf-8")) as Round;
	} catch {
		return false;
	}
	if (round.promptEmbedding) return false;
	const responseVec = responseVectorByRoundFile.get(fileName);
	if (!responseVec) return false;
	round.promptEmbedding = responseVec;
	const tmpPath = `${roundPath}.tmp.${process.pid}`;
	f.writeFileSync(tmpPath, JSON.stringify(round, null, 2));
	f.renameSync(tmpPath, roundPath);
	return true;
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
	const bm25IndexPath = bm25IndexPathForRoundsDir(roundsDir);
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const f = options.fsImpl ?? fs;

	if (options.toolsOnly) {
		const toolIndexPath = toolIndexPathForRoundsDir(roundsDir);
		const rows = f.existsSync(roundsDir) ? buildToolIndexRowsFromRoundsDir(roundsDir, f) : [];
		f.mkdirSync(roundsDir, { recursive: true });
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
	// One pass over the index for the F2 sweep heal below. A per-file scan would
	// make the sweep O(n²); the response vector is the only index row a healed
	// marker can reuse without an embedding call.
	const responseVectorByRoundFile = new Map<string, number[]>();
	for (const entry of loadVectorIndex(indexPath)) {
		if (!entry.filePath.endsWith(":response")) continue;
		const roundFile = path.basename(entry.filePath.slice(0, -":response".length));
		if (!responseVectorByRoundFile.has(roundFile)) responseVectorByRoundFile.set(roundFile, entry.vector);
	}
	out.log(`📊 Already indexed: ${existingRounds.size} rounds`);
	out.log(`📊 Model-mismatched rounds to re-index: ${modelMismatchedRounds.size}\n`);

	// Parse all sessions into a flat list of rounds (skipping already-indexed)
	const allRounds: Round[] = [];
	let skippedTotal = 0;

	for (const { filePath, label } of jsonlFiles) {
		const rounds = parseSessionFile(filePath, label, { fsImpl: f });
		const newRounds = rounds.filter((t) => {
			const key = deriveRoundFile(t.userPrompt, t.responseSequence, t.toolCalls).fileName;
			return !existingRounds.has(key) || modelMismatchedRounds.has(key);
		});
		skippedTotal += rounds.length - newRounds.length;
		allRounds.push(...newRounds);
	}

	// Rounds-dir sweep (issue #133): recovered round files written by the
	// startup backfill may have no session JSONL left to scan from — but they
	// also have no vector-index rows yet (the backfill writes the round file
	// without embedding). Sweep every round file that is not already covered
	// by the session scan and is missing index rows under the current model,
	// so `just index` backfills their embeddings too. Rounds already indexed
	// under the current model are skipped — index presence, not the
	// promptEmbedding field, is the idempotence guard. Since F2 (PR #134
	// review) processRound embeds through embedRound (lib/embed-round.ts),
	// which writes the index rows and the promptEmbedding field together —
	// but the guard stays index presence, so a round whose rows are missing
	// is swept again even if a promptEmbedding marker survived.
	const queuedFiles = new Set(
		allRounds.map((r) => deriveRoundFile(r.userPrompt, r.responseSequence, r.toolCalls).fileName),
	);
	let sweptTotal = 0;
	let healedTotal = 0;
	if (f.existsSync(roundsDir)) {
		for (const fileName of f.readdirSync(roundsDir)) {
			if (!fileName.endsWith(".json") || fileName.startsWith("index")) continue;
			if (queuedFiles.has(fileName)) continue;
			if (existingRounds.has(fileName) && !modelMismatchedRounds.has(fileName)) {
				// F2 (PR #134 review, option b): a round that already has current-model
				// index rows may still lack the `promptEmbedding` marker (e.g. rows
				// survive while the round file was rewritten). The startup pending
				// counter keys on the marker, so `just index` must heal it here — or the
				// count names work this run cannot do. Healing reuses the round's
				// `:response` index-row vector instead of re-embedding: the live
				// combined `concat(prompt, response)` vector is not stored in the index
				// and cannot be reconstructed without an API call. The response-side
				// vector is already the accepted degraded marker on the shared-core
				// short-prompt path (lib/embed-round.ts), and every consumer gates on
				// truthiness and uses it as a real vector. Consequence: healed rounds
				// cannot support topic grouping (src/semblr.ts keys grouping on
				// promptEmbedding), because grouping needs the combined vector. This is
				// accepted for historical rounds — a migration can restore the combined
				// vector if grouping is ever built. Rounds with no `:response` row are
				// left marker-less (still truthfully pending).
				healMissingPromptEmbedding(fileName, roundsDir, responseVectorByRoundFile, f) && healedTotal++;
				continue;
			}
			let round: Round;
			try {
				round = JSON.parse(f.readFileSync(path.join(roundsDir, fileName), "utf-8")) as Round;
			} catch {
				err.error(`  ⚠️  Skipping unreadable round file: ${fileName}`);
				continue;
			}
			// F5 (PR #134 review): parseable-but-incomplete rounds would throw in
			// deriveRoundFile or the toolCalls length check inside processRound —
			// outside its try/catch — and kill the whole run naming no file. Validate
			// the fields the enqueue needs here; invalid files are skipped with a
			// named warning, mirroring the unreadable-file skip above.
			// F3 (PR #134 review): an array is not enough — a non-object element such
			// as `[null]` passes Array.isArray, then throws in deriveRoundFile
			// (computeContentHash) and in the tool-index row builders, outside any
			// try/catch, aborting the run naming no file. Require every element to be
			// a non-null object.
			if (
				typeof round.userPrompt !== "string" ||
				round.userPrompt.length === 0 ||
				typeof round.responseSequence !== "string" ||
				!Array.isArray(round.toolCalls) ||
				!round.toolCalls.every((tc) => tc !== null && typeof tc === "object")
			) {
				err.error(`  ⚠️  Skipping invalid round file: ${fileName}`);
				continue;
			}
			allRounds.push({ ...round, sessionLabel: "<rounds-dir>" });
			sweptTotal++;
		}
	}

	const totalNew = allRounds.length;
	out.log(
		`📊 New rounds to embed: ${totalNew} (${skippedTotal} already indexed, ${sweptTotal} swept from rounds dir${healedTotal > 0 ? `, ${healedTotal} markers healed` : ""})\n`,
	);

	if (totalNew === 0) {
		out.log("✨ Nothing to do — all sessions already indexed!");
		return 0;
	}

	// Parallel embedding with concurrency limit
	let completed = 0;
	let errors = 0;

	async function processRound(round: Round): Promise<void> {
		const roundFile = deriveRoundFile(round.userPrompt, round.responseSequence, round.toolCalls).fileName;
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
		const bm25Index = loadBm25Index(bm25IndexPath, f);
		upsertBm25Round(bm25Index, roundFile, roundTextForBm25(round));
		for (const staleFile of staleFiles) {
			deleteBm25Round(bm25Index, staleFile);
		}
		writeBm25Index(bm25IndexPath, bm25Index, f);
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
				appendToolIndexRows(toolIndexPath, roundsDir, buildToolIndexRows(roundFile, round.toolCalls), {
					fsImpl: f,
				});
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
			// F1+F2 (PR #134 review): redrive embedding goes through the shared policy
			// core embedRound (lib/embed-round.ts) — the same call the live agent_end
			// path and embedRecoveredRounds make. Response clip budget (24,000 bytes,
			// not 8,000 code units), checkpoint `:summary` rows, the short-prompt drop,
			// and the promptEmbedding write are parity by construction. Index rows are
			// buffered here and flushed once the embed resolves (below): full replace on
			// model re-index, append otherwise.
			const entries: VectorIndexEntry[] = [];
			await embedRound(
				{
					fileName: roundFile,
					userPrompt: round.userPrompt,
					responseText: round.responseSequence,
					checkpointSummaryText: round.summary ? buildCheckpointSummaryText(round.summary) : null,
					maxResponseBytes: embeddingMaxTokensToResponseBytes(config.embeddingMaxTokens),
					promptNoiseOptions: {
						fenceMaxChars: config.promptNoiseFenceMaxChars,
						jsonMaxChars: config.promptNoiseJsonMaxChars,
						repeatMaxChars: config.promptNoiseRepeatMaxChars,
					},
					promptMaxTokens: config.embeddingMaxTokens,
				},
				{
					embed: (text) =>
						embedText(text, apiKey, {
							fetchImpl: options.fetchImpl,
							config: embeddingConfig,
							modelRegistry,
						}),
					appendIndexRow: (label, vec, hash) => {
						entries.push({
							vector: vec,
							filePath: label,
							model: config.embeddingModel,
							embeddingInputHash: hash,
						});
					},
					writeRoundEmbedding: (fileName, vec) => {
						round.promptEmbedding = vec;
						f.writeFileSync(path.resolve(roundsDir, fileName), JSON.stringify(round, null, 2));
					},
				},
			);

			if (needsModelReindex) {
				replaceIndexEntriesForRoundFile(indexPath, roundFile, entries);
			} else {
				for (const entry of entries) {
					appendVectorIndexEntry(indexPath, entry.vector, entry.filePath, entry.model, entry.embeddingInputHash);
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
