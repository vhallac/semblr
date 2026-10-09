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
	buildStaleContentMatchMap,
	indexRoundFileFromPath,
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
import { extractCheckpointSummary } from "../lib/session-backfill.ts";

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
 * Outcome of trying to heal a `promptEmbedding` marker onto a round file that
 * already has current-model index rows.
 *
 * - `healed`: a marker was written from the round's `:response` row (no call).
 * - `already-marked`: the round already carries a marker — genuinely done.
 * - `unhealable`: no `:response` row to reuse (or the file is unreadable), so
 *   the marker cannot be synthesized without an API call. The sweep treats
 *   this as a retrieval gap and re-embeds the round when a source exists
 *   (F1, issue #140 decision D1/D2).
 */
type HealOutcome = "healed" | "already-marked" | "unhealable";

/**
 * Heal a marker-less round that already has current-model index rows, without
 * re-embedding. The marker value is the round's `:response` index-row vector
 * (see the call site for why the combined vector is unavailable and why
 * grouping is therefore unsupported for healed rounds).
 *
 * Atomic tmp+rename, matching the extension's round-file write
 * (src/semblr.ts writeRoundEmbedding). Round files are the durable store; a
 * truncating write here could destroy a round that has no session JSONL left
 * to recover from.
 */
function healMissingPromptEmbedding(
	fileName: string,
	roundsDir: string,
	responseVectorByRoundFile: Map<string, number[]>,
	f: typeof fs,
): HealOutcome {
	const roundPath = path.join(roundsDir, fileName);
	let round: Round;
	try {
		round = JSON.parse(f.readFileSync(roundPath, "utf-8")) as Round;
	} catch {
		return "unhealable";
	}
	if (round.promptEmbedding) return "already-marked";
	const responseVec = responseVectorByRoundFile.get(fileName);
	if (!responseVec) return "unhealable";
	round.promptEmbedding = responseVec;
	const tmpPath = `${roundPath}.tmp.${process.pid}`;
	f.writeFileSync(tmpPath, JSON.stringify(round, null, 2));
	f.renameSync(tmpPath, roundPath);
	return "healed";
}

// ─────────────────────────────────────────────
// F3 replace guard (issue #140, decision D3)
// ─────────────────────────────────────────────

/** The index label suffix of a round's row, or "" for a bare round-file row. */
function indexRowSuffix(filePath: string): string {
	const roundFile = path.basename(indexRoundFileFromPath(filePath));
	const label = path.basename(filePath);
	return label.slice(roundFile.length);
}

/**
 * Existing index rows for `roundFile` whose suffix is not reproduced by
 * `entries`. A model-change reindex replaces every row for the round, so any
 * suffix it does not reproduce (e.g. a `:summary` row whose source is gone)
 * would be silently dropped. These are returned so the caller can report and
 * preserve them (issue #140, decision D3: reproduce it or report it, never
 * trade it away). Comparison is by suffix, so the old copies of reproduced
 * suffixes are still replaced by their fresh rows.
 */
function findOrphanIndexEntries(
	indexPath: string,
	roundFile: string,
	entries: VectorIndexEntry[],
	f: typeof fs,
): VectorIndexEntry[] {
	const reproduced = new Set(entries.map((entry) => indexRowSuffix(entry.filePath)));
	return loadVectorIndex(indexPath, f).filter(
		(entry) =>
			path.basename(indexRoundFileFromPath(entry.filePath)) === roundFile &&
			!reproduced.has(indexRowSuffix(entry.filePath)),
	);
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

	// Issue #139: parse index.csv exactly once per run and derive every
	// index-level structure from that single pass — the dedup set, the
	// model-mismatch set, and the F2 sweep-heal response-vector map. processRound
	// used to re-read and re-parse index.csv twice per round (O(n²)); the
	// in-memory sets it now consumes are seeded from this one load.
	// Legacy two-column rows have no model and are treated as current per #62.
	const indexEntries = loadVectorIndex(indexPath, f);
	const existingRounds = new Set<string>();
	const modelMismatchedRounds = new Set<string>();
	// A per-file scan would make the sweep O(n²); the response vector is the only
	// index row a healed marker can reuse without an embedding call.
	const responseVectorByRoundFile = new Map<string, number[]>();
	for (const entry of indexEntries) {
		// indexRoundFileFromPath strips the :prompt/:response/:round/:summary suffix.
		const roundFile = path.basename(indexRoundFileFromPath(entry.filePath));
		existingRounds.add(roundFile);
		if (entry.model !== undefined && entry.model !== config.embeddingModel) modelMismatchedRounds.add(roundFile);
		if (entry.filePath.endsWith(":response") && !responseVectorByRoundFile.has(roundFile)) {
			responseVectorByRoundFile.set(roundFile, entry.vector);
		}
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
	// Issue #139: materialize stale-content matches once for the whole run.
	// processRound used to call findStaleContentMatchesInDir per round, which
	// re-scanned and JSON-parsed every round file on every iteration (O(store)
	// per round → O(n²)). The map is built from a single pass over the same
	// round-dir listing the sweep below walks; corrupt files are skipped exactly
	// as in the per-call helper, which stays for direct callers.
	const staleMatchesByTarget = buildStaleContentMatchMap(roundsDir, f);
	// Issue #140 F1: when the sweep re-embeds a marker-less round with no
	// `:response` row, a legacy file reachable only through a queued session
	// round must not also be swept under its own (stale) name — the queued
	// round's processRound migrates and deletes it, and a second enqueue would
	// unlink an already-removed file. Skip any file listed as a stale match for
	// a queued target; the target's migration is the single owner of that file.
	const staleFilesForQueued = new Set<string>();
	for (const target of queuedFiles) {
		for (const stale of staleMatchesByTarget.get(target) ?? []) staleFilesForQueued.add(stale);
	}
	// Issue #139: hoist the derived-index state the per-round path used to reload.
	// BM25 is loaded once here, upserted in memory inside processRound, and
	// flushed exactly once after the batch (mirroring indexRecoveredRounds'
	// flushBm25 seam, lib/session-backfill.ts). The tool-index and vector-index
	// sets are seeded once and updated in memory as rows are appended/replaced,
	// so processRound never re-parses index.csv or the tool index per round.
	const bm25Index = loadBm25Index(bm25IndexPath, f);
	const toolIndexPath = toolIndexPathForRoundsDir(roundsDir);
	const toolIndexedRounds = loadToolIndexedRoundFiles(toolIndexPath, f);
	// Seed the in-memory indexed/mismatched sets from the one index load and
	// keep them current as rows are appended or replaced.
	const indexedRounds = new Set(existingRounds);
	const modelMismatched = new Set(modelMismatchedRounds);
	let sweptTotal = 0;
	let healedTotal = 0;
	if (f.existsSync(roundsDir)) {
		for (const fileName of f.readdirSync(roundsDir)) {
			if (!fileName.endsWith(".json") || fileName.startsWith("index")) continue;
			if (queuedFiles.has(fileName)) continue;
			if (staleFilesForQueued.has(fileName)) continue;
			if (existingRounds.has(fileName) && !modelMismatchedRounds.has(fileName)) {
				// F2 (PR #134 review, option b): a round that already has current-model
				// index rows may still lack the `promptEmbedding` marker (e.g. rows
				// survive while the round file was rewritten). Healing reuses the round's
				// `:response` index-row vector instead of re-embedding: the live
				// combined `concat(prompt, response)` vector is not stored in the index
				// and cannot be reconstructed without an API call. The response-side
				// vector is already the accepted degraded marker on the shared-core
				// short-prompt path (lib/embed-round.ts), and every consumer gates on
				// truthiness and uses it as a real vector. Consequence: healed rounds
				// cannot support topic grouping (src/semblr.ts keys grouping on
				// promptEmbedding), because grouping needs the combined vector. This is
				// accepted for historical rounds — a migration can restore the combined
				// vector if grouping is ever built.
				//
				// F1 (issue #140, decision D1/D2): a round with current-model rows but no
				// `:response` row cannot be healed without an API call. Because rows are
				// the retrieval truth (D1), such a round is a genuine retrieval gap and
				// the sweep re-embeds it from the round file (the source is in hand)
				// rather than leaving it silently marker-less. The round is forced into
				// the reindex path below — removed from the current-model set and added
				// to the mismatched set — so processRound performs a full row replace
				// and writes rows+marker together, restoring convergence.
				const healOutcome = healMissingPromptEmbedding(fileName, roundsDir, responseVectorByRoundFile, f);
				if (healOutcome === "healed") {
					healedTotal++;
					continue;
				}
				if (healOutcome === "already-marked") {
					// Marker present and rows current — genuinely done, no work.
					continue;
				}
				indexedRounds.delete(fileName);
				modelMismatched.add(fileName);
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
		// content-hash filename. Issue #139: the stale-match map is built once for
		// the whole run (buildStaleContentMatchMap above), so this is a map lookup
		// instead of a per-round full-store scan. If found, migrate old index
		// entries before deleting old files so the round remains retrievable even
		// if embedding is unavailable.
		const staleFiles = staleMatchesByTarget.get(roundFile) ?? [];
		for (const staleFile of staleFiles) {
			migrateIndexEntriesFile(indexPath, staleFile, roundFile);
			// Mirror the on-disk migration in the in-memory dedup sets: the stale
			// filename's rows now belong to roundFile.
			if (indexedRounds.delete(staleFile)) indexedRounds.add(roundFile);
			if (modelMismatched.delete(staleFile)) modelMismatched.add(roundFile);
		}

		// Write round file before deleting stale copies. Atomic tmp+rename (F4, PR
		// #134 review), matching the extension's round-file write (src/semblr.ts
		// writeRoundEmbedding): round files are the durable store, so an interrupt
		// mid-write must not truncate a file that has no session JSONL left to
		// recover from.
		const roundPath = path.resolve(roundsDir, roundFile);
		const roundTmpPath = `${roundPath}.tmp.${process.pid}`;
		f.writeFileSync(roundTmpPath, JSON.stringify(round, null, 2));
		f.renameSync(roundTmpPath, roundPath);
		// Issue #139: BM25 is loaded once at run start and mutated in memory here;
		// the file is flushed exactly once after the batch (below), mirroring the
		// flushBm25 seam of indexRecoveredRounds (lib/session-backfill.ts).
		upsertBm25Round(bm25Index, roundFile, roundTextForBm25(round));
		for (const staleFile of staleFiles) {
			deleteBm25Round(bm25Index, staleFile);
		}
		for (const staleFile of staleFiles) {
			f.unlinkSync(path.join(roundsDir, staleFile));
			err.error(`  ♻️  Migrated stale: ${staleFile} → ${roundFile}`);
		}

		// Tool-call fulltext index — independent of embedding, so re-run it even if
		// this round only needs re-embedding due to a model change. Issue #139: the
		// indexed set is loaded once at run start and updated in memory here.
		if (round.toolCalls.length > 0 && !toolIndexedRounds.has(roundFile)) {
			appendToolIndexRows(toolIndexPath, roundsDir, buildToolIndexRows(roundFile, round.toolCalls), {
				fsImpl: f,
			});
			toolIndexedRounds.add(roundFile);
		}

		// Skip embedding if already indexed under the correct hash and current model.
		// Issue #139: indexedRounds/modelMismatched are kept current in memory (seeded
		// from the one index load, updated on migration above and on row writes below),
		// so this no longer re-parses index.csv per round.
		const needsModelReindex = modelMismatched.has(roundFile);
		if (indexedRounds.has(roundFile) && !needsModelReindex) {
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
			//
			// F3 (issue #140, decision D3): a session-scanned round carries no `summary`
			// field (ParsedPiRound has none), so before this fix a model-change reindex
			// of a checkpointed round reproduced prompt+response but not `:summary`,
			// silently dropping a row the live/offline-embed path writes. Recover the
			// summary from the round's semblr_checkpoint tool call exactly as the
			// startup recovery path does (lib/session-backfill.ts
			// extractCheckpointSummary), so reindex reproduces the same rows.
			const summary = round.summary ?? extractCheckpointSummary(round.toolCalls) ?? undefined;
			const entries: VectorIndexEntry[] = [];
			await embedRound(
				{
					fileName: roundFile,
					userPrompt: round.userPrompt,
					responseText: round.responseSequence,
					checkpointSummaryText: summary ? buildCheckpointSummaryText(summary) : null,
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
						// Atomic tmp+rename (F4, PR #134 review): an interrupt during the
						// marker write must not truncate the durable round file.
						const target = path.resolve(roundsDir, fileName);
						const tmp = `${target}.tmp.${process.pid}`;
						f.writeFileSync(tmp, JSON.stringify(round, null, 2));
						f.renameSync(tmp, target);
					},
				},
			);

			if (needsModelReindex) {
				// F3 (issue #140, decision D3): the full replace must never silently
				// drop a row this reindex did not reproduce. A suffix present on disk
				// but absent from `entries` (e.g. a `:summary` row whose source is gone)
				// is reported and preserved rather than traded away. Reproduced suffixes
				// (`:prompt`/`:response`/`:summary` when derived) are still replaced by
				// their fresh rows; only orphan suffixes are kept.
				const orphanEntries = findOrphanIndexEntries(indexPath, roundFile, entries, f);
				for (const orphan of orphanEntries) {
					err.error(`  ⚠️  Preserving non-reproduced index row: ${orphan.filePath}`);
				}
				replaceIndexEntriesForRoundFile(indexPath, roundFile, [...entries, ...orphanEntries]);
				modelMismatched.delete(roundFile);
			} else {
				for (const entry of entries) {
					appendVectorIndexEntry(indexPath, entry.vector, entry.filePath, entry.model, entry.embeddingInputHash);
				}
			}
			// The round now has current-model rows; keep the in-memory dedup sets
			// consistent with the on-disk write above.
			indexedRounds.add(roundFile);

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

	// Issue #139: flush the in-memory BM25 index once for the whole batch,
	// mirroring the flushBm25 seam of indexRecoveredRounds (lib/session-backfill.ts).
	// A crash before this write self-heals on the next run: the load path is
	// non-writing (PR #138) and round files are the source of truth.
	writeBm25Index(bm25IndexPath, bm25Index, f);

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
