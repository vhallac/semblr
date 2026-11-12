import type * as fs from "node:fs";
import * as path from "node:path";
import { FORCING_INDEX_SUFFIXES, forcingReproducibleSuffixes } from "./embed-round.ts";
import { loadShardVectorIndex, type VectorIndexEntry } from "./index-io.ts";
import { ownShardPath } from "./index-shards.ts";
import type { PromptNoiseOptions } from "./round-capture.ts";
import { readRoundJson } from "./round-io.ts";
import { saveScanCutoff } from "./scan-register.ts";
import type { SemblrConfig } from "./semblr-config.ts";
import type { IndexRecoveredRoundsDeps } from "./session-backfill.ts";
import {
	backfillMissingRounds,
	buildBackfillCandidates,
	buildCurrentModelRowPredicate,
	embedRecoveredRounds,
	indexRecoveredRounds,
	isBackfillStartReason,
	planStartupPendingScan,
	startupEmbedStatusMessage,
} from "./session-backfill.ts";

/**
 * Seam extracted from the session_start handler (R1.F1, PR !160 review): the
 * whole startup backfill block — previous-session recovery, the issue #136
 * pending scan on the skippedComplete/recovered path, and the inline embed /
 * defer pass — driven through injected deps so the handler's wiring is
 * testable with a fake pi/ctx and a real temp rounds dir.
 */
export interface StartupBackfillDeps {
	/** Start reason carried by the session_start event. */
	reason: string;
	/** Filesystem seam for the scan and the marker write. */
	fs: Pick<typeof fs, "readdirSync" | "readFileSync" | "writeFileSync" | "renameSync" | "statSync">;
	roundsDir: string;
	/** State dir backing the scan register (SEMBLR_DIR). */
	semblrDir: string;
	config: Pick<SemblrConfig, "hostId" | "embeddingModel" | "embeddingMaxTokens">;
	/** Live-parity response budget for recovered embeddings (EMBEDDING_RESPONSE_MAX_BYTES). */
	embeddingResponseBytes: number;
	promptNoiseOptions: PromptNoiseOptions;
	/** Pre-built recovery index deps (bm25 + tool-index upserts for recovered rounds). */
	recoveryIndexDeps: IndexRecoveredRoundsDeps;
	/** Live agent_end append surface — recovery rows land in the host's own shard. */
	appendToIndex: (filePath: string, vector: number[], model: string, embeddingInputHash?: string) => void;
	ui: { setStatus(source: string, message: string): void };
	getSessionFile: () => string | undefined;
	getApiKey: () => Promise<string | null>;
	embed: (text: string, apiKey: string) => Promise<number[]>;
	/** pi-supplied previous session file ("new"/"resume"/"fork"). */
	previousSessionFile?: string;
}

export interface StartupBackfillOutcome {
	/**
	 * Merged index entries captured by the pending scan, reusable for the
	 * final status line; null when the scan never ran (no session file,
	 * non-backfill reason, or an error) or when the inline embed pass
	 * appended rows (captured entries stale — the caller re-reads).
	 */
	indexEntries: VectorIndexEntry[] | null;
}

export async function runStartupBackfill(deps: StartupBackfillDeps): Promise<StartupBackfillOutcome> {
	const { fs, roundsDir, semblrDir, config, ui } = deps;
	// Issue #130: recover rounds lost to process death in previous sessions.
	// F2 (PR !131): scan ALL prior session files in the session dir (newest
	// first, per-file tail-completeness early exit) — a source deferred as live
	// on one start is retried automatically on the next. F5 (PR !131): the
	// gate covers every session_start reason, not just startup/resume — pi
	// supplies previousSessionFile for "new"/"resume"/"fork" (pinned first
	// when the dir scan misses it); "reload" carries none but still gets the
	// dir scan, since a source live at startup may be closed by the time
	// /reload restarts the extension. Sessions outside the current cwd's dir
	// are not ours.
	// R1.F2 (PR !160 review): the merged index is loaded at most once per
	// startup. The pending scan's load is reused for the final status line
	// unless the inline embed pass appended rows (then it is stale and the
	// status re-reads); startups without a pending scan still read once.
	if (!isBackfillStartReason(deps.reason)) {
		return { indexEntries: null };
	}
	const sessionFile = deps.getSessionFile();
	if (!sessionFile) {
		return { indexEntries: null };
	}
	const mergedIndexForStatus: { value: VectorIndexEntry[] | null } = { value: null };
	try {
		const candidates = buildBackfillCandidates(
			path.dirname(sessionFile),
			sessionFile,
			deps.previousSessionFile,
			fs,
			{ stateDir: semblrDir },
		);
		const backfill = backfillMissingRounds(candidates, roundsDir);
		// Issue #133 unit-003 / issue #147: advance the scan register whenever
		// the scan conclusively examined at least one source file — skipped
		// complete, live-deferred, or recovered with writes re-verified on
		// disk. Recovery WORK happening no longer freezes the cutoff (a
		// persistently mis-hashing round would otherwise keep recovery
		// non-convergent and the register at null forever).
		if (backfill.examinedMtime !== undefined) {
			saveScanCutoff(semblrDir, path.dirname(sessionFile), backfill.examinedMtime);
		}
		if (backfill.deferredLive) {
			ui.setStatus(
				"semblr",
				`\u{1f9e0} backfill deferred: ${backfill.deferredLive} previous session(s) look live (F2/F3)`,
			);
		}
		if (backfill.recoveredFiles.length > 0) {
			// F3 (PR !131): index recovered rounds exactly like the live agent_end
			// post-write — bm25 upsert AND tool-index rows — so a recovered round
			// is reachable from every search surface, not just bm25. Each round is
			// guarded independently (best-effort, never the round files).
			const indexReport = indexRecoveredRounds(
				backfill.recoveredFiles,
				// Memory-only upsert; the whole batch commits with a single index
				// write (issue #137: per-round rewrites dominated startup recovery).
				deps.recoveryIndexDeps,
			);
			for (const message of indexReport.errors) {
				ui.setStatus("semblr", `\u{1f9e0} backfill index error: ${message}`);
			}
			ui.setStatus(
				"semblr",
				`\u{1f9e0} backfill: recovered ${backfill.recoveredFiles.length} round(s) from previous session`,
			);
		}
		// Issue #136: the pending scan runs on skippedComplete AND after normal
		// recovery — a session that exited complete (early exit) may still have
		// rounds stranded by a crash between round write and embedding. Only
		// the trigger widens; pending semantics, defer threshold, and the pinned
		// defer message are unchanged (decision 011).
		if (backfill.recoveredFiles.length > 0 || backfill.skippedComplete) {
			// Issue #133: defer the embedding burst when a large backlog of
			// recovered rounds is unembedded — inline startup embedding beyond
			// the threshold can block startup for minutes; `just index` sweeps
			// it instead. BM25 + tool indexing above stays synchronous.
			// Issue #140 (D1): the pending count keys on current-model index
			// rows, not the promptEmbedding marker. A recovered round with
			// current-model rows but no marker (rows written without one, e.g.
			// after a crash) is not pending — counting it inflated the startup
			// count and made startup disagree with the `just index` sweep.
			// Issue #149 (D-010): the pending predicate counts rows across ALL
			// shards — a round a foreign host already embedded with the current
			// model must not be re-embedded locally on every startup.
			const indexEntries = loadShardVectorIndex(roundsDir);
			mergedIndexForStatus.value = indexEntries;
			// Issue #149 (D-006 scope): the live label guard consults only the
			// host's own shard — the post-adoption write surface.
			// Issue #154 (PR !151 R1.F4): the own-shard startup read must tolerate
			// corrupt rows like merged reads do — skip with a warning instead of
			// crashing startup; the sweep's strict reload catches corruption at
			// write time.
			const ownShardEntries = loadShardVectorIndex(roundsDir, {
				shards: [ownShardPath(roundsDir, config.hostId)],
			});
			// F2 (PR !141 review): the pending count and the `just index` sweep must
			// classify a recovered round alike. The sweep derives the *forcing*
			// suffix set per round from the round file (a short prompt's `:prompt`
			// row is non-reproducible; `:summary` never forces a reindex).
			// R1.F6 (PR !160 review): one parsed-round cache backs both the suffix
			// derivation (via the coverage predicate) and the pending-scan reader,
			// so a row-failing candidate is read once per startup — the scan
			// callback receives the already-parsed round instead of re-reading it.
			const roundsByFile = new Map<string, Record<string, unknown> | null>();
			const readRoundMemo = (fileName: string): Record<string, unknown> | null => {
				if (roundsByFile.has(fileName)) return roundsByFile.get(fileName) ?? null;
				const round = readRoundJson(roundsDir, fileName);
				roundsByFile.set(fileName, round);
				return round;
			};
			const forcingSuffixesFor = (fileName: string): ReadonlySet<string> => {
				const round = readRoundMemo(fileName);
				return round === null
					? FORCING_INDEX_SUFFIXES
					: forcingReproducibleSuffixes(String(round.userPrompt ?? ""), Boolean(round.summary));
			};
			const roundFileNames = fs
				.readdirSync(roundsDir)
				.filter((fn) => fn.endsWith(".json") && !fn.startsWith("index"));
			// Issue #136: the scan yields the pending file list, so the inline pass
			// runs over exactly the scanned candidates (caller change only).
			// R1.F2 (PR !160 review): build the current-model rows predicate once
			// — the scan and the inline pass share the same instance, so the
			// entry map is not rebuilt and the derived-suffix reads are not
			// re-done for the embed pass.
			const pendingRowsPredicate = buildCurrentModelRowPredicate(
				indexEntries,
				config.embeddingModel,
				forcingSuffixesFor,
			);
			const embedPlan = planStartupPendingScan(
				roundFileNames,
				pendingRowsPredicate,
				(fileName) => readRoundMemo(fileName) as { promptEmbedding?: unknown } | null,
			);
			if (embedPlan.mode === "defer") {
				ui.setStatus("semblr", startupEmbedStatusMessage(embedPlan));
			} else if (embedPlan.pendingFiles.length > 0) {
				try {
					const embedKey = await deps.getApiKey();
					if (embedKey) {
						// Label guard backed by a single index load: the index is read
						// once before embedding, kept as an in-memory Set, and updated on
						// every append. Previously hasIndexRow re-read the full index
						// file per recovered round (O(n × index size) — minutes of
						// blocking startup work with a large index).
						const indexedLabels = new Set(ownShardEntries.map((entry) => entry.filePath));
						const embedResult = await embedRecoveredRounds(
							embedPlan.pendingFiles,
							roundsDir,
							{
								embed: (text) => deps.embed(text, embedKey),
								appendIndexRow: (label, vec, hash) => {
									deps.appendToIndex(label, vec, config.embeddingModel, hash);
									indexedLabels.add(label);
								},
								// F4+F6 (PR !131): prompt derivation, clipping, and row labels now
								// come from the shared core, so recovered rows are in the same domain
								// as agent_end by construction; only the config is passed here.
								// F5 (PR !131): label-guard so a re-run after a crash between the
								// appends and the embedding write does not duplicate index rows.
								hasIndexRow: (label) => indexedLabels.has(label),
								writeRoundEmbedding: (fileName, vec) => {
									const p = `${roundsDir}/${fileName}`;
									const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
									existing.promptEmbedding = vec;
									fs.writeFileSync(p + ".tmp." + process.pid, JSON.stringify(existing, null, 2));
									fs.renameSync(p + ".tmp." + process.pid, p);
								},
							},
							// F3 (PR !131): live-parity response budget — recovered embeddings are
							// clipped with the same configured budget as agent_end, not the default.
							{
								maxResponseBytes: deps.embeddingResponseBytes,
								promptNoiseOptions: deps.promptNoiseOptions,
								promptMaxTokens: config.embeddingMaxTokens,
								// Issue #140 (D1): reuse the same current-model rows predicate as
								// planStartupPendingScan above, so a round counted non-pending is not
								// re-embedded by this pass (rows authoritative, marker derived).
								hasCurrentModelRows: pendingRowsPredicate,
							},
						);
						// The inline pass may have appended index rows — the hoisted
						// load is stale; the final status re-reads.
						mergedIndexForStatus.value = null;
						if (embedResult.embedded.length > 0) {
							ui.setStatus(
								"semblr",
								`\u{1f9e0} backfill: embedded ${embedResult.embedded.length} pending round(s)`,
							);
						}
						for (const message of embedResult.errors) {
							ui.setStatus("semblr", `\u{1f9e0} backfill embed error: ${message}`);
						}
					}
				} catch (err) {
					ui.setStatus("semblr", `\u{1f9e0} backfill embed failed: ${(err as Error).message}`);
				}
			}
		}
	} catch (err) {
		ui.setStatus("semblr", `🧠 backfill failed: ${(err as Error).message}`);
	}
	return { indexEntries: mergedIndexForStatus.value };
}
