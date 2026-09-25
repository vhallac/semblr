/**
 * migrate-prompt-embeddings.ts — Detect stale prompt embeddings and re-embed them
 * (issue #106 re-embed migration).
 *
 * The current convention (post-#106 cleanup, post-#107 F3/F4 provenance + clip):
 * the extension embeds the noise-cleaned prompt, clipped to the configured
 * `embeddingMaxTokens` budget, and stamps the :prompt row with
 * `hashEmbeddingInput(final input)` (4th CSV column). This sweep:
 *
 *   1. Post-#107-F3 rows (round.json carries `promptVecHash`): capture embedded a
 *      verifiable input and stamped it, so a bare-hash mismatch against the current
 *      derivation means the convention (cleanup heuristics, thresholds, clip
 *      budget) changed since the vector was computed → re-embed, restamp plain.
 *   2. Legacy rows (round.json has no `promptVecHash` — captured before #107 F3):
 *      the stored vector was computed by the capture hook over its augmented
 *      prompt, and no plain stamp proves otherwise (the pre-F1 stamp-only branch
 *      wrote hash(clean(raw)) over exactly such augmented-domain vectors, and a
 *      false stamp is indistinguishable from a digest-corrected one). These rows
 *      are re-embedded over the current derivation and stamped `assumed-<hash>`;
 *      a matching assumed- stamp classifies the row current on later runs (zero
 *      API calls — the sweep stays idempotent).
 *
 * Re-embeds the round.json combined vector (`promptEmbedding`) alongside the
 * :prompt row whenever it exists — combined = cleanedPrompt + "\n\n" + response
 * clipped to the configured budget (same convention as capture). :response and
 * :summary rows are not prompt-derived and are left untouched.
 *
 * Known limitations: `assumed-` stamps record a re-derivation from round.json, not
 * the original capture input (unverifiable for legacy rows — #107 F1). A later
 * digest re-embed of a legacy round writes a fresh plain stamp, shedding the
 * marker, so the row re-enters this sweep once (one redundant re-embed; accepted).
 * Response-clipping budget changes are also not detected (out of #106 scope).
 *
 * The re-embed loop runs outside the index lock (live sessions must not stall
 * behind thousands of API calls); the final write takes the shared index lock
 * and merges rows appended by live sessions during the loop (#107 F2). Lock
 * exhaustion fails fast: the index is left untouched — re-run the sweep.
 *
 * Usage:
 *   npx tsx scripts/migrate-prompt-embeddings.ts           # perform migration
 *   npx tsx scripts/migrate-prompt-embeddings.ts --dry-run # preview only
 *   npx tsx scripts/migrate-prompt-embeddings.ts --backup  # backup index.csv first
 *
 * Custom rounds directory:
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/migrate-prompt-embeddings.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { embedText, normalize } from "../lib/embed.ts";
import {
	ASSUMED_STAMP_PREFIX,
	bareEmbeddingInputHash,
	encodeVectorIndexLine,
	indexRoundFileFromPath,
	makeAssumedStamp,
	readIndexLines,
	splitVectorIndexMetadata,
	writeIndexLines,
} from "../lib/index-io.ts";
import {
	buildAgentEndEmbeddingTexts,
	buildPromptEmbeddingInput,
	embeddingMaxTokensToResponseBytes,
} from "../lib/round-capture.ts";
import {
	resolveScriptApiKey,
	resolveScriptConfig,
	resolveScriptIndexPath,
	resolveScriptModelRegistry,
	type ScriptConfigOptions,
	scriptEmbeddingConfig,
} from "../lib/script-config.ts";
import { type AcquireIndexLockDeps, acquireIndexLock } from "../lib/index-storage.ts";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

export interface MigratePromptEmbeddingsOptions extends ScriptConfigOptions {
	indexPath?: string;
	roundsDir?: string;
	dryRun?: boolean;
	backup?: boolean;
	/** Override the embedding fetch impl (tests). */
	fetchImpl?: typeof fetch;
	/** Pre-resolved API key (tests); skips keyring/env resolution. */
	apiKey?: string;
	modelRegistry?: import("../lib/embed.ts").EmbeddingModelRegistry;
	/** Overrides for the final short-lock write's lock protocol (tests). */
	lockDeps?: AcquireIndexLockDeps;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error" | "warn">;
}

interface PromptRowDecision {
	lineIndex: number;
	roundFile: string;
	/** legacy = pre-#107-F3 capture, provenance unverified → re-embed + assumed- stamp. */
	reason: "hash-mismatch" | "legacy" | "current";
	rawLength: number;
	cleanedLength: number;
	/** Hash of the current embedding input, computed during detection. */
	currentHash: string;
}

interface RoundJsonLike {
	userPrompt?: unknown;
	responseSequence?: unknown;
	promptEmbedding?: unknown;
	/** Present (string or null) iff the round was captured with the #107 F3 fix. */
	promptVecHash?: unknown;
}

interface PromptRowGroup {
	lineIndexes: number[];
	decision: PromptRowDecision | null;
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function noiseOptionsFromConfig(config: ReturnType<typeof resolveScriptConfig>) {
	return {
		fenceMaxChars: config.promptNoiseFenceMaxChars,
		jsonMaxChars: config.promptNoiseJsonMaxChars,
		repeatMaxChars: config.promptNoiseRepeatMaxChars,
	};
}

/**
 * Merge rows appended to the index since the detection read (#107 F2): the
 * rewritten snapshot lines stay in position, and every fresh line the
 * detection snapshot does not account for (exact-string multiset difference)
 * is appended in file order. Runtime appends are add-only, so nothing needs
 * removing; a line appended twice by a live session stays twice (faithful).
 */
export function mergeAppendedLines(
	original: readonly string[],
	fresh: readonly string[],
	rewritten: readonly string[],
): string[] {
	const counts = new Map<string, number>();
	for (const line of original) counts.set(line, (counts.get(line) ?? 0) + 1);
	const appended = fresh.filter((line) => {
		const remaining = counts.get(line) ?? 0;
		if (remaining > 0) {
			counts.set(line, remaining - 1);
			return false;
		}
		return true;
	});
	return [...rewritten, ...appended];
}

/** Publish a wholesale index rewrite via tmp + atomic rename (readers never see a partial index). */
function writeIndexLinesAtomic(indexPath: string, entries: string[]): void {
	const tmp = `${indexPath}.tmp.${process.pid}`;
	writeIndexLines(tmp, entries);
	fs.renameSync(tmp, indexPath);
}

/** Atomic round.json update preserving all other fields. */
function updateRoundFile(roundPath: string, mutate: (round: Record<string, unknown>) => void): void {
	const existing = JSON.parse(fs.readFileSync(roundPath, "utf-8")) as Record<string, unknown>;
	mutate(existing);
	fs.writeFileSync(`${roundPath}.tmp.${process.pid}`, JSON.stringify(existing, null, 2));
	fs.renameSync(`${roundPath}.tmp.${process.pid}`, roundPath);
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export async function runPromptEmbeddingsMigration(options: MigratePromptEmbeddingsOptions = {}): Promise<number> {
	const config = resolveScriptConfig(options);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const indexPath = resolveScriptIndexPath(config, roundsDir, options.indexPath);
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const noiseOptions = noiseOptionsFromConfig(config);
	const responseBytes = embeddingMaxTokensToResponseBytes(config.embeddingMaxTokens);

	if (!fs.existsSync(indexPath)) {
		err.error(`❌ Index file not found: ${indexPath}`);
		return 1;
	}

	const lines = readIndexLines(indexPath);
	if (lines.length === 0) {
		out.log("ℹ Index file is empty. Nothing to migrate.");
		return 0;
	}
	// Detection-time snapshot for the final short-lock merge (#107 F2): rows
	// appended between this read and the final write are reconciled there.
	const detectionLines = [...lines];

	// ── Pass 1: detect, per :prompt row, whether the embedding input changed ──
	const promptRows = new Map<string, PromptRowGroup>();
	let promptRowCount = 0;
	let missingRoundFiles = 0;
	let corruptRoundFiles = 0;

	for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
		const line = lines[lineIndex].trim();
		if (!line) continue;
		const firstComma = line.indexOf(",");
		if (firstComma === -1) continue;
		const { filePath } = splitVectorIndexMetadata(line.slice(firstComma + 1));
		if (!filePath.endsWith(":prompt")) continue;
		promptRowCount++;

		const roundFile = path.basename(indexRoundFileFromPath(filePath));
		const roundPath = path.resolve(roundsDir, roundFile);
		if (!fs.existsSync(roundPath)) {
			missingRoundFiles++;
			continue;
		}

		let round: RoundJsonLike;
		try {
			round = JSON.parse(fs.readFileSync(roundPath, "utf-8")) as RoundJsonLike;
		} catch {
			corruptRoundFiles++;
			continue;
		}

		const userPrompt = typeof round.userPrompt === "string" ? round.userPrompt : "";
		const current = buildPromptEmbeddingInput(userPrompt, noiseOptions, config.embeddingMaxTokens);
		const existingHash = splitVectorIndexMetadata(line.slice(firstComma + 1)).embeddingInputHash;
		const bareHash = existingHash !== undefined ? bareEmbeddingInputHash(existingHash) : undefined;
		const hasAssumedStamp = existingHash !== undefined && existingHash.startsWith(ASSUMED_STAMP_PREFIX);
		// Provenance discriminator (#107 F1): round.json gains `promptVecHash` with the
		// #107 F3 capture fix, so its absence marks a legacy capture whose :prompt vector
		// was computed from the hook's augmented prompt. A plain stamp cannot vouch for
		// such a row: the pre-F1 stamp-only branch wrote hash(clean(raw)) over
		// augmented-domain vectors, byte-identical to a digest-corrected stamp.
		const isLegacyCapture = !("promptVecHash" in round);

		let reason: PromptRowDecision["reason"];
		if (isLegacyCapture) {
			// Only a matching assumed- stamp proves a previous sweep already re-embedded
			// this row over the current derivation (idempotent run 2+). Anything else —
			// unstamped, or plainly stamped — is unverified → re-embed and restamp
			// assumed-.
			reason = hasAssumedStamp && bareHash === current.hash ? "current" : "legacy";
		} else {
			// Post-F3 capture: the row's stamp was written over a verifiable input, so
			// the bare-hash comparison decides (missing or mismatched → re-embed).
			reason = bareHash === current.hash ? "current" : "hash-mismatch";
		}

		const existing: PromptRowGroup = promptRows.get(roundFile) ?? { lineIndexes: [], decision: null };
		existing.lineIndexes.push(lineIndex);
		const decision: PromptRowDecision = {
			lineIndex,
			roundFile,
			reason,
			rawLength: userPrompt.length,
			cleanedLength: current.text.length,
			currentHash: current.hash,
		};
		// Escalate: any stale row for a round wins over current.
		const rank: Record<PromptRowDecision["reason"], number> = {
			current: 0,
			legacy: 1,
			"hash-mismatch": 2,
		};
		if (!existing.decision || rank[reason] > rank[existing.decision.reason]) {
			existing.decision = decision;
		}
		promptRows.set(roundFile, existing);
	}

	const staleRounds = [...promptRows.values()].filter(
		(r) => r.decision && (r.decision.reason === "legacy" || r.decision.reason === "hash-mismatch"),
	);
	const legacyStaleRounds = staleRounds.filter((r) => r.decision?.reason === "legacy");
	const currentRounds = [...promptRows.values()].filter((r) => r.decision?.reason === "current");

	out.log(`📊 Index: ${lines.length} rows, ${promptRowCount} :prompt rows`);
	out.log(
		`   stale (re-embed): ${staleRounds.length} (${legacyStaleRounds.length} legacy-unverified) | current: ${currentRounds.length}`,
	);
	if (missingRoundFiles > 0) out.log(`   ⚠ ${missingRoundFiles} :prompt rows without a round file (skipped)`);
	if (corruptRoundFiles > 0) out.log(`   ⚠ ${corruptRoundFiles} unreadable round files (skipped)`);

	if (options.dryRun) {
		out.log("\n🚫 Dry-run mode: no changes written.");
		for (const { decision } of staleRounds) {
			if (!decision) continue;
			out.log(
				`   [${decision.reason}] ${decision.roundFile} prompt ${decision.rawLength} → ${decision.cleanedLength} chars`,
			);
		}
		out.log("\n   Run without --dry-run to apply.");
		return 0;
	}

	if (staleRounds.length === 0) {
		out.log("\n✅ All :prompt rows are current. Nothing to migrate.");
		return 0;
	}

	// ── Pass 2: apply ──
	// Backup before any write (index.csv is rewritten wholesale).
	if (options.backup) {
		const backupPath = `${indexPath}.bak.${Date.now()}`;
		fs.copyFileSync(indexPath, backupPath);
		out.log(`\n💾 Backup saved to: ${backupPath}`);
	}

	const apiKey = await resolveScriptApiKey(config, options);
	const modelRegistry = resolveScriptModelRegistry(config, {
		apiKey: apiKey ?? undefined,
		modelRegistry: options.modelRegistry,
	});
	const embeddingConfig = scriptEmbeddingConfig(config);

	if (!apiKey) {
		err.error("❌ Embedding API key required to re-embed stale rows (no key resolved).");
		err.error("   Set OPENROUTER_API_KEY (or the provider key) and re-run, or use --dry-run to preview.");
		return 1;
	}

	// Stale rows: re-embed via the configured provider/model, update round.json
	// combined vector, and rewrite the :prompt line(s).
	let reembedded = 0;
	let combinedReembedded = 0;
	let assumedStamped = 0;
	for (const { lineIndexes, decision } of staleRounds) {
		if (!decision) continue;
		const roundPath = path.resolve(roundsDir, decision.roundFile);
		const round = JSON.parse(fs.readFileSync(roundPath, "utf-8")) as RoundJsonLike;
		const userPrompt = typeof round.userPrompt === "string" ? round.userPrompt : "";
		const current = buildPromptEmbeddingInput(userPrompt, noiseOptions, config.embeddingMaxTokens);
		// Guard against the round file mutating between detection and apply: if the
		// recomputed hash no longer matches the detection stamp, the row is re-detected
		// on the next run — skip it here rather than write a wrong stamp.
		if (current.hash !== decision.currentHash) continue;
		const responseSequence = typeof round.responseSequence === "string" ? round.responseSequence : "";
		const hasCombined = Array.isArray(round.promptEmbedding) && round.promptEmbedding.length > 0;

		const promptVec = await embedText(current.text, apiKey ?? "", {
			fetchImpl: options.fetchImpl,
			config: embeddingConfig,
			modelRegistry,
		});
		let combinedVec: number[] | null = null;
		if (hasCombined) {
			combinedVec = await embedText(
				buildAgentEndEmbeddingTexts(current.text, responseSequence, responseBytes).combinedText,
				apiKey ?? "",
				{
					fetchImpl: options.fetchImpl,
					config: embeddingConfig,
					modelRegistry,
				},
			);
		}

		if (combinedVec !== null) {
			updateRoundFile(roundPath, (existing) => {
				existing.promptEmbedding = combinedVec;
			});
			combinedReembedded++;
		}

		// Legacy rounds get an assumed- provenance stamp: the fresh vector is a
		// re-derivation from round.json, not the unverifiable original capture input.
		// Post-F3 rounds get a plain stamp: capture used the same derivation.
		const stamp = decision.reason === "legacy" ? makeAssumedStamp(current.hash) : current.hash;
		for (const lineIndex of lineIndexes) {
			const { filePath } = splitVectorIndexMetadata(lines[lineIndex].slice(lines[lineIndex].indexOf(",") + 1));
			lines[lineIndex] = encodeVectorIndexLine(normalize(promptVec), filePath, config.embeddingModel, stamp);
		}
		reembedded++;
		if (decision.reason === "legacy") assumedStamped++;
		out.log(`  ✅ [${reembedded}/${staleRounds.length}] ${decision.roundFile} (${decision.reason})`);
	}

	// ── Final write: short-lock merge (#107 F2) ──
	// The re-embed loop above ran OUTSIDE the lock (thousands of API calls; live
	// sessions must not stall behind it). Only this phase takes the shared
	// index lock: re-read the index, merge the lines live sessions appended
	// during the re-embed window, and publish once via tmp + atomic rename. A
	// session appending mid-merge blocks on the same lock and lands after the
	// rename. On lock exhaustion this fails fast — a wholesale rewrite must
	// never fall back to an unsynchronized write (replacing the whole file
	// could drop concurrent rows; the runtime's single-line append fallback is
	// the pre-existing accepted behavior, and this narrows its exposure window
	// from minutes of re-embedding to the seconds-long merge phase).
	const lock = acquireIndexLock(indexPath, options.lockDeps);
	if (lock === null) {
		err.error(`❌ Could not acquire the index lock (${indexPath}.lock) after all retries — the index was NOT written.`);
		err.error(
			"   A live session is likely holding it. This run's re-embed results were discarded; re-run the migration (already-restamped rows cost zero API calls).",
		);
		return 1;
	}
	let mergedRowCount = lines.length;
	try {
		const freshLines = readIndexLines(indexPath);
		const mergedLines = mergeAppendedLines(detectionLines, freshLines, lines);
		writeIndexLinesAtomic(indexPath, mergedLines);
		mergedRowCount = mergedLines.length;
	} finally {
		lock.release();
	}

	out.log(
		`\n✅ Done. ${reembedded} rounds re-embedded (${combinedReembedded} combined vectors), ${assumedStamped} legacy rows re-stamped with assumed- provenance.`,
	);
	out.log(`   Index: ${indexPath} (${mergedRowCount} rows)`);
	return 0;
}

// ─────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const dryRun = process.argv.includes("--dry-run");
	const backup = process.argv.includes("--backup");
	const exitCode = await runPromptEmbeddingsMigration({ dryRun, backup });
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((err) => {
		console.error("❌ Error:", err);
		process.exit(1);
	});
}
