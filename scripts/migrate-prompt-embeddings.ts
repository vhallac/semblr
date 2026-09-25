/**
 * migrate-prompt-embeddings.ts — Detect stale prompt embeddings and re-embed them
 * (issue #106 re-embed migration).
 *
 * Before the #106 embedding-input cleanup, :prompt index rows were embedded over
 * the raw user prompt. The extension now embeds the noise-cleaned prompt, clipped
 * to the configured `embeddingMaxTokens` budget (clip restored post-cleanup in
 * #107 F4), and stamps the row with `hashEmbeddingInput(final input)` (4th CSV
 * column). This sweep:
 *
 *   1. Stamped rows: recompute the current embedding input; stamp mismatch means
 *      the convention (cleanup heuristics, thresholds, clip budget) changed since
 *      capture → re-embed.
 *   2. Legacy rows (no stamp): if the current convention (cleanup + clip)
 *      transforms the stored raw prompt, the old raw-based vector is stale →
 *      re-embed. If the convention is a no-op for that prompt, the old vector is
 *      still valid → stamp the row without an embedding API call.
 *
 * Re-embeds the round.json combined vector (`promptEmbedding`) alongside the
 * :prompt row whenever it exists — combined = cleanedPrompt + "\n\n" + response
 * clipped to the configured budget (same convention as capture). :response and
 * :summary rows are not prompt-derived and are left untouched.
 *
 * Known limitation: legacy digest rows embedded over `raw.slice(0, embeddingMaxTokens)`
 * carry no stamp. When cleanup is a no-op and the prompt fits the budget, the old
 * vector matches the current convention input → stamp-only. When cleanup is a no-op
 * and the prompt exceeds the budget, the current clipped input differs from the
 * stored raw prompt → the row is detected stale and re-embedded (over the same
 * prefix bytes when the budget is unchanged), gaining a correct stamp.
 * Response-clipping budget changes are also not detected (out of #106 scope).
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
	encodeVectorIndexLine,
	indexRoundFileFromPath,
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
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error" | "warn">;
}

interface PromptRowDecision {
	lineIndex: number;
	roundFile: string;
	reason: "hash-mismatch" | "legacy-clean" | "current" | "stamp-only";
	rawLength: number;
	cleanedLength: number;
	/** Hash of the current embedding input, computed during detection. */
	currentHash: string;
}

interface RoundJsonLike {
	userPrompt?: unknown;
	responseSequence?: unknown;
	promptEmbedding?: unknown;
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

		let reason: PromptRowDecision["reason"];
		if (existingHash !== undefined) {
			reason = existingHash === current.hash ? "current" : "hash-mismatch";
		} else {
			// Legacy row (pre-#106): embedded over the raw prompt. Stale iff the current
			// cleanup transforms the stored prompt; otherwise stamp it without re-embedding.
			reason = current.text !== userPrompt ? "legacy-clean" : "stamp-only";
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
		// Escalate: any stale row for a round wins over stamp-only/current.
		const rank: Record<PromptRowDecision["reason"], number> = {
			current: 0,
			"stamp-only": 1,
			"legacy-clean": 2,
			"hash-mismatch": 3,
		};
		if (!existing.decision || rank[reason] > rank[existing.decision.reason]) {
			existing.decision = decision;
		}
		promptRows.set(roundFile, existing);
	}

	const staleRounds = [...promptRows.values()].filter(
		(r) => r.decision && (r.decision.reason === "legacy-clean" || r.decision.reason === "hash-mismatch"),
	);
	const stampOnlyRounds = [...promptRows.values()].filter((r) => r.decision?.reason === "stamp-only");
	const currentRounds = [...promptRows.values()].filter((r) => r.decision?.reason === "current");

	out.log(`📊 Index: ${lines.length} rows, ${promptRowCount} :prompt rows`);
	out.log(
		`   stale (re-embed): ${staleRounds.length} | stamp-only: ${stampOnlyRounds.length} | current: ${currentRounds.length}`,
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

	const needsEmbedding = staleRounds.length > 0;
	if (!needsEmbedding && stampOnlyRounds.length === 0) {
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

	if (needsEmbedding && !apiKey) {
		err.error("❌ Embedding API key required to re-embed stale rows (no key resolved).");
		err.error("   Set OPENROUTER_API_KEY (or the provider key) and re-run, or use --dry-run to preview.");
		return 1;
	}

	// Stamp-only rows: rewrite the line with the current input hash, no API call.
	// The vector is untouched (the old raw-based input equals the current cleaned input).
	// The CSV schema is positional (vector,filePath,model[,hash]), so a stamped row on a
	// line without a model column gets the current model filled in (same convention as
	// migrate-model-column.ts for unlabeled rows).
	for (const { lineIndexes, decision } of stampOnlyRounds) {
		if (!decision) continue;
		for (const lineIndex of lineIndexes) {
			const line = lines[lineIndex];
			const firstComma = line.indexOf(",");
			const { filePath, model } = splitVectorIndexMetadata(line.slice(firstComma + 1));
			const vector = JSON.parse(Buffer.from(line.slice(0, firstComma), "base64url").toString("utf-8")) as number[];
			lines[lineIndex] = encodeVectorIndexLine(
				vector,
				filePath,
				model ?? config.embeddingModel,
				decision.currentHash,
			);
		}
	}

	// Stale rows: re-embed via the configured provider/model, update round.json
	// combined vector, and rewrite the :prompt line(s).
	let reembedded = 0;
	let combinedReembedded = 0;
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

		for (const lineIndex of lineIndexes) {
			const { filePath } = splitVectorIndexMetadata(lines[lineIndex].slice(lines[lineIndex].indexOf(",") + 1));
			lines[lineIndex] = encodeVectorIndexLine(normalize(promptVec), filePath, config.embeddingModel, current.hash);
		}
		reembedded++;
		out.log(`  ✅ [${reembedded}/${staleRounds.length}] ${decision.roundFile} (${decision.reason})`);
	}

	writeIndexLines(indexPath, lines);

	out.log(
		`\n✅ Done. ${reembedded} rounds re-embedded (${combinedReembedded} combined vectors), ${stampOnlyRounds.length} stamped without embedding.`,
	);
	out.log(`   Index: ${indexPath} (${lines.length} rows)`);
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
