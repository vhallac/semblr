/**
 * digest-session.ts — Parse a pi session JSONL into rounds, embed them through
 * the configured Semblr provider/model, and build a vector index for semblr.
 *
 * Usage:
 *   npx tsx scripts/digest-session.ts <session-file>
 *
 * Output (default, override with Semblr settings or SEMBLR_ROUNDS_DIR):
 *   ~/.pi/agent/semblr/rounds/<id>.json  — each round as a file
 *   ~/.pi/agent/semblr/rounds/index.<hostname> — this host's vector index shard
 *     (base64(vector),filepath,model); reads span every host's shard (issue #149)
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { type AtomicWriteFs, atomicWriteFileSync } from "../lib/atomic-write.ts";
import {
	bm25IndexPathForRoundsDir,
	deleteBm25Round,
	loadBm25Index,
	roundTextForBm25,
	upsertBm25Round,
	writeBm25Index,
} from "../lib/bm25-index.ts";
import { type EmbeddingModelRegistry, embedText, normalize } from "../lib/embed.ts";
import { findStaleContentMatches as findStaleContentMatchesInDir, indexRoundFileFromPath } from "../lib/index-io.ts";
import { parsePiSessionJsonl, reconstructPiSessionRounds } from "../lib/pi-session.ts";
import { buildPromptEmbeddingInput } from "../lib/round-capture.ts";
import {
	resolveScriptApiKey,
	resolveScriptConfig,
	resolveScriptIndexContext,
	resolveScriptModelRegistry,
	type ScriptConfigOptions,
	type ScriptIndexFs,
	scriptEmbeddingConfig,
} from "../lib/script-config.ts";

// ─────────────────────────────────────────────
// Parse session into rounds
// ─────────────────────────────────────────────

export function parseSession(filePath: string) {
	return reconstructPiSessionRounds(parsePiSessionJsonl(fs.readFileSync(filePath, "utf-8")));
}

// ─────────────────────────────────────────────
// Embedding via configured provider/model (wraps lib/embed for backward compat)
// ─────────────────────────────────────────────

export { normalize };

export interface EmbedOptions extends ScriptConfigOptions {
	apiKey?: string;
	fetchImpl?: typeof fetch;
	modelRegistry?: EmbeddingModelRegistry;
}

export async function embed(text: string, options: EmbedOptions = {}): Promise<number[]> {
	const config = resolveScriptConfig(options);
	const apiKey = await resolveScriptApiKey(config, options);
	if (!apiKey) {
		throw new Error("OPENROUTER_API_KEY environment variable required");
	}
	const modelRegistry = resolveScriptModelRegistry(config, options);
	const { embedText } = await import("../lib/embed.ts");
	return embedText(text, apiKey, {
		fetchImpl: options.fetchImpl,
		config: scriptEmbeddingConfig(config),
		modelRegistry,
	});
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface DigestSessionOptions extends ScriptConfigOptions {
	sessionFile?: string;
	roundsDir?: string;
	indexPath?: string;
	apiKey?: string;
	fetchImpl?: typeof fetch;
	modelRegistry?: EmbeddingModelRegistry;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: AtomicWriteFs;
	/** fs seam for the shard reads/writes and the shard lock (tests). */
	indexFsImpl?: ScriptIndexFs;
}

export async function runDigestSession(options: DigestSessionOptions = {}): Promise<number> {
	const sessionFile = options.sessionFile;
	const config = resolveScriptConfig(options);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	// Issue #149: the index is a set of per-host shards — reads see every
	// shard, writes land in the host's own shard (`index.<hostname>`).
	const index = resolveScriptIndexContext(config, roundsDir, options.indexPath, { fsImpl: options.indexFsImpl });
	const bm25IndexPath = bm25IndexPathForRoundsDir(roundsDir);
	const modelRegistry = resolveScriptModelRegistry(config, options);
	const embeddingConfig = scriptEmbeddingConfig(config);
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;

	if (!sessionFile) {
		err.error("Usage: npx tsx scripts/digest-session.ts <session-file>");
		return 1;
	}

	out.log(`📂 Session: ${sessionFile}`);
	const rounds = parseSession(sessionFile);
	out.log(`📊 Parsed ${rounds.length} rounds`);

	// Ensure rounds directory
	fs.mkdirSync(roundsDir, { recursive: true });

	let embedded = 0;
	let skipped = 0;

	for (const { round, roundFile } of rounds) {
		// Check for stale files whose stored full hash material belongs under this
		// content-hash filename. If found, migrate old index entries before deleting
		// old files so the round remains retrievable even if embedding is unavailable.
		const staleFiles = findStaleContentMatchesInDir(roundsDir, roundFile);
		const removableStaleFiles: string[] = [];
		for (const staleFile of staleFiles) {
			// Claim a legacy `index.csv` before rewriting rows, so its rows are
			// part of the own shard (and not stranded in the legacy file).
			index.adopt();
			// The migration rewrites the host's own shard atomically and under
			// the shard's append lock (a crash leaves the shard byte-identical;
			// a concurrent append blocks and lands after the publish).
			index.migrateOwnRows(staleFile, roundFile);
			// A foreign shard may still hold rows for the stale file: this host
			// rewrites only its own shard, so deleting the file would leave
			// those rows dangling. Keep the file and report it instead.
			if (index.foreignShardsReferencing(staleFile).length > 0) {
				out.log(`  ⚠️  Stale file kept (referenced by another host's shard): ${staleFile}`);
				continue;
			}
			removableStaleFiles.push(staleFile);
		}

		// Always write the round file (idempotent) before deleting stale copies.
		// Atomic (decision-015): tmp write + rename, so a failure leaves the
		// existing round file byte-identical.
		atomicWriteFileSync(path.resolve(roundsDir, roundFile), JSON.stringify(round, null, 2), options.fsImpl);
		const bm25Index = loadBm25Index(bm25IndexPath);
		upsertBm25Round(bm25Index, roundFile, roundTextForBm25(round));
		for (const staleFile of removableStaleFiles) {
			deleteBm25Round(bm25Index, staleFile);
		}
		writeBm25Index(bm25IndexPath, bm25Index);
		for (const staleFile of removableStaleFiles) {
			fs.unlinkSync(path.join(roundsDir, staleFile));
			out.log(`  ♻️  Migrated stale: ${staleFile} → ${roundFile}`);
		}

		// Refresh existing set after potential deletions — the merged shard
		// view, so a round another host embedded is not embedded twice.
		const freshExisting = new Set(
			index.readMergedEntries().map((e) => path.basename(indexRoundFileFromPath(e.filePath))),
		);
		if (freshExisting.has(roundFile)) {
			skipped++;
			continue;
		}

		// Embed prompt — noise-cleaned under the current #106 cleanup heuristics, clipped to
		// the configured embeddingMaxTokens (issue #107 F4), and stamped with the embedding-input
		// hash so `just migrate` can detect later convention changes. Matches the extension capture path.
		out.log(`  🔄 Embedding round ${round.turnIndex + 1}/${rounds.length}...`);
		const apiKey = await resolveScriptApiKey(config, options);
		if (!apiKey) {
			throw new Error("OPENROUTER_API_KEY environment variable required");
		}
		const { text: promptInput, hash: promptInputHash } = buildPromptEmbeddingInput(
			round.userPrompt,
			{
				fenceMaxChars: config.promptNoiseFenceMaxChars,
				jsonMaxChars: config.promptNoiseJsonMaxChars,
				repeatMaxChars: config.promptNoiseRepeatMaxChars,
			},
			config.embeddingMaxTokens,
		);
		const promptVector = await embedText(promptInput, apiKey, {
			fetchImpl: options.fetchImpl,
			config: embeddingConfig,
			modelRegistry,
		});
		index.appendRow(`${roundFile}:prompt`, normalize(promptVector), config.embeddingModel, promptInputHash);

		// Embed response
		const respVector = await embedText(round.responseSequence.slice(0, config.embeddingMaxTokens), apiKey, {
			fetchImpl: options.fetchImpl,
			config: embeddingConfig,
			modelRegistry,
		});
		index.appendRow(`${roundFile}:response`, normalize(respVector), config.embeddingModel);

		embedded++;
	}

	out.log(`\n✅ Done. ${embedded} new rounds embedded, ${skipped} already in index.`);
	out.log(`   Index: ${index.ownShardPath} (${index.readMergedEntries().length} vectors)`);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const exitCode = await runDigestSession({ sessionFile: process.argv[2] });
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((err) => {
		console.error("❌ Error:", err);
		process.exit(1);
	});
}
