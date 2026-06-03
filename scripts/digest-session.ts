/**
 * digest-session.ts — Parse a pi session JSONL into rounds, embed them via OpenRouter,
 * and build a vector index for semblr.
 *
 * Usage:
 *   npx tsx scripts/digest-session.ts <session-file>
 *
 * Output (default, override with SEMBLR_ROUNDS_DIR):
 *   ~/.pi/agent/semblr/rounds/<id>.json  — each round as a file
 *   ~/.pi/agent/semblr/rounds/index.csv  — vector index (base64(vector),filepath)
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { embedText, normalize } from "../lib/embed.ts";
import { computeContentHash } from "../lib/hash.ts";
import {
	appendVectorIndexEntry,
	findStaleContentMatches as findStaleContentMatchesInDir,
	loadVectorIndex,
	migrateIndexEntries as migrateIndexEntriesFile,
} from "../lib/index-io.ts";
import { type ParsedPiRound, parsePiSessionJsonl } from "../lib/pi-session.ts";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

type Round = ParsedPiRound;

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");
const MAX_PROMPT_CHARS = 8000;
const MAX_RESPONSE_CHARS = 8000;

// ─────────────────────────────────────────────
// Parse session into rounds
// ─────────────────────────────────────────────

export function parseSession(filePath: string): Round[] {
	return parsePiSessionJsonl(fs.readFileSync(filePath, "utf-8"));
}

// ─────────────────────────────────────────────
// Embedding via OpenRouter (wraps lib/embed for backward compat)
// ─────────────────────────────────────────────

export { normalize };

export interface EmbedOptions {
	apiKey?: string;
	fetchImpl?: typeof fetch;
}

export async function embed(text: string, options: EmbedOptions = {}): Promise<number[]> {
	const apiKey = options.apiKey ?? process.env.OPENROUTER_API_KEY;
	if (!apiKey) {
		throw new Error("OPENROUTER_API_KEY environment variable required");
	}
	const { embedText } = await import("../lib/embed.ts");
	return embedText(text, apiKey, { fetchImpl: options.fetchImpl });
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface DigestSessionOptions {
	sessionFile?: string;
	roundsDir?: string;
	indexPath?: string;
	apiKey?: string;
	fetchImpl?: typeof fetch;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
}

export async function runDigestSession(options: DigestSessionOptions = {}): Promise<number> {
	const sessionFile = options.sessionFile;
	const roundsDir = options.roundsDir ?? ROUNDS_DIR;
	const indexPath = options.indexPath ?? path.resolve(roundsDir, "index.csv");
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

	for (const round of rounds) {
		const roundFile = `${computeContentHash(round.userPrompt, round.responseSequence, round.toolCalls)}.json`;

		// Check for stale files whose stored full hash material belongs under this
		// content-hash filename. If found, migrate old index entries before deleting
		// old files so the round remains retrievable even if embedding is unavailable.
		const staleFiles = findStaleContentMatchesInDir(roundsDir, roundFile);
		for (const staleFile of staleFiles) {
			migrateIndexEntriesFile(indexPath, staleFile, roundFile);
		}

		// Always write the round file (idempotent) before deleting stale copies.
		fs.writeFileSync(path.resolve(roundsDir, roundFile), JSON.stringify(round, null, 2));
		for (const staleFile of staleFiles) {
			fs.unlinkSync(path.join(roundsDir, staleFile));
			out.log(`  ♻️  Migrated stale: ${staleFile} → ${roundFile}`);
		}

		// Refresh existing set after potential deletions
		const freshExisting = new Set(
			loadVectorIndex(indexPath).map((e) => path.basename(e.filePath.replace(/(:prompt|:response|:round)$/, ""))),
		);
		if (freshExisting.has(roundFile)) {
			skipped++;
			continue;
		}

		// Embed prompt
		out.log(`  🔄 Embedding round ${round.turnIndex + 1}/${rounds.length}...`);
		const apiKey = options.apiKey ?? process.env.OPENROUTER_API_KEY;
		if (!apiKey) {
			throw new Error("OPENROUTER_API_KEY environment variable required");
		}
		const promptVector = await embedText(round.userPrompt.slice(0, MAX_PROMPT_CHARS), apiKey, {
			fetchImpl: options.fetchImpl,
		});
		appendVectorIndexEntry(indexPath, normalize(promptVector), `${roundFile}:prompt`);

		// Embed response
		const respVector = await embedText(round.responseSequence.slice(0, MAX_RESPONSE_CHARS), apiKey, {
			fetchImpl: options.fetchImpl,
		});
		appendVectorIndexEntry(indexPath, normalize(respVector), `${roundFile}:response`);

		embedded++;
	}

	out.log(`\n✅ Done. ${embedded} new rounds embedded, ${skipped} already in index.`);
	out.log(`   Index: ${indexPath} (${loadVectorIndex(indexPath).length} vectors)`);
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
