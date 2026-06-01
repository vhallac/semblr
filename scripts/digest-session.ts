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
import { computeContentHash } from "../src/core/hash.ts";
import {
	appendVectorIndexEntry,
	findStaleContentMatches as findStaleContentMatchesInDir,
	loadVectorIndex,
	migrateIndexEntries as migrateIndexEntriesFile,
} from "../src/core/index-io.ts";
import { type ParsedPiRound, parsePiSessionJsonl } from "../src/core/pi-session.ts";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

type Round = ParsedPiRound;

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const _OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"; // actually embeddings endpoint
const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");
const INDEX_PATH = path.resolve(ROUNDS_DIR, "index.csv");
const MAX_PROMPT_CHARS = 8000;
const MAX_RESPONSE_CHARS = 8000;

// ─────────────────────────────────────────────
// Parse session into rounds
// ─────────────────────────────────────────────

function parseSession(filePath: string): Round[] {
	return parsePiSessionJsonl(fs.readFileSync(filePath, "utf-8"));
}

// ─────────────────────────────────────────────
// Embedding via OpenRouter
// ─────────────────────────────────────────────

async function embed(text: string): Promise<number[]> {
	if (!OPENROUTER_API_KEY) {
		throw new Error("OPENROUTER_API_KEY environment variable required");
	}

	const response = await fetch("https://openrouter.ai/api/v1/embeddings", {
		method: "POST",
		headers: {
			Authorization: `Bearer ${OPENROUTER_API_KEY}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			model: EMBEDDING_MODEL,
			input: text,
		}),
	});

	if (!response.ok) {
		const err = await response.text();
		throw new Error(`OpenRouter embedding error (${response.status}): ${err}`);
	}

	const data = (await response.json()) as {
		data: Array<{ embedding: number[] }>;
	};
	return data.data[0].embedding;
}

// ─────────────────────────────────────────────
// Vector helpers
// ─────────────────────────────────────────────

function normalize(v: number[]): number[] {
	const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
	return mag === 0 ? v : v.map((x) => x / mag);
}

function _cosineSimilarity(a: number[], b: number[]): number {
	let dot = 0;
	for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
	return dot;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

async function main() {
	const sessionFile = process.argv[2];
	if (!sessionFile) {
		console.error("Usage: npx tsx scripts/digest-session.ts <session-file>");
		process.exit(1);
	}

	console.log(`📂 Session: ${sessionFile}`);
	const rounds = parseSession(sessionFile);
	console.log(`📊 Parsed ${rounds.length} rounds`);

	// Ensure rounds directory
	fs.mkdirSync(ROUNDS_DIR, { recursive: true });

	let embedded = 0;
	let skipped = 0;

	for (const round of rounds) {
		const roundFile = `${computeContentHash(round.userPrompt, round.responseSequence, round.toolCalls)}.json`;

		// Check for stale files whose stored full hash material belongs under this
		// content-hash filename. If found, migrate old index entries before deleting
		// old files so the round remains retrievable even if embedding is unavailable.
		const staleFiles = findStaleContentMatchesInDir(ROUNDS_DIR, roundFile);
		for (const staleFile of staleFiles) {
			migrateIndexEntriesFile(INDEX_PATH, staleFile, roundFile);
		}

		// Always write the round file (idempotent) before deleting stale copies.
		fs.writeFileSync(path.resolve(ROUNDS_DIR, roundFile), JSON.stringify(round, null, 2));
		for (const staleFile of staleFiles) {
			fs.unlinkSync(path.join(ROUNDS_DIR, staleFile));
			console.log(`  ♻️  Migrated stale: ${staleFile} → ${roundFile}`);
		}

		// Refresh existing set after potential deletions
		const freshExisting = new Set(
			loadVectorIndex(INDEX_PATH).map((e) => path.basename(e.filePath.replace(/(:prompt|:response|:round)$/, ""))),
		);
		if (freshExisting.has(roundFile)) {
			skipped++;
			continue;
		}

		// Embed prompt
		console.log(`  🔄 Embedding round ${round.turnIndex + 1}/${rounds.length}...`);
		const promptVector = await embed(round.userPrompt.slice(0, MAX_PROMPT_CHARS));
		appendVectorIndexEntry(INDEX_PATH, normalize(promptVector), `${roundFile}:prompt`);

		// Embed response
		const respVector = await embed(round.responseSequence.slice(0, MAX_RESPONSE_CHARS));
		appendVectorIndexEntry(INDEX_PATH, normalize(respVector), `${roundFile}:response`);

		embedded++;
	}

	console.log(`\n✅ Done. ${embedded} new rounds embedded, ${skipped} already in index.`);
	console.log(`   Index: ${INDEX_PATH} (${loadVectorIndex(INDEX_PATH).length} vectors)`);
}

main().catch((err) => {
	console.error("❌ Error:", err);
	process.exit(1);
});
