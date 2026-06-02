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

const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const _OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"; // actually embeddings endpoint
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
// Embedding via OpenRouter
// ─────────────────────────────────────────────

export interface EmbedOptions {
	apiKey?: string;
	fetchImpl?: typeof fetch;
}

export async function embed(text: string, options: EmbedOptions = {}): Promise<number[]> {
	const apiKey = options.apiKey ?? process.env.OPENROUTER_API_KEY;
	if (!apiKey) {
		throw new Error("OPENROUTER_API_KEY environment variable required");
	}

	const fetchImpl = options.fetchImpl ?? fetch;
	const response = await fetchImpl("https://openrouter.ai/api/v1/embeddings", {
		method: "POST",
		headers: {
			Authorization: `Bearer ${apiKey}`,
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

export function normalize(v: number[]): number[] {
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
		const promptVector = await embed(round.userPrompt.slice(0, MAX_PROMPT_CHARS), {
			apiKey: options.apiKey,
			fetchImpl: options.fetchImpl,
		});
		appendVectorIndexEntry(indexPath, normalize(promptVector), `${roundFile}:prompt`);

		// Embed response
		const respVector = await embed(round.responseSequence.slice(0, MAX_RESPONSE_CHARS), {
			apiKey: options.apiKey,
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
