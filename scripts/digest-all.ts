/**
 * digest-all.ts — Bulk-embed all pi session JSONL files into the semblr index.
 *
 * Iterates every session in ~/.pi/agent/sessions/, skips already-indexed rounds,
 * parallelizes embedding via OpenRouter.
 *
 * Usage:
 *   OPENROUTER_API_KEY="$(pass show ai/openrouter)" npx tsx scripts/digest-all.ts
 *
 * Safe to run while pi is using the extension — the index is append-only.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { embedText, normalize } from "../lib/embed.ts";
import { computeContentHash } from "../lib/hash.ts";
import {
	appendVectorIndexEntry,
	findStaleContentMatches as findStaleContentMatchesInDir,
	loadIndexedRoundFiles,
	migrateIndexEntries as migrateIndexEntriesFile,
} from "../src/core/index-io.ts";
import { type ParsedPiRound, parsePiSessionJsonl } from "../src/core/pi-session.ts";

// ─────────────────────────────────────────────
// Config (matches digest-session.ts)
// ─────────────────────────────────────────────

const SESSIONS_DIR = path.resolve(os.homedir(), ".pi", "agent", "sessions");
const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");
const INDEX_PATH = path.resolve(ROUNDS_DIR, "index.csv");
// Keep bulk digest single-worker: stale-hash migrations rewrite index.csv and delete
// duplicate files, so concurrent duplicate-content rounds can race and leave stale refs.
const CONCURRENCY = 1;
const MAX_PROMPT_CHARS = 8000;
const MAX_RESPONSE_CHARS = 8000;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

type Round = ParsedPiRound & { sessionLabel: string };

// ─────────────────────────────────────────────
// Parse a single JSONL file into rounds
// ─────────────────────────────────────────────

function parseSessionFile(filePath: string, sessionLabel: string): Round[] {
	return parsePiSessionJsonl(fs.readFileSync(filePath, "utf-8"), {
		sessionLabel,
		skipShortFinalResponse: true,
	});
}

// ─────────────────────────────────────────────
// Count user messages in a JSONL (for progress)
// ─────────────────────────────────────────────

function _countUserMessages(filePath: string): number {
	const content = fs.readFileSync(filePath, "utf-8");
	let count = 0;
	for (const line of content.trim().split("\n").filter(Boolean)) {
		try {
			const obj = JSON.parse(line);
			if (obj.type === "message" && obj.message?.role === "user") count++;
		} catch {}
	}
	return count;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

async function main() {
	const apiKey = process.env.OPENROUTER_API_KEY;
	if (!apiKey) {
		console.error("❌ OPENROUTER_API_KEY environment variable required");
		process.exit(1);
	}
	// TypeScript doesn't know process.exit never returns
	// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
	const key: string = apiKey;

	// Gather all session JSONL files
	const sessionDirs = fs
		.readdirSync(SESSIONS_DIR)
		.filter((d) => d.startsWith("--"))
		.map((d) => path.join(SESSIONS_DIR, d));

	const jsonlFiles: Array<{ filePath: string; label: string }> = [];
	for (const dir of sessionDirs) {
		const label = path.basename(dir);
		const files = fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl"));
		for (const f of files) {
			jsonlFiles.push({ filePath: path.join(dir, f), label });
		}
	}

	jsonlFiles.sort((a, b) => a.filePath.localeCompare(b.filePath));

	console.log(`📂 Found ${jsonlFiles.length} session files across ${sessionDirs.length} directories\n`);

	// Ensure the .pi/rounds dir
	fs.mkdirSync(ROUNDS_DIR, { recursive: true });

	// Load existing index dedup set
	const existingRounds = loadIndexedRoundFiles(INDEX_PATH);
	console.log(`📊 Already indexed: ${existingRounds.size} rounds\n`);

	// Parse all sessions into a flat list of rounds (skipping already-indexed)
	const allRounds: Round[] = [];
	let skippedTotal = 0;

	for (const { filePath, label } of jsonlFiles) {
		const rounds = parseSessionFile(filePath, label);
		const newRounds = rounds.filter((t) => {
			const key = `${computeContentHash(t.userPrompt, t.responseSequence, t.toolCalls)}.json`;
			return !existingRounds.has(key);
		});
		skippedTotal += rounds.length - newRounds.length;
		allRounds.push(...newRounds);
	}

	const totalNew = allRounds.length;
	console.log(`📊 New rounds to embed: ${totalNew} (${skippedTotal} already indexed)\n`);

	if (totalNew === 0) {
		console.log("✨ Nothing to do — all sessions already indexed!");
		return;
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
		const staleFiles = findStaleContentMatchesInDir(ROUNDS_DIR, roundFile);
		for (const staleFile of staleFiles) {
			migrateIndexEntriesFile(INDEX_PATH, staleFile, roundFile);
		}

		// Write round file before deleting stale copies.
		fs.writeFileSync(path.resolve(ROUNDS_DIR, roundFile), JSON.stringify(round, null, 2));
		for (const staleFile of staleFiles) {
			fs.unlinkSync(path.join(ROUNDS_DIR, staleFile));
			process.stderr.write(`  ♻️  Migrated stale: ${staleFile} → ${roundFile}\n`);
		}

		// Skip embedding if already indexed under the correct hash
		// (must reload after potential deletions above)
		const indexedAfterCleanup = loadIndexedRoundFiles(INDEX_PATH);
		if (indexedAfterCleanup.has(roundFile)) {
			completed++;
			process.stderr.write(`  ⏭  [${completed}/${totalNew}] ${roundId} (already indexed)\n`);
			return;
		}

		try {
			const promptVector = await embedText(round.userPrompt.slice(0, MAX_PROMPT_CHARS), key);
			appendVectorIndexEntry(INDEX_PATH, normalize(promptVector), `${roundFile}:prompt`);

			const respText = round.responseSequence.slice(0, MAX_RESPONSE_CHARS);
			if (respText) {
				const respVector = await embedText(respText, key);
				appendVectorIndexEntry(INDEX_PATH, normalize(respVector), `${roundFile}:response`);
			}

			completed++;
			const pct = ((completed / totalNew) * 100).toFixed(1);
			process.stderr.write(`  ✅ [${completed}/${totalNew} ${pct}%] ${roundId}\n`);
		} catch (err) {
			errors++;
			process.stderr.write(`  ❌ [ERROR] ${roundId}: ${(err as Error).message}\n`);
		}
	}

	// Run with concurrency limit
	async function runQueue(): Promise<void> {
		const queue = [...allRounds];
		const workers: Promise<void>[] = [];

		for (let i = 0; i < CONCURRENCY; i++) {
			workers.push(
				(async () => {
					while (queue.length > 0) {
						const round = queue.shift();
						if (round) await processRound(round);
					}
				})(),
			);
		}

		await Promise.all(workers);
	}

	const startTime = Date.now();
	await runQueue();
	const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

	const finalCount = fs.existsSync(INDEX_PATH)
		? fs.readFileSync(INDEX_PATH, "utf-8").trim().split("\n").filter(Boolean).length
		: 0;

	console.log(`\n✅ Done in ${elapsed}s. ${completed} rounds embedded, ${errors} errors.`);
	console.log(`   Index: ${finalCount} vectors at ${INDEX_PATH}`);
	console.log(
		`   Rounds: ${fs.readdirSync(ROUNDS_DIR).filter((f) => f.endsWith(".json") && !f.startsWith("index")).length} files`,
	);
}

main().catch((err) => {
	console.error("❌ Fatal:", err);
	process.exit(1);
});
