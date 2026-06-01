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
import { computeContentHash } from "../src/core/hash.ts";

// ─────────────────────────────────────────────
// Config (matches digest-session.ts)
// ─────────────────────────────────────────────

const SESSIONS_DIR = path.resolve(os.homedir(), ".pi", "agent", "sessions");
const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");
const INDEX_PATH = path.resolve(ROUNDS_DIR, "index.csv");

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";
// Keep bulk digest single-worker: stale-hash migrations rewrite index.csv and delete
// duplicate files, so concurrent duplicate-content rounds can race and leave stale refs.
const CONCURRENCY = 1;
const MAX_PROMPT_CHARS = 8000;
const MAX_RESPONSE_CHARS = 8000;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface ToolCallDetail {
	index: number;
	name: string;
	arguments: string;
	result_summary: string;
	result_full?: string;
	result_truncated?: boolean;
}

interface ResponseSegment {
	type: "text" | "toolCall";
	text?: string;
	toolCallIndex?: number;
}

interface Round {
	id: string;
	userPrompt: string;
	responseSequence: string;
	responseSegments: ResponseSegment[];
	userTimestamp: number;
	responseEndTimestamp: number;
	turnIndex: number; // serialized — keep name for backward compat
	sessionLabel: string; // human-readable label for this session file
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
}

// ─────────────────────────────────────────────
// Parse a single JSONL file into rounds
// ─────────────────────────────────────────────

function parseSessionFile(filePath: string, sessionLabel: string): Round[] {
	const content = fs.readFileSync(filePath, "utf-8");
	const lines = content.trim().split("\n").filter(Boolean);
	const entries: Array<Record<string, unknown>> = lines.map((l) => JSON.parse(l));

	const rounds: Round[] = [];
	let currentUserMsg: Record<string, unknown> | null = null;
	let responseParts: string[] = [];
	let responseSegments: ResponseSegment[] = [];
	let toolNames: string[] = [];
	let toolCallCount = 0;
	let toolCalls: ToolCallDetail[] = [];
	let roundIndex = 0;

	for (const entry of entries) {
		if (entry.type !== "message") continue;
		const msg = entry.message as Record<string, unknown> | undefined;
		if (!msg) continue;

		const role = msg.role as string;
		const content = msg.content as Array<{ type: string; text?: string }> | undefined;
		const timestamp = msg.timestamp as number | undefined;

		if (role === "user") {
			// Save previous round if exists
			if (currentUserMsg) {
				rounds.push({
					id: currentUserMsg.id as string,
					userPrompt: extractText(
						(currentUserMsg.message as Record<string, unknown>)?.content as
							| Array<{ type: string; text?: string }>
							| undefined,
					),
					responseSequence: responseParts.join("\n\n").trim(),
					responseSegments,
					userTimestamp: ((currentUserMsg.message as Record<string, unknown>)?.timestamp as number) ?? 0,
					responseEndTimestamp: timestamp ?? Date.now(),
					turnIndex: roundIndex,
					sessionLabel,
					toolCallCount,
					toolCallNames: [...new Set(toolNames)],
					toolCalls,
				});
				roundIndex++;
			}
			currentUserMsg = entry;
			responseParts = [];
			responseSegments = [];
			toolNames = [];
			toolCallCount = 0;
			toolCalls = [];
		} else if (role === "assistant" && currentUserMsg && content) {
			// Single ordered pass: interleave text and tool call blocks
			for (const block of content) {
				if (block.type === "text" && block.text) {
					responseParts.push(block.text);
					responseSegments.push({ type: "text", text: block.text });
				} else if (block.type === "toolCall") {
					toolCallCount++;
					const blockRec = block as Record<string, unknown>;
					const name = blockRec.name as string | undefined;
					if (name) toolNames.push(name);
					toolCalls.push({
						index: toolCalls.length,
						name: name ?? "unknown",
						arguments: JSON.stringify(blockRec.arguments ?? {}),
						result_summary: "",
					});
					responseSegments.push({ type: "toolCall", toolCallIndex: toolCalls.length - 1 });
				}
			}
		} else if (role === "toolResult" && currentUserMsg) {
			// Count tool results and pair with pending tool calls (sequential match)
			const toolName = msg.toolName as string | undefined;
			if (toolName) toolNames.push(toolName);
			// Pair with the most recent tool call that lacks a result
			for (let i = toolCalls.length - 1; i >= 0; i--) {
				if (toolCalls[i].result_summary === "") {
					const resultContent = msg.content as Array<{ type: string; text?: string }> | undefined;
					const resultText = resultContent ? extractText(resultContent) : "";
					toolCalls[i].result_summary = resultText.slice(0, 300);
					toolCalls[i].result_full = resultText;
					toolCalls[i].result_truncated = false;
					break;
				}
			}
		}
	}

	// Save last round — only if it has a non-empty response or we have multiple rounds.
	// Skip rounds whose response is trivially short (< 20 chars) — these are usually
	// session files that ended mid-stream (truncated assistant response).
	const finalResponse = responseParts.join("\n\n").trim();
	if (currentUserMsg && (finalResponse.length >= 20 || roundIndex > 0)) {
		rounds.push({
			id: currentUserMsg.id as string,
			userPrompt: extractText(
				(currentUserMsg.message as Record<string, unknown>)?.content as
					| Array<{ type: string; text?: string }>
					| undefined,
			),
			responseSequence: finalResponse,
			responseSegments,
			userTimestamp: ((currentUserMsg.message as Record<string, unknown>)?.timestamp as number) ?? 0,
			responseEndTimestamp: Date.now(),
			turnIndex: roundIndex,
			sessionLabel,
			toolCallCount,
			toolCallNames: [...new Set(toolNames)],
			toolCalls,
		});
	}

	return rounds;
}

function extractText(content?: Array<{ type: string; text?: string }>): string {
	if (!content || !Array.isArray(content)) return "";
	return content
		.filter((c): c is { type: string; text: string } => c.type === "text" && typeof c.text === "string")
		.map((c) => c.text)
		.join(" ")
		.trim();
}

// ─────────────────────────────────────────────
// Embedding (single call)
// ─────────────────────────────────────────────

async function embed(text: string): Promise<number[]> {
	const response = await fetch(OPENROUTER_URL, {
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

// ─────────────────────────────────────────────
// Index I/O
// ─────────────────────────────────────────────

function loadIndexFilePaths(): Set<string> {
	if (!fs.existsSync(INDEX_PATH)) return new Set();
	const lines = fs.readFileSync(INDEX_PATH, "utf-8").trim().split("\n").filter(Boolean);
	return new Set(
		lines.map((line) => {
			const [, filePath] = line.split(",", 2);
			return filePath.replace(/(:prompt|:response|:round)$/, "");
		}),
	);
}

function loadIndexLines(): string[] {
	if (!fs.existsSync(INDEX_PATH)) return [];
	return fs.readFileSync(INDEX_PATH, "utf-8").trim().split("\n").filter(Boolean);
}

function appendToIndex(vector: number[], filePath: string): void {
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	fs.appendFileSync(INDEX_PATH, `${b64},${filePath}\n`);
}

/**
 * Rewrite index entries from an old round filename to the new content-hash filename.
 * This preserves accessibility even if re-embedding would fail or be skipped.
 */
function migrateIndexEntries(oldRoundFile: string, newRoundFile: string): void {
	if (!fs.existsSync(INDEX_PATH)) return;
	const lines = loadIndexLines();
	const migrated = lines.map((line) => {
		const comma = line.indexOf(",");
		const fp = line.slice(comma + 1);
		if (!fp.startsWith(oldRoundFile)) return line;
		return `${line.slice(0, comma + 1)}${newRoundFile}${fp.slice(oldRoundFile.length)}`;
	});
	fs.writeFileSync(INDEX_PATH, migrated.join("\n") + (migrated.length > 0 ? "\n" : ""));
}

/**
 * Scan the rounds directory for stale .json files whose stored content hashes to
 * the given round filename. Returns matching filenames (without paths).
 */
function findStaleContentMatches(roundFile: string): string[] {
	const files = fs.readdirSync(ROUNDS_DIR).filter((f) => f.endsWith(".json") && !f.startsWith("index"));
	const matches: string[] = [];
	for (const f of files) {
		if (f === roundFile) continue;
		try {
			const data = JSON.parse(fs.readFileSync(path.join(ROUNDS_DIR, f), "utf-8")) as Partial<Round>;
			const hash = `${computeContentHash(data.userPrompt ?? "", data.responseSequence ?? "", data.toolCalls)}.json`;
			if (hash === roundFile) {
				matches.push(f);
			}
		} catch {
			// Corrupt file — skip
		}
	}
	return matches;
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
	if (!OPENROUTER_API_KEY) {
		console.error("❌ OPENROUTER_API_KEY environment variable required");
		process.exit(1);
	}

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
	const existingRounds = loadIndexFilePaths();
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
		const staleFiles = findStaleContentMatches(roundFile);
		for (const staleFile of staleFiles) {
			migrateIndexEntries(staleFile, roundFile);
		}

		// Write round file before deleting stale copies.
		fs.writeFileSync(path.resolve(ROUNDS_DIR, roundFile), JSON.stringify(round, null, 2));
		for (const staleFile of staleFiles) {
			fs.unlinkSync(path.join(ROUNDS_DIR, staleFile));
			process.stderr.write(`  ♻️  Migrated stale: ${staleFile} → ${roundFile}\n`);
		}

		// Skip embedding if already indexed under the correct hash
		// (must reload after potential deletions above)
		const indexedAfterCleanup = loadIndexFilePaths();
		if (indexedAfterCleanup.has(roundFile)) {
			completed++;
			process.stderr.write(`  ⏭  [${completed}/${totalNew}] ${roundId} (already indexed)\n`);
			return;
		}

		try {
			const promptVector = await embed(round.userPrompt.slice(0, MAX_PROMPT_CHARS));
			appendToIndex(normalize(promptVector), `${roundFile}:prompt`);

			const respText = round.responseSequence.slice(0, MAX_RESPONSE_CHARS);
			if (respText) {
				const respVector = await embed(respText);
				appendToIndex(normalize(respVector), `${roundFile}:response`);
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
