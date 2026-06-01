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
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
}

interface SessionEntry {
	type: string;
	id: string;
	parentId: string | null;
	timestamp?: string;
	message?: {
		role: string;
		content: Array<{ type: string; text?: string }>;
		timestamp?: number;
	};
}

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"; // actually embeddings endpoint
const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");
const INDEX_PATH = path.resolve(ROUNDS_DIR, "index.csv");

// ─────────────────────────────────────────────
// Parse session into rounds
// ─────────────────────────────────────────────

function parseSession(filePath: string): Round[] {
	const content = fs.readFileSync(filePath, "utf-8");
	const lines = content.trim().split("\n").filter(Boolean);
	const entries: SessionEntry[] = lines.map((l) => JSON.parse(l));

	const rounds: Round[] = [];
	let currentUserMsg: SessionEntry | null = null;
	let responseParts: string[] = [];
	let responseSegments: ResponseSegment[] = [];
	let toolNames: string[] = [];
	let toolCallCount = 0;
	let toolCalls: ToolCallDetail[] = [];
	let roundIndex = 0;

	for (const entry of entries) {
		if (entry.type !== "message" || !entry.message) continue;
		const { role, content } = entry.message;

		if (role === "user") {
			// Save previous round if exists
			if (currentUserMsg) {
				rounds.push({
					id: currentUserMsg.id,
					userPrompt: extractText(
						(currentUserMsg.message?.content ?? []) as Array<{ type: string; text?: string }>,
					),
					responseSequence: responseParts.join("\n\n").trim(),
					responseSegments,
					userTimestamp: currentUserMsg.message?.timestamp ?? 0,
					responseEndTimestamp: entry.message?.timestamp ?? 0,
					turnIndex: roundIndex,
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
		} else if (role === "assistant" && currentUserMsg) {
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
			const toolName = (entry.message as Record<string, unknown>).toolName as string | undefined;
			if (toolName) toolNames.push(toolName);
			// Pair with the most recent tool call that lacks a result
			for (let i = toolCalls.length - 1; i >= 0; i--) {
				if (toolCalls[i].result_summary === "") {
					const resultContent = entry.message.content;
					const resultText = resultContent ? extractText(resultContent) : "";
					toolCalls[i].result_summary = resultText.slice(0, 300);
					toolCalls[i].result_full = resultText;
					toolCalls[i].result_truncated = false;
					break;
				}
			}
		}
	}

	// Save last round
	if (currentUserMsg) {
		rounds.push({
			id: currentUserMsg.id,
			userPrompt: extractText((currentUserMsg.message?.content ?? []) as Array<{ type: string; text?: string }>),
			responseSequence: responseParts.join("\n\n").trim(),
			responseSegments,
			userTimestamp: currentUserMsg.message?.timestamp ?? 0,
			responseEndTimestamp: Date.now(),
			turnIndex: roundIndex,
			toolCallCount,
			toolCallNames: [...new Set(toolNames)],
			toolCalls,
		});
	}

	return rounds;
}

function extractText(content: Array<{ type: string; text?: string }>): string {
	return content
		.filter((c) => c.type === "text" && c.text)
		.map((c) => c.text ?? "")
		.join(" ")
		.trim();
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

function cosineSimilarity(a: number[], b: number[]): number {
	let dot = 0;
	for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
	return dot;
}

// ─────────────────────────────────────────────
// Index I/O
// ─────────────────────────────────────────────

function loadIndex(): Array<{ vector: number[]; filePath: string }> {
	if (!fs.existsSync(INDEX_PATH)) return [];
	const lines = fs.readFileSync(INDEX_PATH, "utf-8").trim().split("\n").filter(Boolean);
	return lines.map((line) => {
		const comma = line.indexOf(",");
		const b64 = line.slice(0, comma);
		const filePath = line.slice(comma + 1);
		const vector = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
		return { vector, filePath };
	});
}

function appendToIndex(vector: number[], filePath: string): void {
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	fs.appendFileSync(INDEX_PATH, `${b64},${filePath}\n`);
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

	// Check existing index to skip already-processed rounds
	const existing = new Set(loadIndex().map((e) => path.basename(e.filePath)));

	let embedded = 0;
	let skipped = 0;

	for (const round of rounds) {
		const crypto = require("node:crypto");
		const roundFile = `${crypto
			.createHash("md5")
			.update(round.userPrompt + round.responseSequence)
			.digest("hex")}.json`;

		// Always write the round file (idempotent)
		fs.writeFileSync(path.resolve(ROUNDS_DIR, roundFile), JSON.stringify(round, null, 2));

		if (existing.has(roundFile)) {
			skipped++;
			continue;
		}

		// Embed prompt
		console.log(`  🔄 Embedding round ${round.turnIndex + 1}/${rounds.length}...`);
		const promptVector = await embed(round.userPrompt);
		appendToIndex(normalize(promptVector), `${roundFile}:prompt`);

		// Embed response
		const respVector = await embed(round.responseSequence.slice(0, 8000)); // conservative token limit
		appendToIndex(normalize(respVector), `${roundFile}:response`);

		embedded++;
	}

	console.log(`\n✅ Done. ${embedded} new rounds embedded, ${skipped} already in index.`);
	console.log(`   Index: ${INDEX_PATH} (${loadIndex().length} vectors)`);
}

main().catch((err) => {
	console.error("❌ Error:", err);
	process.exit(1);
});
