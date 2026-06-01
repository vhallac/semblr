/**
 * import-claude-code.ts — Import Claude Code JSONL history into Semblr.
 *
 * Reads JSONL files under ~/.claude/projects, converts each real user turn into the
 * Semblr round format, and embeds prompt + response into the shared index.
 *
 * Usage:
 *   OPENROUTER_API_KEY="$(pass show ai/openrouter)" npx tsx scripts/import-claude-code.ts
 *
 * Options:
 *   --dry-run              Parse/count only; do not write or embed
 *   --include-sidechains   Include Claude Code subagent/sidechain transcripts
 *   --limit N             Import at most N new rounds (useful for testing)
 */

import { spawnSync } from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const CLAUDE_PROJECTS_DIR = process.env.CLAUDE_PROJECTS_DIR || path.resolve(os.homedir(), ".claude", "projects");
const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");
const INDEX_PATH = path.resolve(ROUNDS_DIR, "index.csv");

const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";
const CONCURRENCY = Number(process.env.SEMBLR_IMPORT_CONCURRENCY || "5");
const MAX_RESPONSE_CHARS = 8000;

interface ToolCallDetail {
	index: number;
	name: string;
	arguments: string;
	result_summary: string;
}

interface ResponseSegment {
	type: "text" | "toolCall";
	text?: string;
	toolCallIndex?: number;
}

interface Round {
	id: string;
	source: "claude-code";
	userPrompt: string;
	responseSequence: string;
	responseSegments: ResponseSegment[];
	userTimestamp: number;
	responseEndTimestamp: number;
	turnIndex: number;
	sessionLabel: string;
	claudeSessionId?: string;
	cwd?: string;
	gitBranch?: string;
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
}

function argValue(name: string): string | null {
	const idx = process.argv.indexOf(name);
	return idx >= 0 ? (process.argv[idx + 1] ?? null) : null;
}

const DRY_RUN = process.argv.includes("--dry-run");
const INCLUDE_SIDECHAINS = process.argv.includes("--include-sidechains");
const LIMIT = argValue("--limit") ? Number(argValue("--limit")) : null;

function getApiKey(): string | null {
	if (process.env.OPENROUTER_API_KEY) return process.env.OPENROUTER_API_KEY;
	try {
		const result = spawnSync("pass", ["show", "ai/openrouter"], {
			timeout: 5000,
			stdio: ["pipe", "pipe", "pipe"],
		});
		if (result.status === 0) return result.stdout.toString().trim() || null;
	} catch {}
	return null;
}

function walkJsonlFiles(dir: string): string[] {
	if (!fs.existsSync(dir)) return [];
	const out: string[] = [];
	for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) out.push(...walkJsonlFiles(full));
		else if (entry.isFile() && entry.name.endsWith(".jsonl")) out.push(full);
	}
	return out.sort();
}

function textFromContent(content: unknown, opts: { includeToolResults?: boolean } = {}): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const block of content as Array<Record<string, unknown>>) {
		if (block.type === "text" && typeof block.text === "string") parts.push(block.text);
		if (opts.includeToolResults && block.type === "tool_result") {
			const c = block.content;
			if (typeof c === "string") parts.push(c);
			else if (Array.isArray(c)) parts.push(textFromContent(c, { includeToolResults: true }));
		}
	}
	return parts.join("\n").trim();
}

function isRealUserPrompt(entry: Record<string, any>): boolean {
	if (entry.type !== "user") return false;
	const content = entry.message?.content;
	if (typeof content === "string") return content.trim().length > 0;
	if (!Array.isArray(content)) return false;
	// Claude tool results are stored as user messages; they are not new turns.
	return content.some((b: any) => b?.type === "text" && typeof b.text === "string" && b.text.trim());
}

function roundFileName(round: Pick<Round, "userPrompt" | "responseSequence">): string {
	return `${crypto
		.createHash("md5")
		.update(round.userPrompt + round.responseSequence)
		.digest("hex")}.json`;
}

function parseClaudeFile(filePath: string): Round[] {
	const raw = fs.readFileSync(filePath, "utf-8");
	const entries = raw
		.split("\n")
		.filter(Boolean)
		.map((line, i) => {
			try {
				return JSON.parse(line);
			} catch (e) {
				throw new Error(`${filePath}:${i + 1}: invalid JSON: ${(e as Error).message}`);
			}
		});

	const rounds: Round[] = [];
	let currentUser: Record<string, any> | null = null;
	let responseParts: string[] = [];
	let responseSegments: ResponseSegment[] = [];
	let toolCalls: ToolCallDetail[] = [];
	let toolCallNames: string[] = [];
	let pendingById = new Map<string, ToolCallDetail>();
	let roundIndex = 0;
	let responseEndTimestamp = 0;

	function flush() {
		if (!currentUser) return;
		const userPrompt = textFromContent(currentUser.message?.content);
		const responseSequence = responseParts.join("\n\n").trim();
		if (!userPrompt || responseSequence.length < 20) return;
		const cwd = currentUser.cwd as string | undefined;
		const sessionLabel = path.relative(CLAUDE_PROJECTS_DIR, filePath) || path.basename(filePath);
		const round: Round = {
			id: crypto
				.createHash("md5")
				.update(userPrompt + responseSequence)
				.digest("hex"),
			source: "claude-code",
			userPrompt,
			responseSequence,
			responseSegments,
			userTimestamp: Date.parse(currentUser.timestamp) || 0,
			responseEndTimestamp: responseEndTimestamp || Date.now(),
			turnIndex: roundIndex,
			sessionLabel,
			claudeSessionId: currentUser.sessionId,
			cwd,
			gitBranch: currentUser.gitBranch,
			toolCallCount: toolCalls.length,
			toolCallNames: [...new Set(toolCallNames)],
			toolCalls,
		};
		rounds.push(round);
		roundIndex++;
	}

	function resetFor(entry: Record<string, any>) {
		currentUser = entry;
		responseParts = [];
		responseSegments = [];
		toolCalls = [];
		toolCallNames = [];
		pendingById = new Map();
		responseEndTimestamp = Date.parse(entry.timestamp) || 0;
	}

	for (const entry of entries) {
		if (!INCLUDE_SIDECHAINS && entry.isSidechain) continue;

		if (isRealUserPrompt(entry)) {
			flush();
			resetFor(entry);
			continue;
		}

		if (!currentUser) continue;

		if (entry.type === "assistant" && Array.isArray(entry.message?.content)) {
			responseEndTimestamp = Date.parse(entry.timestamp) || responseEndTimestamp;
			for (const block of entry.message.content as Array<Record<string, any>>) {
				if (block.type === "text" && typeof block.text === "string" && block.text.trim()) {
					responseParts.push(block.text);
					responseSegments.push({ type: "text", text: block.text });
				} else if (block.type === "tool_use") {
					const detail: ToolCallDetail = {
						index: toolCalls.length,
						name: typeof block.name === "string" ? block.name : "unknown",
						arguments: JSON.stringify(block.input ?? {}),
						result_summary: "",
					};
					toolCalls.push(detail);
					toolCallNames.push(detail.name);
					if (typeof block.id === "string") pendingById.set(block.id, detail);
					responseSegments.push({ type: "toolCall", toolCallIndex: detail.index });
				}
			}
		} else if (entry.type === "user" && Array.isArray(entry.message?.content)) {
			// Tool results are encoded as user messages with tool_result blocks.
			for (const block of entry.message.content as Array<Record<string, any>>) {
				if (block.type !== "tool_result") continue;
				const id = typeof block.tool_use_id === "string" ? block.tool_use_id : undefined;
				const detail = id ? pendingById.get(id) : [...toolCalls].reverse().find((tc) => !tc.result_summary);
				if (detail) detail.result_summary = textFromContent([block], { includeToolResults: true }).slice(0, 300);
			}
		}
	}

	flush();
	return rounds;
}

async function embed(text: string, apiKey: string): Promise<number[]> {
	const response = await fetch(OPENROUTER_URL, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${apiKey}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
	});
	if (!response.ok) throw new Error(`OpenRouter embedding error (${response.status}): ${await response.text()}`);
	const data = (await response.json()) as { data: Array<{ embedding: number[] }> };
	return data.data[0].embedding;
}

function normalize(v: number[]): number[] {
	const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
	return mag === 0 ? v : v.map((x) => x / mag);
}

function appendToIndex(vector: number[], filePath: string): void {
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	fs.appendFileSync(INDEX_PATH, `${b64},${filePath}\n`);
}

function loadIndexedRoundFiles(): Set<string> {
	if (!fs.existsSync(INDEX_PATH)) return new Set();
	return new Set(
		fs
			.readFileSync(INDEX_PATH, "utf-8")
			.split("\n")
			.filter(Boolean)
			.map((line) => line.slice(line.indexOf(",") + 1).replace(/:prompt$|:response$/, "")),
	);
}

async function main() {
	const apiKey = getApiKey();
	if (!DRY_RUN && !apiKey) {
		console.error("❌ OPENROUTER_API_KEY required (or pass show ai/openrouter must work)");
		process.exit(1);
	}

	const files = walkJsonlFiles(CLAUDE_PROJECTS_DIR);
	console.log(`📂 Found ${files.length} Claude Code JSONL files in ${CLAUDE_PROJECTS_DIR}`);
	console.log(`   sidechains: ${INCLUDE_SIDECHAINS ? "included" : "skipped (use --include-sidechains to include)"}`);

	const allRounds: Round[] = [];
	let parseErrors = 0;
	for (const file of files) {
		try {
			allRounds.push(...parseClaudeFile(file));
		} catch (e) {
			parseErrors++;
			console.warn(`⚠ ${(e as Error).message}`);
		}
	}

	const indexed = loadIndexedRoundFiles();
	const seen = new Set<string>();
	let duplicates = 0;
	let alreadyIndexed = 0;
	let newRounds = allRounds.filter((round) => {
		const file = roundFileName(round);
		if (seen.has(file)) {
			duplicates++;
			return false;
		}
		seen.add(file);
		if (indexed.has(file)) {
			alreadyIndexed++;
			return false;
		}
		return true;
	});
	if (LIMIT != null && Number.isFinite(LIMIT)) newRounds = newRounds.slice(0, LIMIT);

	console.log(`📊 Parsed rounds: ${allRounds.length}`);
	console.log(
		`📊 New rounds to import: ${newRounds.length} (${alreadyIndexed} already indexed, ${duplicates} duplicate in source, ${parseErrors} parse errors)`,
	);

	if (DRY_RUN || newRounds.length === 0) return;

	fs.mkdirSync(ROUNDS_DIR, { recursive: true });
	let completed = 0;
	let errors = 0;

	async function processRound(round: Round) {
		const file = roundFileName(round);
		fs.writeFileSync(path.resolve(ROUNDS_DIR, file), JSON.stringify(round, null, 2));
		try {
			const promptVec = await embed(round.userPrompt, apiKey!);
			appendToIndex(normalize(promptVec), `${file}:prompt`);
			const respText = round.responseSequence.slice(0, MAX_RESPONSE_CHARS);
			if (respText) {
				const respVec = await embed(respText, apiKey!);
				appendToIndex(normalize(respVec), `${file}:response`);
			}
			completed++;
			process.stderr.write(`  ✅ [${completed}/${newRounds.length}] ${file} ${round.cwd ?? ""}\n`);
		} catch (e) {
			errors++;
			process.stderr.write(`  ❌ [ERROR] ${file}: ${(e as Error).message}\n`);
		}
	}

	const queue = [...newRounds];
	const workers = Array.from({ length: Math.max(1, CONCURRENCY) }, async () => {
		while (queue.length > 0) await processRound(queue.shift()!);
	});

	const started = Date.now();
	await Promise.all(workers);
	const elapsed = ((Date.now() - started) / 1000).toFixed(1);
	const vectors = fs.existsSync(INDEX_PATH)
		? fs.readFileSync(INDEX_PATH, "utf-8").split("\n").filter(Boolean).length
		: 0;
	const roundFiles = fs.readdirSync(ROUNDS_DIR).filter((f) => f.endsWith(".json")).length;

	console.log(`\n✅ Done in ${elapsed}s. ${completed} Claude Code rounds imported, ${errors} errors.`);
	console.log(`   Index: ${vectors} vectors at ${INDEX_PATH}`);
	console.log(`   Rounds: ${roundFiles} files at ${ROUNDS_DIR}`);
}

main().catch((e) => {
	console.error("❌ Fatal:", e);
	process.exit(1);
});
