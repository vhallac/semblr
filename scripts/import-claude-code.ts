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
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { type ClaudeRound, claudeRoundFileName, parseClaudeCodeJsonl } from "../src/core/claude-code.ts";
import {
	appendVectorIndexEntry,
	loadIndexedRoundFiles as loadIndexedRoundFilesFromIndex,
} from "../src/core/index-io.ts";

const CLAUDE_PROJECTS_DIR = process.env.CLAUDE_PROJECTS_DIR || path.resolve(os.homedir(), ".claude", "projects");
const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");
const INDEX_PATH = path.resolve(ROUNDS_DIR, "index.csv");

const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";
const CONCURRENCY = Number(process.env.SEMBLR_IMPORT_CONCURRENCY || "5");
const MAX_RESPONSE_CHARS = 8000;

type Round = ClaudeRound;

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

function roundFileName(round: Pick<Round, "userPrompt" | "responseSequence">): string {
	return claudeRoundFileName(round);
}

function parseClaudeFile(filePath: string): Round[] {
	return parseClaudeCodeJsonl(fs.readFileSync(filePath, "utf-8"), {
		filePath,
		projectsDir: CLAUDE_PROJECTS_DIR,
		includeSidechains: INCLUDE_SIDECHAINS,
	});
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

function loadIndexedRoundFiles(): Set<string> {
	return loadIndexedRoundFilesFromIndex(INDEX_PATH);
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
			appendVectorIndexEntry(INDEX_PATH, normalize(promptVec), `${file}:prompt`);
			const respText = round.responseSequence.slice(0, MAX_RESPONSE_CHARS);
			if (respText) {
				const respVec = await embed(respText, apiKey!);
				appendVectorIndexEntry(INDEX_PATH, normalize(respVec), `${file}:response`);
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
