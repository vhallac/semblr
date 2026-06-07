/**
 * erase-short-embeddings.ts — Remove embeddings for short-prompt rounds from index.csv
 *
 * Short prompts (< RELEVANCE_LIST_MIN_WORDS words, default 20) produce noisy embeddings
 * that waste API credits, pollute the vector index, and cause false-positive search matches.
 *
 * This script:
 *   1. Reads all :prompt rows from index.csv
 *   2. For each, loads the round JSON and checks userPrompt word count
 *   3. If word count < 20, deletes BOTH :prompt and :response rows from index.csv
 *   4. Deletes promptEmbedding field from the round JSON file
 *
 * Usage:
 *   npx tsx scripts/erase-short-embeddings.ts          # perform removal
 *   npx tsx scripts/erase-short-embeddings.ts --dry-run # preview only
 *   npx tsx scripts/erase-short-embeddings.ts --backup  # create backup before modifying
 *
 * Override rounds dir (default ~/.pi/agent/semblr/rounds):
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/erase-short-embeddings.ts
 */

import * as fs from "node:fs";
import { pathToFileURL } from "node:url";
import { readIndexLines, writeIndexLines } from "../lib/index-io.ts";
import { resolveScriptConfig, resolveScriptIndexPath, type ScriptConfigOptions } from "../lib/script-config.ts";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const MIN_WORDS = Number.parseInt(process.env.RELEVANCE_LIST_MIN_WORDS ?? "20", 10);

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function countWords(text: string): number {
	return text.split(/\s+/).filter((w) => w.length > 0).length;
}

interface IndexEntry {
	vectorB64: string;
	suffix: string; // :prompt, :response, or :round
	roundFile: string; // e.g. "abc123.json"
}

function parseLine(line: string): IndexEntry | null {
	const commaIdx = line.indexOf(",");
	if (commaIdx === -1) return null;
	const vectorB64 = line.slice(0, commaIdx);
	const filePath = line.slice(commaIdx + 1);

	// filePath is "abc123.json:prompt" or "abc123.json:response" or "abc123.json:round"
	const colonIdx = filePath.lastIndexOf(":");
	if (colonIdx === -1) return null;
	const roundFile = filePath.slice(0, colonIdx);
	const suffix = filePath.slice(colonIdx);

	return { vectorB64, suffix, roundFile };
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface EraseShortEmbeddingsOptions extends ScriptConfigOptions {
	args?: string[];
	roundsDir?: string;
	indexPath?: string;
	minWords?: number;
	now?: () => number;
	stdout?: Pick<typeof console, "log" | "error">;
}

export function runEraseShortEmbeddings(options: EraseShortEmbeddingsOptions = {}): number {
	const args = options.args ?? [];
	const dryRun = args.includes("--dry-run");
	const doBackup = args.includes("--backup");
	const verbose = args.includes("--verbose");
	const config = resolveScriptConfig(options);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const indexPath = resolveScriptIndexPath(config, roundsDir, options.indexPath);
	const minWords = options.minWords ?? MIN_WORDS;
	const out = options.stdout ?? console;

	if (!fs.existsSync(roundsDir)) {
		out.error(`✗ Rounds directory does not exist: ${roundsDir}`);
		return 1;
	}

	const indexLines = readIndexLines(indexPath);
	if (indexLines.length === 0) {
		console.log("No index entries found. Nothing to do.");
		return 0;
	}

	out.log(`Found ${indexLines.length} index entries (${indexPath})`);

	// Backup
	if (doBackup && !dryRun) {
		const backupPath = `${indexPath}.bak-${options.now?.() ?? Date.now()}`;
		fs.copyFileSync(indexPath, backupPath);
		out.log(`Backup created: ${backupPath}`);
	}

	// Parse all entries
	const entries = indexLines.map(parseLine).filter((e): e is IndexEntry => e !== null);
	out.log(`Parsed ${entries.length} valid index entries`);

	// Find which round files have short prompts
	const shortRoundFiles = new Set<string>();
	const shortRoundPaths = new Set<string>();

	for (const entry of entries) {
		if (entry.suffix !== ":prompt") continue;
		if (shortRoundFiles.has(entry.roundFile)) continue; // already checked

		const roundPath = `${roundsDir}/${entry.roundFile}`;
		if (!fs.existsSync(roundPath)) {
			if (verbose) console.log(`  ✗ Round file missing: ${entry.roundFile}`);
			continue;
		}

		try {
			const data = JSON.parse(fs.readFileSync(roundPath, "utf-8"));
			const promptText = data.userPrompt ?? "";
			const wordCount = countWords(promptText);

			if (wordCount < minWords) {
				shortRoundFiles.add(entry.roundFile);
				shortRoundPaths.add(roundPath);
				if (verbose) {
					out.log(`  ✓ Short prompt (${wordCount} words): ${entry.roundFile} — "${promptText.slice(0, 80)}..."`);
				}
			}
		} catch {
			out.error(`  ✗ Failed to parse round file: ${entry.roundFile}`);
		}
	}

	if (shortRoundFiles.size === 0) {
		console.log("No short-prompt rounds found. Nothing to do.");
		return 0;
	}

	out.log(`\nFound ${shortRoundFiles.size} short-prompt round files to clean`);

	// Filter out entries for short-prompt rounds (both :prompt and :response)
	const retained = indexLines.filter((line) => {
		const entry = parseLine(line);
		if (!entry) return true; // keep unparseable lines
		return !shortRoundFiles.has(entry.roundFile);
	});

	const removed = indexLines.length - retained.length;
	out.log(`Would remove ${removed} index entries (${retained.length} retained)`);

	if (!dryRun) {
		writeIndexLines(indexPath, retained);
		out.log(`✓ Updated index.csv (${removed} entries removed)`);

		// Delete promptEmbedding field from round JSON files
		let cleanedFiles = 0;
		for (const roundPath of shortRoundPaths) {
			try {
				const data = JSON.parse(fs.readFileSync(roundPath, "utf-8"));
				if (data.promptEmbedding) {
					delete data.promptEmbedding;
					fs.writeFileSync(`${roundPath}.tmp.${process.pid}`, JSON.stringify(data, null, 2));
					fs.renameSync(`${roundPath}.tmp.${process.pid}`, roundPath);
					cleanedFiles++;
				}
			} catch {
				out.error(`  ✗ Failed to clean round file: ${roundPath}`);
			}
		}
		out.log(`✓ Cleaned promptEmbedding from ${cleanedFiles} round files`);
	} else {
		out.log(`~ Dry run: would remove ${removed} index entries`);
		out.log(`~ Dry run: would clean promptEmbedding from ${shortRoundPaths.size} round files`);
	}

	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

if (isMainModule(import.meta.url)) {
	const exitCode = runEraseShortEmbeddings({ args: process.argv.slice(2) });
	if (exitCode !== 0) process.exit(exitCode);
}
