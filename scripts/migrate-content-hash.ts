/**
 * migrate-content-hash.ts — Re-hash round files using content-only hash.
 *
 * The new hash algorithm includes userPrompt + responseText + toolCalls[].arguments
 * + toolCalls[].result_full (falling back to result_summary, then empty string).
 * This ensures metadata additions (embeddings, groups, etc.) don't change filenames.
 *
 * Usage:
 *   npx tsx scripts/migrate-content-hash.ts          # perform migration
 *   npx tsx scripts/migrate-content-hash.ts --dry-run # preview only
 *   npx tsx scripts/migrate-content-hash.ts --backup  # create backup before migrating
 *
 * Override rounds dir (default ~/.pi/agent/semblr/rounds):
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/migrate-content-hash.ts
 */

import * as fs from "node:fs";
import * as os from "node:os";
import { pathToFileURL } from "node:url";
import { computeContentHash } from "../src/core/hash.ts";
import {
	filterIndexLinesExcludingFilenames,
	readIndexByFilename,
	readIndexLines,
	replaceIndexLineFilename,
	writeIndexLines,
} from "../src/core/index-io.ts";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const PI_CONFIG_DIR = process.env.PI_CODING_AGENT_DIR || `${os.homedir()}/.pi/agent`;
const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR || `${PI_CONFIG_DIR}/semblr/rounds`;
const INDEX_PATH = `${ROUNDS_DIR}/index.csv`;

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

interface RoundData {
	id?: string;
	userPrompt: string;
	responseSequence?: string;
	turnIndex?: number;
	userTimestamp?: number;
	toolCallCount?: number;
	toolCallNames?: string[];
	toolCalls?: ToolCallDetail[];
	[key: string]: unknown; // allow other metadata
}

export interface ContentHashMigrationOptions {
	args?: string[];
	roundsDir?: string;
	indexPath?: string;
	now?: () => number;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export function runContentHashMigration(options: ContentHashMigrationOptions = {}): number {
	const args = options.args ?? process.argv.slice(2);
	const dryRun = args.includes("--dry-run");
	const doBackup = args.includes("--backup");
	const roundsDir = options.roundsDir ?? ROUNDS_DIR;
	const indexPath = options.indexPath ?? INDEX_PATH;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;

	if (!fs.existsSync(roundsDir)) {
		err.error(`✗ Rounds directory does not exist: ${roundsDir}`);
		return 1;
	}

	const files = fs.readdirSync(roundsDir).filter((f) => f.endsWith(".json") && !f.includes("index"));
	if (files.length === 0) {
		out.log(`No round files found in ${roundsDir}`);
		return 0;
	}

	out.log(`Found ${files.length} round files in ${roundsDir}`);

	// Read index
	const indexMap = readIndexByFilename(indexPath);
	out.log(`Found ${indexMap.size} unique filenames in index.csv`);

	// Backup
	if (doBackup && !dryRun) {
		const backupDir = `${roundsDir}.bak-${options.now?.() ?? Date.now()}`;
		fs.cpSync(roundsDir, backupDir, { recursive: true });
		out.log(`Backup created: ${backupDir}`);
	}

	// Process rounds
	const indexUpdates: string[] = [];
	let renamed = 0;
	const _skipped = 0;
	let errors = 0;
	let hashChanged = 0;
	let hashUnchanged = 0;
	const oldRenamedFilenames = new Set<string>();

	for (const filename of files) {
		const filePath = `${roundsDir}/${filename}`;
		let data: RoundData;
		try {
			data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
		} catch (readError) {
			err.error(`✗ Failed to parse ${filename}: ${(readError as Error).message}`);
			errors++;
			continue;
		}

		const userPrompt = data.userPrompt ?? "";
		const responseText = data.responseSequence ?? "";
		const toolCalls = data.toolCalls;

		const newHash = computeContentHash(userPrompt, responseText, toolCalls);
		const _oldHash = filename.replace(/\.json$/, "");
		const newFilename = `${newHash}.json`;

		if (newFilename === filename) {
			// Hash hasn't changed — still might need to update the id field
			if (data.id !== newHash) {
				data.id = newHash;
				if (!dryRun) {
					fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
					out.log(`  ✓ updated id field: ${filename}`);
				} else {
					out.log(`  ~ would update id field: ${filename}`);
				}
			}
			hashUnchanged++;
			continue;
		}

		hashChanged++;
		const newPath = `${roundsDir}/${newFilename}`;

		if (fs.existsSync(newPath)) {
			// Collision — the new hash target already exists. This means the content
			// is equivalent to an existing round, so delete the old file.
			out.log(`  ~ collision with ${newFilename}, deleting ${filename}`);
			if (!dryRun) {
				fs.unlinkSync(filePath);
			}
		} else {
			out.log(`  → ${filename} → ${newFilename}`);
			if (!dryRun) {
				// Add the new id field
				data.id = newHash;
				fs.writeFileSync(newPath, JSON.stringify(data, null, 2));
				fs.unlinkSync(filePath);
			}
		}
		renamed++;

		// Track old filenames that were renamed (for index filtering)
		const oldFilenameForIndex = filename;
		oldRenamedFilenames.add(oldFilenameForIndex);

		// Collect index updates for this file
		const oldLines = indexMap.get(filename) ?? [];
		const newLines = oldLines.map((line) => replaceIndexLineFilename(line, newFilename));
		indexUpdates.push(...newLines);
	}

	// Write updated index
	if (indexUpdates.length > 0) {
		out.log(`\nIndex: ${indexUpdates.length} entries to update`);
		if (!dryRun) {
			// oldRenamedFilenames was populated in the rename loop above

			// Read all entries, filter out old renamed ones, add new ones
			const retained = filterIndexLinesExcludingFilenames(readIndexLines(indexPath), oldRenamedFilenames);
			writeIndexLines(indexPath, [...retained, ...indexUpdates]);
			out.log(`✓ Updated index.csv (${indexUpdates.length} entries rewritten)`);
		} else {
			out.log(`~ Would update index.csv (${indexUpdates.length} entries)`);
		}
	}

	// Summary
	out.log("\n─── Summary ───");
	out.log(`Total files:       ${files.length}`);
	out.log(`Hash changed:      ${hashChanged}`);
	out.log(`Hash unchanged:    ${hashUnchanged}`);
	out.log(`Renamed:           ${renamed}`);
	out.log(`Skipped (errors):  ${errors}`);
	out.log(`Dry run:           ${dryRun ? "yes" : "no"}`);
	return errors > 0 ? 1 : 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

if (isMainModule(import.meta.url)) {
	const exitCode = runContentHashMigration();
	if (exitCode !== 0) process.exit(exitCode);
}
