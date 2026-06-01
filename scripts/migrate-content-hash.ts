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

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

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

// ─────────────────────────────────────────────
// Hash computation (mirrors src/semblr.ts)
// ─────────────────────────────────────────────

function computeContentHash(userPrompt: string, responseText: string, toolCalls?: ToolCallDetail[]): string {
	const parts: string[] = [userPrompt, responseText];
	if (toolCalls) {
		for (const tc of toolCalls) {
			parts.push(tc.arguments);
			parts.push(tc.result_full ?? tc.result_summary ?? "");
		}
	}
	return crypto.createHash("md5").update(parts.join("")).digest("hex");
}

// ─────────────────────────────────────────────
// Index helpers
// ─────────────────────────────────────────────

function readIndex(): Map<string, string[]> {
	// Returns: oldFilename → [indexLines...]
	const index = new Map<string, string[]>();
	if (!fs.existsSync(INDEX_PATH)) return index;
	const lines = fs
		.readFileSync(INDEX_PATH, "utf-8")
		.split("\n")
		.filter((l) => l.trim());
	for (const line of lines) {
		// Format: <b64vector>,<filename>:<type>
		const commaIdx = line.indexOf(",");
		if (commaIdx === -1) continue;
		const entry = line.slice(commaIdx + 1);
		// entry is like "abc123.json:prompt" or "abc123.json:response"
		const colonIdx = entry.lastIndexOf(":");
		if (colonIdx === -1) continue;
		const oldFilename = entry.slice(0, colonIdx);
		if (!index.has(oldFilename)) index.set(oldFilename, []);
		index.get(oldFilename)?.push(line);
	}
	return index;
}

function writeIndex(entries: string[]): void {
	fs.writeFileSync(INDEX_PATH, `${entries.join("\n")}\n`);
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

function main(): void {
	const args = process.argv.slice(2);
	const dryRun = args.includes("--dry-run");
	const doBackup = args.includes("--backup");

	if (!fs.existsSync(ROUNDS_DIR)) {
		console.error(`✗ Rounds directory does not exist: ${ROUNDS_DIR}`);
		process.exit(1);
	}

	const files = fs.readdirSync(ROUNDS_DIR).filter((f) => f.endsWith(".json") && !f.includes("index"));
	if (files.length === 0) {
		console.log(`No round files found in ${ROUNDS_DIR}`);
		process.exit(0);
	}

	console.log(`Found ${files.length} round files in ${ROUNDS_DIR}`);

	// Read index
	const indexMap = readIndex();
	console.log(`Found ${indexMap.size} unique filenames in index.csv`);

	// Backup
	if (doBackup && !dryRun) {
		const backupDir = `${ROUNDS_DIR}.bak-${Date.now()}`;
		fs.cpSync(ROUNDS_DIR, backupDir, { recursive: true });
		console.log(`Backup created: ${backupDir}`);
	}

	// Process rounds
	const indexUpdates: string[] = [];
	let renamed = 0;
	const skipped = 0;
	let errors = 0;
	let hashChanged = 0;
	let hashUnchanged = 0;
	const oldRenamedFilenames = new Set<string>();

	for (const filename of files) {
		const filePath = `${ROUNDS_DIR}/${filename}`;
		let data: RoundData;
		try {
			data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
		} catch (err) {
			console.error(`✗ Failed to parse ${filename}: ${(err as Error).message}`);
			errors++;
			continue;
		}

		const userPrompt = data.userPrompt ?? "";
		const responseText = data.responseSequence ?? "";
		const toolCalls = data.toolCalls;

		const newHash = computeContentHash(userPrompt, responseText, toolCalls);
		const oldHash = filename.replace(/\.json$/, "");
		const newFilename = `${newHash}.json`;

		if (newFilename === filename) {
			// Hash hasn't changed — still might need to update the id field
			if (data.id !== newHash) {
				data.id = newHash;
				if (!dryRun) {
					fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
					console.log(`  ✓ updated id field: ${filename}`);
				} else {
					console.log(`  ~ would update id field: ${filename}`);
				}
			}
			hashUnchanged++;
			continue;
		}

		hashChanged++;
		const newPath = `${ROUNDS_DIR}/${newFilename}`;

		if (fs.existsSync(newPath)) {
			// Collision — the new hash target already exists. This means the content
			// is equivalent to an existing round, so delete the old file.
			console.log(`  ~ collision with ${newFilename}, deleting ${filename}`);
			if (!dryRun) {
				fs.unlinkSync(filePath);
			}
		} else {
			console.log(`  → ${filename} → ${newFilename}`);
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
		const newLines = oldLines.map((line) => {
			// Replace old filename prefix with new filename
			const commaIdx = line.indexOf(",");
			const rest = line.slice(commaIdx + 1);
			const namePart = rest.replace(/^[^:]+/, newFilename);
			return line.slice(0, commaIdx + 1) + namePart;
		});
		indexUpdates.push(...newLines);
	}

	// Write updated index
	if (indexUpdates.length > 0) {
		console.log(`\nIndex: ${indexUpdates.length} entries to update`);
		if (!dryRun) {
			// oldRenamedFilenames was populated in the rename loop above

			// Read all entries, filter out old renamed ones, add new ones
			const allLines = fs
				.readFileSync(INDEX_PATH, "utf-8")
				.split("\n")
				.filter((l) => l.trim());
			const retained = allLines.filter((line) => {
				const commaIdx = line.indexOf(",");
				if (commaIdx === -1) return true;
				const entry = line.slice(commaIdx + 1);
				const colonIdx = entry.lastIndexOf(":");
				if (colonIdx === -1) return true;
				const filename = entry.slice(0, colonIdx);
				// Remove entries for renamed files; keep entries for unchanged files
				return !oldRenamedFilenames.has(filename);
			});
			writeIndex([...retained, ...indexUpdates]);
			console.log(`✓ Updated index.csv (${indexUpdates.length} entries rewritten)`);
		} else {
			console.log(`~ Would update index.csv (${indexUpdates.length} entries)`);
		}
	}

	// Summary
	console.log("\n─── Summary ───");
	console.log(`Total files:       ${files.length}`);
	console.log(`Hash changed:      ${hashChanged}`);
	console.log(`Hash unchanged:    ${hashUnchanged}`);
	console.log(`Renamed:           ${renamed}`);
	console.log(`Skipped (errors):  ${errors}`);
	console.log(`Dry run:           ${dryRun ? "yes" : "no"}`);
}

main();
