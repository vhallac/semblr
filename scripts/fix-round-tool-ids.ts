/**
 * fix-round-tool-ids.ts — Repair round files with incorrect tool-result assignments
 * caused by parallel tool invocations in pi sessions.
 *
 * Re-parses all session JSONL files with the fixed parser (ID-based matching) and
 * compares content hashes against existing round files. When the hash differs,
 * migrates the round to the corrected file.
 *
 * Usage:
 *   npx tsx scripts/fix-round-tool-ids.ts
 *   npx tsx scripts/fix-round-tool-ids.ts --dry-run
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { migrateIndexEntryLine, readIndexLines } from "../lib/index-io.ts";
import { parsePiSessionJsonl } from "../lib/pi-session.ts";
import { deriveRoundFile } from "../lib/round-capture.ts";
import { resolveScriptConfig, resolveScriptIndexPath, type ScriptConfigOptions } from "../lib/script-config.ts";

// ─────────────────────────────────────────────
// File gathering (mirrors digest-all.ts)
// ─────────────────────────────────────────────

function defaultSessionsDir(agentDir: string): string {
	return path.resolve(agentDir, "sessions");
}

function gatherSessionFiles(
	sessionsDir: string,
	deps: { fsImpl?: typeof fs } = {},
): Array<{ filePath: string; label: string }> {
	const f = deps.fsImpl ?? fs;
	if (!f.existsSync(sessionsDir)) return [];

	const sessionDirs = f
		.readdirSync(sessionsDir)
		.filter((d) => d.startsWith("--"))
		.map((d) => path.join(sessionsDir, d));

	const jsonlFiles: Array<{ filePath: string; label: string }> = [];
	for (const dir of sessionDirs) {
		const label = path.basename(dir);
		const files = f.readdirSync(dir).filter((fn) => fn.endsWith(".jsonl"));
		for (const fn of files) {
			jsonlFiles.push({ filePath: path.join(dir, fn), label });
		}
	}

	jsonlFiles.sort((a, b) => a.filePath.localeCompare(b.filePath));
	return jsonlFiles;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface FixRoundToolIdsOptions extends ScriptConfigOptions {
	sessionsDir?: string;
	roundsDir?: string;
	indexPath?: string;
	dryRun?: boolean;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: typeof fs;
}

export async function runFixRoundToolIds(options: FixRoundToolIdsOptions = {}): Promise<number> {
	const config = resolveScriptConfig(options);
	const sessionsDir = options.sessionsDir ?? defaultSessionsDir(config.agentDir);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const indexPath = resolveScriptIndexPath(config, roundsDir, options.indexPath);
	const dryRun = options.dryRun ?? false;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const f = options.fsImpl ?? fs;

	const jsonlFiles = gatherSessionFiles(sessionsDir, { fsImpl: f });
	const sessionsCount = new Set(jsonlFiles.map((j) => j.label)).size;
	out.log(`📂 Found ${jsonlFiles.length} session files across ${sessionsCount} directories`);
	if (dryRun) out.log("🔍 DRY RUN — no files will be modified\n");
	else out.log("");

	// Build a round-id → filename lookup from all existing round files.
	// We match by the round's `id` field (the session message ID from pi),
	// which is unique per round and stored in the round file.
	const existingFiles = f.readdirSync(roundsDir).filter((rf) => rf.endsWith(".json") && !rf.startsWith("index"));

	const roundIdToFile = new Map<string, string>();
	for (const filename of existingFiles) {
		try {
			const data = JSON.parse(f.readFileSync(path.join(roundsDir, filename), "utf-8"));
			if (typeof data.id === "string" && data.id) {
				roundIdToFile.set(data.id, filename);
			}
		} catch {
			// Skip corrupt files
		}
	}

	let totalParsed = 0;
	let totalChanged = 0;
	let totalUnchanged = 0;
	let totalErrors = 0;

	for (const { filePath, label } of jsonlFiles) {
		const fileId = `${label}/${path.basename(filePath)}`;
		try {
			const raw = f.readFileSync(filePath, "utf-8");
			const rounds = parsePiSessionJsonl(raw, { sessionLabel: label, skipShortFinalResponse: true });

			let changed = 0;
			let unchanged = 0;

			for (const round of rounds) {
				totalParsed++;
				const newHash = deriveRoundFile(round.userPrompt, round.responseSequence, round.toolCalls).fileName;

				if (f.existsSync(path.join(roundsDir, newHash))) {
					// Already correct — file with this hash exists
					unchanged++;
					totalUnchanged++;
					continue;
				}

				// Look up the broken file by round id
				const brokenFile = round.id ? roundIdToFile.get(round.id) : undefined;

				if (brokenFile) {
					changed++;
					totalChanged++;

					if (!dryRun) {
						// Migrate index entries from broken hash to corrected hash.
						// Read/write through the provided fs impl so tests can inject a mock.
						if (f.existsSync(indexPath)) {
							const lines = f.readFileSync(indexPath, "utf-8").trim().split("\n").filter(Boolean);
							const migrated = lines.map((line: string) => migrateIndexEntryLine(line, brokenFile, newHash));
							f.writeFileSync(indexPath, migrated.join("\n") + (migrated.length > 0 ? "\n" : ""));
						}

						// Write corrected round file
						f.writeFileSync(path.join(roundsDir, newHash), JSON.stringify(round, null, 2));

						// Delete old broken file
						f.unlinkSync(path.join(roundsDir, brokenFile));

						// Update the lookup map for this round id
						if (round.id) roundIdToFile.set(round.id, newHash);
					}
				} else {
					unchanged++;
					totalUnchanged++;
				}
			}

			if (changed > 0) {
				out.log(`  🔧 ${fileId}: ${changed}/${changed + unchanged} rounds fixed`);
			}
		} catch (e) {
			totalErrors++;
			err.error(`  ❌ ${fileId}: ${(e as Error).message}`);
		}
	}

	// Count how many index lines were affected
	const indexLines = readIndexLines(indexPath).length;

	out.log(
		`\n${dryRun ? "🔍 Would fix" : "✅ Fixed"} ${totalChanged} rounds (${totalUnchanged} unchanged, ${totalParsed} total parsed)`,
	);
	out.log(`   Index: ${indexLines} vectors at ${indexPath}`);
	if (totalErrors > 0) out.log(`   Errors: ${totalErrors}`);

	if (dryRun && totalChanged > 0) {
		out.log("\n   Run without --dry-run to apply fixes.");
	}

	return totalErrors > 0 ? 1 : 0;
}

// ─────────────────────────────────────────────
// CLI entry point
// ─────────────────────────────────────────────

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const dryRun = process.argv.includes("--dry-run");
	const exitCode = await runFixRoundToolIds({ dryRun });
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((err) => {
		console.error("❌ Fatal:", err);
		process.exit(1);
	});
}
