/**
 * migrate-model-column.ts — Add embedding model column to existing index.csv entries.
 *
 * Entries created before issue #62 was implemented have only 2 columns:
 *   base64url(vector),filePath
 *
 * This script adds the configured Semblr embedding model as a 3rd column to
 * entries that don't already have one.
 *
 * Usage:
 *   npx tsx scripts/migrate-model-column.ts
 *
 * Dry-run (no changes):
 *   npx tsx scripts/migrate-model-column.ts --dry-run
 *
 * Custom rounds directory:
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/migrate-model-column.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { LEGACY_INDEX_FILENAME } from "../lib/index-shards.ts";
import {
	reportForeignShards,
	resolveScriptConfig,
	resolveScriptIndexContext,
	type ScriptConfigOptions,
} from "../lib/script-config.ts";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/**
 * Check if a CSV line already has a 3rd column (model).
 * Format: base64url(vector),filePath[,model]
 */
function hasModelColumn(line: string): boolean {
	const firstComma = line.indexOf(",");
	if (firstComma === -1) return false;
	const rest = line.slice(firstComma + 1);
	return rest.includes(",");
}

/**
 * Add the configured embedding model as a 3rd column.
 */
function addModelColumn(line: string, model: string): string {
	if (hasModelColumn(line)) return line;
	return `${line},${model}`;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface MigrateModelColumnOptions extends ScriptConfigOptions {
	indexPath?: string;
	dryRun?: boolean;
	backup?: boolean;
	/** Hostname override for tests; defaults to os.hostname(). */
	hostname?: string;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error" | "warn">;
}

export function runModelColumnMigration(options: MigrateModelColumnOptions = {}): number {
	const config = resolveScriptConfig(options);
	const indexPath = options.indexPath ?? process.env.SEMBLR_INDEX_PATH ?? config.indexPath;
	const model = config.embeddingModel;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	// Issue #149: the index is a set of per-host shards. The migration rewrites
	// this host's own shard only (D-005) — a foreign shard keeps its rows and
	// is updated when that host runs this script.
	const index = resolveScriptIndexContext(config, config.roundsDir, indexPath, { hostname: options.hostname });

	const foreignShards = index.foreignShards();
	const legacyPath = path.join(index.shardRoot, LEGACY_INDEX_FILENAME);
	const hasIndexArtifact = fs.existsSync(index.ownShardPath) || fs.existsSync(legacyPath);
	if (!hasIndexArtifact && foreignShards.length === 0) {
		err.error(`❌ Index file not found: ${indexPath}`);
		return 1;
	}

	const lines = index.readRewriteRows();
	if (lines.length === 0) {
		out.log(
			hasIndexArtifact
				? "ℹ Index file is empty. Nothing to migrate."
				: "ℹ This host's shard holds no rows. Nothing to migrate.",
		);
		reportForeignShards(index, out);
		return 0;
	}
	const migrated: number[] = [];

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i].trim();
		if (!line) continue;
		if (!hasModelColumn(line)) {
			migrated.push(i);
		}
	}

	if (migrated.length === 0) {
		out.log("✅ All entries already have a model column. Nothing to migrate.");
		return 0;
	}

	out.log(`📋 Found ${migrated.length} entries without a model column:`);
	for (const idx of migrated) {
		out.log(`   L${idx + 1}: ${lines[idx]}`);
	}
	out.log(`\n📝 Model to add: ${model}`);

	if (options.dryRun) {
		out.log("\n🚫 Dry-run mode: no changes written.");
		out.log(`   Run without --dry-run to apply.`);
		return 0;
	}

	// Create backup if requested
	if (options.backup) {
		// Claim the legacy `index.csv` first so the backup covers the rows the
		// migration publishes; the backup name is excluded from shard discovery.
		index.adopt();
		if (fs.existsSync(index.ownShardPath)) {
			const backupPath = `${index.ownShardPath}.bak.${Date.now()}`;
			fs.copyFileSync(index.ownShardPath, backupPath);
			out.log(`\n💾 Backup saved to: ${backupPath}`);
		}
	}

	// Apply migration
	for (const idx of migrated) {
		lines[idx] = addModelColumn(lines[idx], model);
	}
	// Issue #149: publish to the host's own shard (atomic tmp+rename); a
	// foreign shard is never rewritten here (D-005).
	index.writeOwnLines(lines);

	out.log(`\n✅ Done. ${migrated.length} entries migrated.`);
	reportForeignShards(index, out);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

if (isMainModule(import.meta.url)) {
	const dryRun = process.argv.includes("--dry-run");
	const backup = process.argv.includes("--backup");
	const exitCode = runModelColumnMigration({ dryRun, backup });
	if (exitCode !== 0) process.exit(exitCode);
}
