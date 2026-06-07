/**
 * migrate-rounds.ts — Add missing `responseSegments` to existing round files.
 *
 * Round files created before step2 do not have the `responseSegments` field,
 * which allows `get_round_details()` to show tool call positions interleaved
 * with the assistant's text response.
 *
 * This script reconstructs `responseSegments` from the existing `responseSequence`
 * (all text at once) and `toolCalls[]` (all tool calls appended at end). While
 * exact positions are lost for old rounds, the interleaved markers are still
 * more useful than the previous flat "ALL REDACTED" blob.
 *
 * The hash (file name) does NOT change — we add the field in-place.
 *
 * Usage:
 *   npx tsx scripts/migrate-rounds.ts
 *
 * Override rounds dir (default ~/.pi/agent/semblr/rounds):
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/migrate-rounds.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { resolveScriptConfig, type ScriptConfigOptions } from "../lib/script-config.ts";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface ResponseSegment {
	type: "text" | "toolCall";
	text?: string;
	toolCallIndex?: number;
}

interface ToolCallDetail {
	index: number;
	name: string;
	arguments: string;
	result_summary: string;
}

interface Round {
	userPrompt: string;
	responseSequence: string;
	responseSegments?: ResponseSegment[];
	toolCalls?: ToolCallDetail[];
	[key: string]: unknown;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

export interface MigrateRoundsOptions extends ScriptConfigOptions {
	roundsDir?: string;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error" | "warn">;
}

export function runResponseSegmentsMigration(options: MigrateRoundsOptions = {}): number {
	const config = resolveScriptConfig(options);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;

	if (!fs.existsSync(roundsDir)) {
		err.error(`❌ Rounds directory not found: ${roundsDir}`);
		return 1;
	}

	const files = fs.readdirSync(roundsDir).filter((f) => f.endsWith(".json"));

	let migrated = 0;
	let skipped = 0;

	for (const file of files) {
		const filePath = path.resolve(roundsDir, file);

		let data: Round;
		try {
			const raw = fs.readFileSync(filePath, "utf-8");
			data = JSON.parse(raw);
		} catch {
			err.warn(`⚠ Skipping unparseable: ${file}`);
			continue;
		}

		// Already has responseSegments
		if (data.responseSegments && Array.isArray(data.responseSegments) && data.responseSegments.length > 0) {
			skipped++;
			continue;
		}

		// Reconstruct from existing fields
		const segments: ResponseSegment[] = [];

		// Add all text as one block (exact positions lost, but better than nothing)
		if (data.responseSequence?.trim()) {
			segments.push({ type: "text", text: data.responseSequence });
		}

		// Add tool call markers at end
		if (data.toolCalls && Array.isArray(data.toolCalls)) {
			for (let i = 0; i < data.toolCalls.length; i++) {
				segments.push({ type: "toolCall", toolCallIndex: i });
			}
		}

		// Only write if we have something useful (skip empty rounds)
		if (segments.length === 0) {
			err.warn(`⚠ Skipping empty: ${file}`);
			continue;
		}

		data.responseSegments = segments;

		// Write back with 2-space indent
		fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
		migrated++;
	}

	out.log(`\n✅ Done. ${migrated} round files migrated, ${skipped} already had responseSegments.`);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

if (isMainModule(import.meta.url)) {
	const exitCode = runResponseSegmentsMigration();
	if (exitCode !== 0) process.exit(exitCode);
}
