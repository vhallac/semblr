/**
 * prune-marker-duplicates.ts — Store hygiene for pre-fix marker duplicates (F1/F2,
 * PR !131 reconciliation unit-003).
 *
 * Round files written before the shared round derivation (F1) stored the
 * followup marker in `responseSequence` and hashed the raw text, so a
 * marker-bearing round landed under a filename that diverges from the live
 * write path. When backfill later recovers the same round, both copies exist:
 * a pre-fix marker file (`needsFollowup: false` or absent) and a clean
 * stripped-hash copy (`needsFollowup: true`).
 *
 * This script deletes the pre-fix copy — and only the pre-fix copy: a marker
 * file is removed solely when its derived (marker-stripped) filename exists
 * on disk AND that twin carries `needsFollowup: true`. Marker files whose
 * twin is missing are left untouched (backfill's job, not pruning's).
 *
 * Dry-run by default. Use --execute to actually delete.
 *
 * Usage:
 *   npx tsx scripts/prune-marker-duplicates.ts [--execute]
 *
 * Override rounds dir (default from Semblr config):
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/prune-marker-duplicates.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { deriveRoundFile } from "../lib/round-capture.ts";
import { resolveScriptConfig, type ScriptConfigOptions } from "../lib/script-config.ts";

interface MarkerRound {
	userPrompt: string;
	responseSequence: string;
	toolCalls?: { arguments: string; result_summary?: string; result_full?: string }[];
	needsFollowup?: boolean;
	[key: string]: unknown;
}

export interface PruneScanResult {
	scanned: number;
	markerRounds: number;
	duplicates: string[];
	twinMissing: string[];
	twinNotFollowup: string[];
	unparseable: string[];
}

export interface PruneOptions extends ScriptConfigOptions {
	roundsDir?: string;
	execute?: boolean;
	fsMod?: Pick<typeof fs, "existsSync" | "readFileSync" | "readdirSync" | "unlinkSync">;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error" | "warn">;
}

export function scanMarkerDuplicates(
	roundsDir: string,
	fsMod: Pick<typeof fs, "existsSync" | "readFileSync" | "readdirSync">,
): PruneScanResult {
	const result: PruneScanResult = {
		scanned: 0,
		markerRounds: 0,
		duplicates: [],
		twinMissing: [],
		twinNotFollowup: [],
		unparseable: [],
	};
	if (!fsMod.existsSync(roundsDir)) return result;

	const files = fsMod.readdirSync(roundsDir).filter((f) => f.endsWith(".json"));
	const present = new Set(files);

	for (const file of files) {
		let data: MarkerRound;
		try {
			data = JSON.parse(fsMod.readFileSync(path.resolve(roundsDir, file), "utf-8")) as MarkerRound;
		} catch {
			result.unparseable.push(file);
			continue;
		}
		result.scanned++;

		const derived = deriveRoundFile(data.userPrompt ?? "", data.responseSequence ?? "", data.toolCalls);
		if (!derived.needsFollowup) continue;
		result.markerRounds++;

		if (derived.fileName === file) continue;
		if (!present.has(derived.fileName)) {
			result.twinMissing.push(file);
			continue;
		}
		let twin: MarkerRound;
		try {
			twin = JSON.parse(fsMod.readFileSync(path.resolve(roundsDir, derived.fileName), "utf-8")) as MarkerRound;
		} catch {
			result.twinMissing.push(file);
			continue;
		}
		if (twin.needsFollowup === true) {
			result.duplicates.push(file);
		} else {
			result.twinNotFollowup.push(file);
		}
	}
	return result;
}

export function runPruneMarkerDuplicates(options: PruneOptions = {}): number {
	const config = resolveScriptConfig(options);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const fsMod = options.fsMod ?? fs;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;

	if (!fsMod.existsSync(roundsDir)) {
		err.error(`❌ Rounds directory not found: ${roundsDir}`);
		return 1;
	}

	const scan = scanMarkerDuplicates(roundsDir, fsMod);

	out.log(
		`Scanned ${scan.scanned} round files: ${scan.markerRounds} marker rounds, ` +
			`${scan.duplicates.length} duplicates (pre-fix copy + stripped-hash twin), ` +
			`${scan.twinMissing.length} twin-missing, ${scan.twinNotFollowup.length} twin-without-flag, ` +
			`${scan.unparseable.length} unparseable.`,
	);

	if (scan.duplicates.length === 0) {
		out.log("✅ Store is clean — nothing to prune.");
		return 0;
	}

	if (!options.execute) {
		out.log("Dry run — pass --execute to delete these pre-fix copies:");
		for (const f of scan.duplicates) out.log(`  ${f}`);
		return 0;
	}

	let deleted = 0;
	for (const f of scan.duplicates) {
		try {
			fsMod.unlinkSync(path.resolve(roundsDir, f));
			deleted++;
		} catch (e) {
			err.warn(`⚠ Failed to delete ${f}: ${String(e)}`);
		}
	}
	out.log(`✅ Deleted ${deleted}/${scan.duplicates.length} pre-fix marker duplicates.`);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

if (isMainModule(import.meta.url)) {
	const exitCode = runPruneMarkerDuplicates({ execute: process.argv.includes("--execute") });
	if (exitCode !== 0) process.exit(exitCode);
}
