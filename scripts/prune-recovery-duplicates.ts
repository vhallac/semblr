/**
 * prune-recovery-duplicates.ts — Detect and remove duplicate rounds created by
 * the recovery/backfill hash-mismatch bug.
 *
 * Background: the backfill parse path trimmed tool-result text while the live
 * capture path did not, so rounds containing tool results with trailing
 * whitespace hashed differently on recovery and were re-recovered as duplicate
 * round files under new IDs. The capture path has been fixed; this tool cleans
 * up the resulting duplicates (recover-then-keep the live/original round).
 *
 * Detection: two rounds are duplicates when their canonical content matches —
 * userPrompt, responseSequence, and tool-call arguments/results (all compared
 * trailing-whitespace-insensitively, since that divergence was the bug).
 * Within each duplicate group the keeper is a non-recovered round (earliest
 * first, falling back to the earliest recovered round). Only recovered rounds
 * newer than the cutoff (default: 7 days) are ever pruned.
 *
 * Usage:
 *   npx tsx scripts/prune-recovery-duplicates.ts            # dry run (default)
 *   npx tsx scripts/prune-recovery-duplicates.ts --apply    # actually delete
 *   npx tsx scripts/prune-recovery-duplicates.ts --days=14  # widen the window
 *
 * Override rounds dir (default from config, ~/.pi/agent/semblr/rounds):
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/prune-recovery-duplicates.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { bm25IndexPathForRoundsDir, buildBm25Index, roundTextForBm25, writeBm25Index } from "../lib/bm25-index.ts";
import { filterIndexLinesExcludingFilenames, readIndexLines, writeIndexLines } from "../lib/index-io.ts";
import { resolveScriptConfig, resolveScriptIndexPath, type ScriptConfigOptions } from "../lib/script-config.ts";
import { toolIndexPathForRoundsDir } from "../lib/search-tools.ts";

interface ToolCallDetail {
	name?: string;
	arguments?: string;
	result_summary?: string;
	result_full?: string;
}

interface RoundLike {
	userPrompt?: string;
	responseSequence?: string;
	toolCalls?: ToolCallDetail[];
	recovered?: boolean;
}

export interface RoundRecord {
	filename: string;
	key: string;
	recovered: boolean;
	mtimeMs: number;
}

export interface PrunePlan {
	keeper: string;
	prunes: string[];
}

export function canonicalDuplicateKey(round: RoundLike): string {
	const tools = (round.toolCalls ?? []).map((tc) =>
		[tc.name ?? "", tc.arguments ?? "", (tc.result_full ?? tc.result_summary ?? "").trimEnd()].join("\u0000"),
	);
	return JSON.stringify([(round.userPrompt ?? "").trim(), (round.responseSequence ?? "").trim(), tools]);
}

export function selectPrunes(records: RoundRecord[], cutoffMs: number): PrunePlan[] {
	const groups = new Map<string, RoundRecord[]>();
	for (const rec of records) {
		const group = groups.get(rec.key) ?? [];
		group.push(rec);
		groups.set(rec.key, group);
	}

	const plans: PrunePlan[] = [];
	for (const group of groups.values()) {
		if (group.length < 2) continue;
		// Prefer keeping a live (non-recovered) round; earliest file wins.
		const sorted = [...group].sort((a, b) => a.mtimeMs - b.mtimeMs);
		const live = sorted.filter((r) => !r.recovered);
		const keeper = (live.length > 0 ? live : sorted)[0];
		const prunes = sorted
			.filter((r) => r.filename !== keeper.filename && r.recovered && r.mtimeMs >= cutoffMs)
			.map((r) => r.filename);
		if (prunes.length > 0) plans.push({ keeper: keeper.filename, prunes });
	}
	return plans;
}

function parseDaysArg(args: string[]): number {
	for (const arg of args) {
		const m = /^--days=(\d+)$/.exec(arg);
		if (m) return Number(m[1]);
	}
	return 7;
}

export interface PruneOptions extends ScriptConfigOptions {
	args?: string[];
	roundsDir?: string;
	indexPath?: string;
	cutoffMs?: number;
	out?: (line: string) => void;
}

export async function pruneRecoveryDuplicates(options: PruneOptions = {}): Promise<void> {
	const out = options.out ?? console.log;
	const args = options.args ?? process.argv.slice(2);
	const apply = args.includes("--apply");
	const cutoffMs = options.cutoffMs ?? Date.now() - parseDaysArg(args) * 24 * 60 * 60 * 1000;

	const config = resolveScriptConfig(options);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const indexPath = resolveScriptIndexPath(config, roundsDir, options.indexPath);

	const records: RoundRecord[] = [];
	for (const entry of fs.readdirSync(roundsDir)) {
		if (!entry.endsWith(".json")) continue;
		const filePath = path.join(roundsDir, entry);
		const stat = fs.statSync(filePath);
		let round: RoundLike;
		try {
			round = JSON.parse(fs.readFileSync(filePath, "utf-8"));
		} catch {
			out(`SKIP (unparseable): ${entry}`);
			continue;
		}
		records.push({
			filename: entry,
			key: canonicalDuplicateKey(round),
			recovered: round.recovered === true,
			mtimeMs: stat.mtimeMs,
		});
	}

	const plans = selectPrunes(records, cutoffMs);
	const allPrunes = new Set(plans.flatMap((p) => p.prunes));

	out(`Rounds scanned: ${records.length}`);
	out(`Duplicate groups with prunable recovered twins: ${plans.length}`);
	out(`Rounds to prune: ${allPrunes.size} (cutoff: ${new Date(cutoffMs).toISOString()})`);
	out(apply ? "Mode: APPLY" : "Mode: DRY RUN (pass --apply to delete)");

	for (const plan of plans) {
		out(`\nkeeper: ${plan.keeper}`);
		for (const victim of plan.prunes) out(`  prune: ${victim}`);
	}

	if (!apply || allPrunes.size === 0) return;

	for (const filename of allPrunes) {
		fs.unlinkSync(path.join(roundsDir, filename));
	}

	for (const indexFile of [indexPath, toolIndexPathForRoundsDir(roundsDir)]) {
		if (!fs.existsSync(indexFile)) continue;
		const lines = readIndexLines(indexFile);
		const kept =
			indexFile === indexPath
				? filterIndexLinesExcludingFilenames(lines, allPrunes)
				: // Tools fulltext index lines start with "<filename>,<toolIndex>,...".
					lines.filter((line) => !allPrunes.has(line.slice(0, line.indexOf(","))));
		if (kept.length !== lines.length) {
			writeIndexLines(indexFile, kept);
			out(`Index cleaned: ${indexFile} (${lines.length - kept.length} lines removed)`);
		}
	}

	const bm25Path = bm25IndexPathForRoundsDir(roundsDir);
	const documents = fs
		.readdirSync(roundsDir)
		.filter((f) => f.endsWith(".json"))
		.map((f) => ({
			fileName: f,
			text: roundTextForBm25(JSON.parse(fs.readFileSync(path.join(roundsDir, f), "utf-8"))),
		}));
	writeBm25Index(bm25Path, buildBm25Index(documents), fs);
	out(`BM25 index rebuilt: ${documents.length} rounds at ${bm25Path}`);
	out(`Pruned ${allPrunes.size} duplicate round files.`);
}

const isMain = process.argv[1] !== undefined && pathToFileURL(process.argv[1]).href === import.meta.url;
if (isMain) {
	pruneRecoveryDuplicates().catch((err) => {
		console.error(err);
		process.exitCode = 1;
	});
}
