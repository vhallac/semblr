import * as fs from "node:fs";
import * as path from "node:path";
import { atomicWriteFileSync } from "./atomic-write.ts";
import { appendLineWithLock, type IndexStorageFs, type LockedAppendDeps } from "./index-storage.ts";
import type { ToolCallDetail } from "./round-data.ts";

/** Legacy tool-index artifact name, adopted by rename on first write (decision 024). */
export const LEGACY_TOOL_INDEX_FILENAME = "index-tools.fulltext.csv";

export interface ToolIndexRow {
	hash: string;
	toolIndex: number;
	toolName: string;
	searchableText: string;
}

export interface ToolSearchMatch {
	hash: string;
	matchedKeywordCount: number;
	totalKeywords: number;
	matchedTools: Array<{ index: number; name: string }>;
}

/**
 * Legacy artifact path — superseded by `ownToolIndexPath` (issue #152); kept
 * until every caller routes to the host's own artifact (units 005/006).
 */
export function toolIndexPathForRoundsDir(roundsDir: string): string {
	return path.join(roundsDir, LEGACY_TOOL_INDEX_FILENAME);
}

/**
 * Path of the host's own tool-index artifact (`index-tools.<hostId>.fulltext.csv`,
 * issue #152). The host id is resolved by the caller (`resolveHostId`,
 * lib/host-id.ts) and is only ever used as a filename component — the artifact
 * format is unchanged (assumption A-2).
 */
export function ownToolIndexPath(roundsDir: string, hostId: string): string {
	return path.join(roundsDir, `index-tools.${hostId}.fulltext.csv`);
}

function collectStringValues(value: unknown, out: string[]): void {
	if (typeof value === "string") {
		out.push(value);
	} else if (Array.isArray(value)) {
		for (const item of value) collectStringValues(item, out);
	} else if (value && typeof value === "object") {
		for (const v of Object.values(value)) collectStringValues(v, out);
	}
}

/**
 * Build the searchable text for one tool call: tool name + all string
 * argument values, lowercased with whitespace collapsed to single spaces.
 */
export function buildSearchableText(toolName: string, argumentsJson: string): string {
	const parts: string[] = [toolName];
	try {
		collectStringValues(JSON.parse(argumentsJson), parts);
	} catch {
		// Non-JSON or malformed arguments — fall back to the raw string.
		parts.push(argumentsJson);
	}
	return parts.join(" ").toLowerCase().replace(/\s+/g, " ").trim();
}

/** Build one index row per tool call for a single round. */
export function buildToolIndexRows(roundFile: string, toolCalls: readonly ToolCallDetail[]): ToolIndexRow[] {
	return toolCalls.map((tc) => ({
		hash: roundFile,
		toolIndex: tc.index,
		toolName: tc.name,
		searchableText: buildSearchableText(tc.name, tc.arguments),
	}));
}

// searchableText is always the last field and never contains a raw newline
// (whitespace is collapsed when built), so plain comma-splitting on the
// first three commas is sufficient — no CSV quoting needed, matching the
// manual-split convention used by lib/index-io.ts.
export function encodeToolIndexRow(row: ToolIndexRow): string {
	return `${row.hash},${row.toolIndex},${row.toolName},${row.searchableText}`;
}

export function parseToolIndexLine(line: string): ToolIndexRow | null {
	const firstComma = line.indexOf(",");
	if (firstComma === -1) return null;
	const hash = line.slice(0, firstComma);
	const rest1 = line.slice(firstComma + 1);

	const secondComma = rest1.indexOf(",");
	if (secondComma === -1) return null;
	const toolIndex = Number(rest1.slice(0, secondComma));
	if (!Number.isFinite(toolIndex)) return null;
	const rest2 = rest1.slice(secondComma + 1);

	const thirdComma = rest2.indexOf(",");
	if (thirdComma === -1) return null;
	const toolName = rest2.slice(0, thirdComma);
	const searchableText = rest2.slice(thirdComma + 1);

	return { hash, toolIndex, toolName, searchableText };
}

export function readToolIndexLines(
	indexPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): string[] {
	if (!fsImpl.existsSync(indexPath)) return [];
	return fsImpl.readFileSync(indexPath, "utf-8").trim().split("\n").filter(Boolean);
}

export function loadToolIndex(
	indexPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): ToolIndexRow[] {
	const rows: ToolIndexRow[] = [];
	for (const line of readToolIndexLines(indexPath, fsImpl)) {
		const row = parseToolIndexLine(line);
		if (row) rows.push(row);
	}
	return rows;
}

/** Set of round filenames already present in the tool index (dedup helper for bulk rebuilds). */
export function loadToolIndexedRoundFiles(
	indexPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): Set<string> {
	return new Set(loadToolIndex(indexPath, fsImpl).map((row) => row.hash));
}

export interface AppendToolIndexDeps extends LockedAppendDeps {
	fsImpl?: IndexStorageFs;
}

/** Append rows to the tool index using the same lockfile pattern as index.csv. */
export function appendToolIndexRows(
	indexPath: string,
	roundsDir: string,
	rows: readonly ToolIndexRow[],
	deps: AppendToolIndexDeps = {},
): void {
	if (rows.length === 0) return;
	const fsImpl = deps.fsImpl ?? fs;
	// Issue #152 F3: the append path adopts the legacy artifact too — the first
	// runtime append on a store that only has `index-tools.fulltext.csv` claims
	// it by rename, so the upgraded store searches the legacy tool history
	// without a full rewrite. Same clobber-guard/lost-race semantics as the
	// rewrite path (writeToolIndexRows).
	fsImpl.mkdirSync(path.dirname(indexPath), { recursive: true });
	adoptLegacyToolIndex(indexPath, fsImpl);
	const line = `${rows.map(encodeToolIndexRow).join("\n")}\n`;
	appendLineWithLock(indexPath, roundsDir, line, deps);
}

/** fs surface needed to rewrite the tool index: adoption checks plus the atomic tmp+rename write. */
export type ToolIndexWriteFs = Pick<typeof fs, "existsSync" | "mkdirSync" | "writeFileSync" | "renameSync">;

/**
 * Adopt a legacy `index-tools.fulltext.csv` into the host's own artifact path
 * by rename (decision 024): on the first write to a path other than the legacy
 * one, when the legacy artifact exists and the target does not, the legacy
 * file is renamed to the target — an atomic same-directory rename (decision
 * 015). The rewrite that follows is built from locally visible round files
 * (the tool index is a wholesale rebuild, not an upsert). A host that already
 * owns its artifact leaves the legacy file untouched (POSIX rename overwrites,
 * so renaming would clobber the host's own artifact). A lost adoption race
 * (the legacy file was claimed by another host between the check and the
 * rename) is swallowed: the host proceeds with its own artifact.
 */
function adoptLegacyToolIndex(indexPath: string, fsImpl: ToolIndexWriteFs): boolean {
	if (path.basename(indexPath) === LEGACY_TOOL_INDEX_FILENAME) return false;
	const legacyPath = path.join(path.dirname(indexPath), LEGACY_TOOL_INDEX_FILENAME);
	if (!fsImpl.existsSync(legacyPath) || fsImpl.existsSync(indexPath)) return false;
	try {
		fsImpl.renameSync(legacyPath, indexPath);
		return true;
	} catch {
		// Lost the adoption race: the legacy file is gone (claimed by another
		// host) — proceed with the host's own artifact.
		return false;
	}
}

/**
 * Overwrite the tool index file with exactly these rows (used for full
 * rebuilds), atomically (decision 015): the content is staged to a sibling
 * tmp file (`<target>.tmp.<pid>`) and renamed over the target, so concurrent
 * readers never see a partial artifact and a crash mid-write leaves the
 * previous artifact byte-identical. In-place `writeFileSync` of the artifact
 * is gone (issue #152; decision 015 already named BM25/tool writes).
 */
export function writeToolIndexRows(
	indexPath: string,
	rows: readonly ToolIndexRow[],
	fsImpl: ToolIndexWriteFs = fs,
): void {
	fsImpl.mkdirSync(path.dirname(indexPath), { recursive: true });
	adoptLegacyToolIndex(indexPath, fsImpl);
	const content = rows.length > 0 ? `${rows.map(encodeToolIndexRow).join("\n")}\n` : "";
	atomicWriteFileSync(indexPath, content, fsImpl);
}

/** Scan every round file in roundsDir and rebuild tool index rows from scratch. */
export function buildToolIndexRowsFromRoundsDir(
	roundsDir: string,
	fsImpl: Pick<typeof fs, "readdirSync" | "readFileSync"> = fs,
): ToolIndexRow[] {
	const files = fsImpl.readdirSync(roundsDir).filter((f) => f.endsWith(".json") && !f.startsWith("index"));

	const rows: ToolIndexRow[] = [];
	for (const file of files) {
		try {
			const data = JSON.parse(fsImpl.readFileSync(path.join(roundsDir, file), "utf-8")) as {
				toolCalls?: ToolCallDetail[];
			};
			if (data.toolCalls && data.toolCalls.length > 0) {
				rows.push(...buildToolIndexRows(file, data.toolCalls));
			}
		} catch {
			// Corrupt round files are skipped during index rebuild.
		}
	}
	return rows;
}

/**
 * Search the tool index for rows whose searchable text contains every
 * keyword (case-insensitive substring match). Results are grouped by round
 * and sorted by descending unique-keyword coverage.
 */
export function searchToolIndex(
	rows: readonly ToolIndexRow[],
	query: string,
	scopeRounds?: readonly string[] | null,
): ToolSearchMatch[] {
	const keywords = query.toLowerCase().split(/\s+/).filter(Boolean);
	if (keywords.length === 0) return [];

	const scopeSet = scopeRounds && scopeRounds.length > 0 ? new Set(scopeRounds) : null;
	const scopedRows = scopeSet ? rows.filter((row) => scopeSet.has(row.hash)) : rows;

	const matchedKeywordsByHash = new Map<string, Set<string>>();
	const matchedToolsByHash = new Map<string, Map<number, string>>();
	const order: string[] = [];

	for (const row of scopedRows) {
		for (const keyword of keywords) {
			if (!row.searchableText.includes(keyword)) continue;

			if (!matchedKeywordsByHash.has(row.hash)) {
				matchedKeywordsByHash.set(row.hash, new Set());
				matchedToolsByHash.set(row.hash, new Map());
				order.push(row.hash);
			}
			matchedKeywordsByHash.get(row.hash)?.add(keyword);
			matchedToolsByHash.get(row.hash)?.set(row.toolIndex, row.toolName);
		}
	}

	const results = order.map((hash) => {
		const matchedTools = Array.from(matchedToolsByHash.get(hash) ?? new Map(), ([index, name]) => ({
			index,
			name,
		})).sort((a, b) => a.index - b.index);
		return {
			hash,
			matchedKeywordCount: matchedKeywordsByHash.get(hash)?.size ?? 0,
			totalKeywords: keywords.length,
			matchedTools,
		};
	});

	return results.sort((a, b) => b.matchedKeywordCount - a.matchedKeywordCount);
}
