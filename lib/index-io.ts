import * as fs from "node:fs";
import * as path from "node:path";
import { computeContentHash, type HashToolCallDetail } from "./hash.ts";

export interface VectorIndexEntry {
	vector: number[];
	filePath: string;
	model?: string;
	/**
	 * Optional 4th CSV column: sha256-base64url of the exact prompt-side embedding
	 * input text the vector was computed from (issue #106 re-embed migration stamp).
	 * Only meaningful on :prompt rows; consumers must treat it as opaque metadata.
	 */
	embeddingInputHash?: string;
}

interface RoundHashContent {
	userPrompt?: string;
	responseSequence?: string;
	toolCalls?: HashToolCallDetail[];
}

export function encodeVectorIndexLine(
	vector: number[],
	filePath: string,
	model?: string,
	embeddingInputHash?: string,
): string {
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	const parts = [b64, filePath];
	if (model !== undefined) parts.push(model);
	if (embeddingInputHash !== undefined) parts.push(embeddingInputHash);
	return parts.join(",");
}

/**
 * Split an index line's metadata columns (everything after the vector column).
 * Metadata columns (filePath, model, embeddingInputHash) contain no commas
 * themselves (round filenames are content hashes; model slugs and hashes are
 * comma-free), so the trailing segments map positionally from the right:
 *   [filePath] | [filePath, model] | [filePath, model, embeddingInputHash]
 */
export function splitVectorIndexMetadata(rest: string): {
	filePath: string;
	model?: string;
	embeddingInputHash?: string;
} {
	const segments = rest.split(",");
	if (segments.length >= 3) {
		return {
			filePath: segments.slice(0, segments.length - 2).join(","),
			model: segments[segments.length - 2],
			embeddingInputHash: segments[segments.length - 1],
		};
	}
	if (segments.length === 2) {
		return { filePath: segments[0], model: segments[1] };
	}
	return { filePath: segments[0] ?? "" };
}

function parseVectorIndexLine(line: string): VectorIndexEntry {
	const firstComma = line.indexOf(",");
	const b64 = line.slice(0, firstComma);
	const vector = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
	// Metadata columns are parsed right-to-left so the optional 4th column
	// (embeddingInputHash) does not corrupt the model or filePath fields.
	const { filePath, model, embeddingInputHash } = splitVectorIndexMetadata(line.slice(firstComma + 1));
	const entry: VectorIndexEntry = { vector, filePath };
	if (model !== undefined) entry.model = model;
	if (embeddingInputHash !== undefined) entry.embeddingInputHash = embeddingInputHash;
	return entry;
}

export function readIndexLines(indexPath: string): string[] {
	if (!fs.existsSync(indexPath)) return [];
	return fs.readFileSync(indexPath, "utf-8").trim().split("\n").filter(Boolean);
}

export function writeIndexLines(indexPath: string, entries: string[]): void {
	fs.writeFileSync(indexPath, entries.join("\n") + (entries.length > 0 ? "\n" : ""));
}

export function appendVectorIndexEntry(
	indexPath: string,
	vector: number[],
	filePath: string,
	model?: string,
	embeddingInputHash?: string,
): void {
	fs.appendFileSync(indexPath, `${encodeVectorIndexLine(vector, filePath, model, embeddingInputHash)}\n`);
}

export function loadVectorIndex(indexPath: string): VectorIndexEntry[] {
	return readIndexLines(indexPath).map(parseVectorIndexLine);
}

export function indexRoundFileFromPath(filePath: string): string {
	return filePath.replace(/(:prompt|:response|:round|:summary)$/, "");
}

export function loadIndexedRoundFiles(indexPath: string): Set<string> {
	return new Set(loadVectorIndex(indexPath).map((entry) => path.basename(indexRoundFileFromPath(entry.filePath))));
}

export function loadRoundFilesWithDifferentModel(indexPath: string, currentModel: string): Set<string> {
	const mismatched = loadVectorIndex(indexPath)
		.filter((entry) => entry.model !== undefined && entry.model !== currentModel)
		.map((entry) => path.basename(indexRoundFileFromPath(entry.filePath)));
	return new Set(mismatched);
}

export function replaceIndexEntriesForRoundFile(
	indexPath: string,
	roundFile: string,
	entries: VectorIndexEntry[],
): void {
	const remaining = readIndexLines(indexPath).filter((line) => {
		const filename = indexEntryFilename(line);
		return !filename || path.basename(filename) !== roundFile;
	});
	const replacement = entries.map((entry) =>
		encodeVectorIndexLine(entry.vector, entry.filePath, entry.model, entry.embeddingInputHash),
	);
	writeIndexLines(indexPath, [...remaining, ...replacement]);
}

export function migrateIndexEntryLine(line: string, oldRoundFile: string, newRoundFile: string): string {
	const firstComma = line.indexOf(",");
	if (firstComma === -1) return line;
	// First metadata segment is "filePath[:suffix]"; trailing model/embeddingInputHash
	// columns (comma-free) are preserved verbatim.
	const segments = line.slice(firstComma + 1).split(",");
	if (!segments[0].startsWith(oldRoundFile)) return line;
	segments[0] = newRoundFile + segments[0].slice(oldRoundFile.length);
	return `${line.slice(0, firstComma + 1)}${segments.join(",")}`;
}

export function migrateIndexEntries(indexPath: string, oldRoundFile: string, newRoundFile: string): void {
	if (!fs.existsSync(indexPath)) return;
	const migrated = readIndexLines(indexPath).map((line) => migrateIndexEntryLine(line, oldRoundFile, newRoundFile));
	writeIndexLines(indexPath, migrated);
}

export function indexEntryFilename(line: string): string | null {
	const commaIdx = line.indexOf(",");
	if (commaIdx === -1) return null;
	// First metadata segment is "filePath[:suffix]"; trailing model/hash columns are ignored.
	const filePath = line.slice(commaIdx + 1).split(",")[0];
	const colonIdx = filePath.lastIndexOf(":");
	if (colonIdx === -1) return null;
	return filePath.slice(0, colonIdx);
}

export function readIndexByFilename(indexPath: string): Map<string, string[]> {
	const index = new Map<string, string[]>();
	for (const line of readIndexLines(indexPath)) {
		const filename = indexEntryFilename(line);
		if (!filename) continue;
		if (!index.has(filename)) index.set(filename, []);
		index.get(filename)?.push(line);
	}
	return index;
}

export function replaceIndexLineFilename(line: string, newFilename: string): string {
	const commaIdx = line.indexOf(",");
	if (commaIdx === -1) return line;
	// Replace the filename in the first metadata segment ("filePath[:suffix]");
	// trailing model/embeddingInputHash columns are preserved verbatim.
	const segments = line.slice(commaIdx + 1).split(",");
	segments[0] = segments[0].replace(/^[^:]+/, newFilename);
	return line.slice(0, commaIdx + 1) + segments.join(",");
}

export function filterIndexLinesExcludingFilenames(lines: string[], filenames: Set<string>): string[] {
	return lines.filter((line) => {
		const filename = indexEntryFilename(line);
		return !filename || !filenames.has(filename);
	});
}

export function findStaleContentMatches(roundsDir: string, roundFile: string): string[] {
	const files = fs.readdirSync(roundsDir).filter((f) => f.endsWith(".json") && !f.startsWith("index"));
	const matches: string[] = [];
	for (const file of files) {
		if (file === roundFile) continue;
		try {
			const data = JSON.parse(fs.readFileSync(path.join(roundsDir, file), "utf-8")) as RoundHashContent;
			const hash = `${computeContentHash(data.userPrompt ?? "", data.responseSequence ?? "", data.toolCalls)}.json`;
			if (hash === roundFile) matches.push(file);
		} catch {
			// Corrupt round files are ignored during stale-content discovery.
		}
	}
	return matches;
}
