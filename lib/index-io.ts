import * as fs from "node:fs";
import * as path from "node:path";
import { computeContentHash, type HashToolCallDetail } from "./hash.ts";

export interface VectorIndexEntry {
	vector: number[];
	filePath: string;
	model?: string;
}

interface RoundHashContent {
	userPrompt?: string;
	responseSequence?: string;
	toolCalls?: HashToolCallDetail[];
}

export function encodeVectorIndexLine(vector: number[], filePath: string, model?: string): string {
	const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
	if (model !== undefined) {
		return `${b64},${filePath},${model}`;
	}
	return `${b64},${filePath}`;
}

function parseVectorIndexLine(line: string): VectorIndexEntry {
	const firstComma = line.indexOf(",");
	const b64 = line.slice(0, firstComma);
	const rest = line.slice(firstComma + 1);
	const lastComma = rest.lastIndexOf(",");
	if (lastComma === -1) {
		// 2-column format: vector,filePath
		const vector = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
		return { vector, filePath: rest };
	}
	// 3-column format: vector,filePath,model
	const filePath = rest.slice(0, lastComma);
	const model = rest.slice(lastComma + 1);
	const vector = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
	return { vector, filePath, model };
}

export function readIndexLines(indexPath: string): string[] {
	if (!fs.existsSync(indexPath)) return [];
	return fs.readFileSync(indexPath, "utf-8").trim().split("\n").filter(Boolean);
}

export function writeIndexLines(indexPath: string, entries: string[]): void {
	fs.writeFileSync(indexPath, entries.join("\n") + (entries.length > 0 ? "\n" : ""));
}

export function appendVectorIndexEntry(indexPath: string, vector: number[], filePath: string, model?: string): void {
	fs.appendFileSync(indexPath, `${encodeVectorIndexLine(vector, filePath, model)}\n`);
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
	const replacement = entries.map((entry) => encodeVectorIndexLine(entry.vector, entry.filePath, entry.model));
	writeIndexLines(indexPath, [...remaining, ...replacement]);
}

export function migrateIndexEntryLine(line: string, oldRoundFile: string, newRoundFile: string): string {
	const firstComma = line.indexOf(",");
	const rest = line.slice(firstComma + 1);
	// rest is "filePath" (2-col) or "filePath,model" (3-col)
	const lastComma = rest.lastIndexOf(",");
	if (lastComma === -1) {
		// 2-column: rest is the filePath
		if (!rest.startsWith(oldRoundFile)) return line;
		return `${line.slice(0, firstComma + 1)}${newRoundFile}${rest.slice(oldRoundFile.length)}`;
	}
	// 3-column: rest is "filePath,model"
	const fp = rest.slice(0, lastComma);
	const model = rest.slice(lastComma);
	if (!fp.startsWith(oldRoundFile)) return line;
	return `${line.slice(0, firstComma + 1)}${newRoundFile}${fp.slice(oldRoundFile.length)}${model}`;
}

export function migrateIndexEntries(indexPath: string, oldRoundFile: string, newRoundFile: string): void {
	if (!fs.existsSync(indexPath)) return;
	const migrated = readIndexLines(indexPath).map((line) => migrateIndexEntryLine(line, oldRoundFile, newRoundFile));
	writeIndexLines(indexPath, migrated);
}

export function indexEntryFilename(line: string): string | null {
	const commaIdx = line.indexOf(",");
	if (commaIdx === -1) return null;
	let entry = line.slice(commaIdx + 1);
	// Strip model column if present (3-column format: vector,filePath,model)
	const lastComma = entry.lastIndexOf(",");
	if (lastComma !== -1) {
		entry = entry.slice(0, lastComma);
	}
	const colonIdx = entry.lastIndexOf(":");
	if (colonIdx === -1) return null;
	return entry.slice(0, colonIdx);
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
	const rest = line.slice(commaIdx + 1);
	// rest is "filePath:prompt" (2-col) or "filePath:prompt,model" (3-col)
	const lastComma = rest.lastIndexOf(",");
	if (lastComma === -1) {
		// 2-column
		const namePart = rest.replace(/^[^:]+/, newFilename);
		return line.slice(0, commaIdx + 1) + namePart;
	}
	// 3-column: replace filename in the filePath portion only
	const fp = rest.slice(0, lastComma);
	const model = rest.slice(lastComma);
	const namePart = fp.replace(/^[^:]+/, newFilename);
	return line.slice(0, commaIdx + 1) + namePart + model;
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
