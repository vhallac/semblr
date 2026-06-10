import * as fs from "node:fs";
import { formatFileSize } from "./context-format.ts";
import { indexRoundFileFromPath } from "./index-io.ts";
import type { RoundData } from "./round-data.ts";

/**
 * Read and parse a round JSON file from the rounds directory.
 * Returns the parsed JSON object, or null if the file doesn't exist
 * or cannot be parsed.
 */
export function readRoundJson(roundsDir: string, fileName: string): Record<string, unknown> | null {
	const fullPath = `${roundsDir}/${fileName}`;
	try {
		if (!fs.existsSync(fullPath)) return null;
		return JSON.parse(fs.readFileSync(fullPath, "utf-8"));
	} catch {
		return null;
	}
}

/**
 * Read a round file from the rounds directory, resolving index-path suffixes.
 * filePath may be "xxx.json:prompt", "xxx.json:response", "xxx.json:round",
 * or "xxx.json:summary" — the suffix is stripped to get the actual file.
 */
export function readRoundFileFromDir(
	filePath: string,
	roundsDir: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync"> = fs,
): RoundData | null {
	const actualFile = indexRoundFileFromPath(filePath);
	const fullPath = `${roundsDir}/${actualFile}`;
	if (!fsImpl.existsSync(fullPath)) return null;
	try {
		const data = JSON.parse(fsImpl.readFileSync(fullPath, "utf-8"));
		return {
			userPrompt: data.userPrompt ?? "",
			responseSequence: data.responseSequence ?? "",
			turnIndex: data.turnIndex ?? 0,
			userTimestamp: data.userTimestamp,
			toolCallCount: data.toolCallCount,
			toolCallNames: data.toolCallNames,
			toolCalls: data.toolCalls,
		};
	} catch {
		return null;
	}
}

/**
 * Stat a round file and return its formatted size string, or null on failure.
 */
export function getRoundFileSize(roundsDir: string, fileName: string): string | null {
	try {
		const stat = fs.statSync(`${roundsDir}/${fileName}`);
		return formatFileSize(stat.size);
	} catch {
		return null;
	}
}
