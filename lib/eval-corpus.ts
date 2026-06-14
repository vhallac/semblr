import * as fs from "node:fs";
import * as path from "node:path";
import type { RoundData } from "./round-data.ts";

export function parseArgValue(args: readonly string[], name: string): string | null {
	const index = args.indexOf(name);
	return index >= 0 ? (args[index + 1] ?? null) : null;
}

export function gatherSessionFiles(sessionsDir: string, fsImpl: typeof fs = fs): string[] {
	if (!fsImpl.existsSync(sessionsDir)) return [];
	const sessionDirs = fsImpl
		.readdirSync(sessionsDir)
		.filter((entry) => entry.startsWith("--"))
		.map((entry) => path.join(sessionsDir, entry));

	return sessionDirs
		.flatMap((dir) =>
			fsImpl
				.readdirSync(dir)
				.filter((entry) => entry.endsWith(".jsonl"))
				.map((entry) => path.join(dir, entry)),
		)
		.sort();
}

export function readCorpusRounds(roundsDir: string, fsImpl: typeof fs = fs): Map<string, RoundData> {
	const roundFiles = fsImpl
		.readdirSync(roundsDir)
		.filter((entry) => entry.endsWith(".json") && entry !== "chain-read-stats.json")
		.sort();

	return new Map(
		roundFiles.map((roundFile) => [
			roundFile,
			JSON.parse(fsImpl.readFileSync(path.join(roundsDir, roundFile), "utf-8")) as RoundData,
		]),
	);
}
