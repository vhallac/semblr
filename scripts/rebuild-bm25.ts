/**
 * rebuild-bm25.ts — Rebuild Semblr's BM25 keyword index from saved round files.
 *
 * Usage:
 *   npx tsx scripts/rebuild-bm25.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { bm25IndexPathForRoundsDir, buildBm25Index, roundTextForBm25, writeBm25Index } from "../lib/bm25-index.ts";
import { resolveScriptConfig, type ScriptConfigOptions } from "../lib/script-config.ts";

export interface RebuildBm25Options extends ScriptConfigOptions {
	roundsDir?: string;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: Pick<typeof fs, "existsSync" | "mkdirSync" | "readFileSync" | "readdirSync" | "writeFileSync">;
}

export async function runRebuildBm25(options: RebuildBm25Options = {}): Promise<number> {
	const config = resolveScriptConfig(options);
	const roundsDir = options.roundsDir ?? config.roundsDir;
	const bm25IndexPath = bm25IndexPathForRoundsDir(roundsDir);
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const f = options.fsImpl ?? fs;

	if (!f.existsSync(roundsDir)) {
		err.error(`Rounds directory does not exist: ${roundsDir}`);
		return 1;
	}

	const documents = [];
	for (const fileName of f
		.readdirSync(roundsDir)
		.filter((file) => file.endsWith(".json") && !file.startsWith("index"))) {
		try {
			const round = JSON.parse(f.readFileSync(path.join(roundsDir, fileName), "utf-8"));
			documents.push({ fileName, text: roundTextForBm25(round) });
		} catch (error) {
			err.error(`Skipping unreadable round ${fileName}: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	writeBm25Index(bm25IndexPath, buildBm25Index(documents), f);
	out.log(`Rebuilt BM25 index: ${documents.length} rounds at ${bm25IndexPath}`);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const exitCode = await runRebuildBm25();
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((err) => {
		console.error("Error:", err);
		process.exit(1);
	});
}
