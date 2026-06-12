import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { resolveScriptConfig, type ScriptConfigOptions } from "../lib/script-config.ts";
import { createDefaultStatsState } from "../lib/stats.ts";

function parseArgValue(args: readonly string[], name: string): string | null {
	const index = args.indexOf(name);
	return index >= 0 ? (args[index + 1] ?? null) : null;
}

function expandHome(value: string, homedir: () => string): string {
	return value.replace(/^~(?=$|\/)/, homedir());
}

function timestampForPath(now: Date): string {
	return now.toISOString().replace(/[:.]/g, "-");
}

export interface RunSnapshotOptions extends ScriptConfigOptions {
	args?: string[];
	outDir?: string;
	stdout?: Pick<typeof console, "log">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: typeof fs;
	homedir?: () => string;
	now?: Date;
}

export async function runSnapshot(options: RunSnapshotOptions = {}): Promise<number> {
	const args = options.args ?? process.argv.slice(2);
	const config = resolveScriptConfig(options);
	const fsImpl = options.fsImpl ?? fs;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const homedir = options.homedir ?? os.homedir;
	const now = options.now ?? new Date();

	const roundsDir = config.roundsDir;
	const indexPath = config.indexPath;
	const semblrDir = path.dirname(roundsDir);
	const statsPath = path.join(semblrDir, "chain-read-stats.json");
	const sessionsDir = path.join(config.agentDir, "sessions");
	const requestedOutDir = options.outDir ?? parseArgValue(args, "--out");
	const snapshotDir = requestedOutDir
		? path.resolve(expandHome(requestedOutDir, homedir))
		: path.join(semblrDir, "snapshots", `corpus-${timestampForPath(now)}`);

	if (!fsImpl.existsSync(roundsDir)) {
		err.error(`Source rounds directory does not exist: ${roundsDir}`);
		return 1;
	}
	if (!fsImpl.statSync(roundsDir).isDirectory()) {
		err.error(`Source rounds path is not a directory: ${roundsDir}`);
		return 1;
	}
	if (!fsImpl.existsSync(indexPath)) {
		err.error(`Source index does not exist: ${indexPath}`);
		return 1;
	}
	if (fsImpl.existsSync(snapshotDir)) {
		err.error(`Snapshot target already exists: ${snapshotDir}`);
		return 1;
	}

	fsImpl.mkdirSync(path.dirname(snapshotDir), { recursive: true });
	fsImpl.mkdirSync(snapshotDir, { recursive: true });
	fsImpl.cpSync(roundsDir, path.join(snapshotDir, "rounds"), { recursive: true });

	const snapshotStatsPath = path.join(snapshotDir, "chain-read-stats.json");
	if (fsImpl.existsSync(statsPath)) {
		fsImpl.copyFileSync(statsPath, snapshotStatsPath);
	} else {
		fsImpl.writeFileSync(snapshotStatsPath, JSON.stringify(createDefaultStatsState(now.toISOString()), null, 2));
	}

	const snapshotSessionsDir = path.join(snapshotDir, "sessions");
	if (fsImpl.existsSync(sessionsDir)) {
		fsImpl.cpSync(sessionsDir, snapshotSessionsDir, { recursive: true });
	} else {
		fsImpl.mkdirSync(snapshotSessionsDir, { recursive: true });
	}

	out.log(`Snapshot written to ${snapshotDir}`);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const exitCode = await runSnapshot();
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((error) => {
		console.error("❌ Error:", error);
		process.exit(1);
	});
}
