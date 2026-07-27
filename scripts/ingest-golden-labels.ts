import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import type { GoldenPoolFile } from "./build-golden-pool.ts";

function parseArgValue(args: readonly string[], name: string): string | null {
	const index = args.indexOf(name);
	return index >= 0 ? (args[index + 1] ?? null) : null;
}

function expandHome(value: string, homedir: () => string): string {
	return value.replace(/^~(?=$|\/)/, homedir());
}

function defaultPoolPath(): string {
	return path.resolve("docs/eval/golden-pool.local.json");
}

function defaultWorksheetPath(): string {
	return path.resolve("docs/eval/golden-worksheet.local.md");
}

function defaultOutPath(): string {
	return path.resolve("docs/eval/golden-labels.local.json");
}

interface WorksheetSelection {
	labels: string[];
	primary: string | null;
	warnings: string[];
}

export function parseWorksheetSelections(raw: string): Map<string, WorksheetSelection> {
	const selections = new Map<string, WorksheetSelection>();
	let currentQuery: string | null = null;
	for (const line of raw.split("\n")) {
		const heading = /^##\s+\d+\.\s+(.+)$/.exec(line.trim());
		if (heading) {
			currentQuery = heading[1];
			selections.set(currentQuery, { labels: [], primary: null, warnings: [] });
			continue;
		}
		if (!currentQuery) continue;
		const trimmedLine = line.trim();
		const candidate = /^-\s+\[(?<checked>[ xX])\]\s+(?<file>\S+)(?<rest>.*)$/.exec(trimmedLine);
		const selection = selections.get(currentQuery);
		if (!selection) continue;
		if (!candidate?.groups) {
			if (/^-\s+\[.*\]/.test(trimmedLine)) {
				selection.warnings.push(`Worksheet checkbox line could not be parsed for ${currentQuery}: ${trimmedLine}`);
			}
			continue;
		}
		const file = candidate.groups.file;
		const rest = candidate.groups.rest;
		const isChecked = candidate.groups.checked.toLowerCase() === "x";
		const isPrimary = /\(primary\)/.test(rest);
		if (isChecked) selection.labels.push(file);
		if (isPrimary) {
			if (selection.primary && selection.primary !== file) {
				throw new Error(`Worksheet has multiple primary labels for ${currentQuery}`);
			}
			selection.primary = file;
			if (!isChecked) {
				selection.labels.push(file);
				selection.warnings.push(
					`Primary label auto-selected for ${currentQuery}: ${file} was marked (primary) without [x]`,
				);
			}
		}
	}
	for (const selection of selections.values()) {
		selection.labels = [...new Set(selection.labels)];
	}
	return selections;
}

export interface GoldenLabelsFile {
	kind: "golden-labels";
	version: 1;
	source_pool: string;
	queries: Array<{
		query: string;
		prompt: string;
		difficulty: string;
		primary: string | null;
		labels: string[];
		mode?: "similarity" | "tool";
		tool_query?: string;
	}>;
}

export interface RunIngestGoldenLabelsOptions {
	args?: string[];
	poolFile?: string;
	worksheetFile?: string;
	outFile?: string;
	stdout?: Pick<typeof console, "log" | "warn">;
	stderr?: Pick<typeof console, "error">;
	fsImpl?: typeof fs;
	homedir?: () => string;
}

export async function runIngestGoldenLabels(options: RunIngestGoldenLabelsOptions = {}): Promise<number> {
	const args = options.args ?? process.argv.slice(2);
	const fsImpl = options.fsImpl ?? fs;
	const out = options.stdout ?? console;
	const err = options.stderr ?? console;
	const homedir = options.homedir ?? os.homedir;
	const poolFile = options.poolFile ?? parseArgValue(args, "--pool") ?? defaultPoolPath();
	const worksheetFile = options.worksheetFile ?? parseArgValue(args, "--worksheet") ?? defaultWorksheetPath();
	const outFile = options.outFile ?? parseArgValue(args, "--out") ?? defaultOutPath();
	const resolvedPoolFile = path.resolve(expandHome(poolFile, homedir));
	const resolvedWorksheetFile = path.resolve(expandHome(worksheetFile, homedir));
	const resolvedOutFile = path.resolve(expandHome(outFile, homedir));

	if (!fsImpl.existsSync(resolvedPoolFile)) {
		err.error(`Golden pool does not exist: ${resolvedPoolFile}`);
		return 1;
	}
	if (!fsImpl.existsSync(resolvedWorksheetFile)) {
		err.error(`Golden worksheet does not exist: ${resolvedWorksheetFile}`);
		return 1;
	}

	const pool = JSON.parse(fsImpl.readFileSync(resolvedPoolFile, "utf-8")) as GoldenPoolFile;
	const worksheet = fsImpl.readFileSync(resolvedWorksheetFile, "utf-8");
	const selectedByQuery = parseWorksheetSelections(worksheet);
	const queries = pool.queries.map((query) => {
		const selected = selectedByQuery.get(query.query) ?? { labels: [], primary: null, warnings: [] };
		const candidateFiles = query.candidates.map((candidate) => candidate.file);
		for (const label of selected.labels) {
			if (!candidateFiles.includes(label)) {
				throw new Error(`Worksheet selected candidate not present in pool for ${query.query}: ${label}`);
			}
		}
		for (const warning of selected.warnings) out.warn(warning);
		return {
			query: query.query,
			prompt: query.prompt,
			difficulty: query.difficulty,
			primary: selected.primary,
			labels: [...new Set(selected.labels)],
		};
	});

	const output: GoldenLabelsFile = {
		kind: "golden-labels",
		version: 1,
		source_pool: resolvedPoolFile,
		queries,
	};
	fsImpl.mkdirSync(path.dirname(resolvedOutFile), { recursive: true });
	fsImpl.writeFileSync(resolvedOutFile, `${JSON.stringify(output, null, 2)}\n`);
	out.log(`Golden labels written to ${resolvedOutFile}`);
	return 0;
}

export function isMainModule(metaUrl: string, argv1 = process.argv[1]): boolean {
	return argv1 ? pathToFileURL(argv1).href === metaUrl : false;
}

async function main() {
	const exitCode = await runIngestGoldenLabels();
	if (exitCode !== 0) process.exit(exitCode);
}

if (isMainModule(import.meta.url)) {
	main().catch((error) => {
		console.error("❌ Error:", error);
		process.exit(1);
	});
}
