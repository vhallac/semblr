import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { encodeVectorIndexLine, readIndexLines } from "../lib/index-io.ts";
import { isMainModule, runModelColumnMigration } from "./migrate-model-column.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-migrate-model-test-"));
}

function logger() {
	const stdout: string[] = [];
	const stderr: string[] = [];
	return {
		stdout,
		stderr,
		out: { log: (line: string) => stdout.push(line) },
		err: { error: (line: string) => stderr.push(line), warn: (line: string) => stderr.push(line) },
	};
}

describe("migrate-model-column script", () => {
	it("adds the configured embedding model to legacy index entries", () => {
		const root = tmpDir();
		const roundsDir = path.join(root, "configured-rounds");
		fs.mkdirSync(path.join(root, ".pi"), { recursive: true });
		fs.mkdirSync(roundsDir, { recursive: true });
		fs.writeFileSync(
			path.join(root, ".pi", "settings.json"),
			JSON.stringify({ semblr: { roundsDir: "configured-rounds", embeddingModel: "configured-model" } }),
		);
		const indexPath = path.join(roundsDir, "index.csv");
		fs.writeFileSync(indexPath, `${encodeVectorIndexLine([1], "round.json:prompt")}\n`);
		const logs = logger();

		expect(
			runModelColumnMigration({
				stdout: logs.out,
				stderr: logs.err,
				configDeps: { cwd: root, agentDir: path.join(root, "agent"), env: {} },
			}),
		).toBe(0);

		expect(readIndexLines(indexPath)).toEqual([
			`${encodeVectorIndexLine([1], "round.json:prompt")},configured-model`,
		]);
		expect(logs.stdout.join("\n")).toContain("Model to add: configured-model");
	});

	it("reports missing index path", () => {
		const logs = logger();

		expect(
			runModelColumnMigration({ indexPath: path.join(tmpDir(), "missing.csv"), stdout: logs.out, stderr: logs.err }),
		).toBe(1);

		expect(logs.stderr).toEqual([expect.stringContaining("Index file not found")]);
	});

	it("detects direct CLI execution", () => {
		expect(isMainModule("file:///tmp/migrate-model-column.ts", "/tmp/migrate-model-column.ts")).toBe(true);
		expect(isMainModule("file:///tmp/migrate-model-column.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/migrate-model-column.ts", undefined)).toBe(false);
	});
});
