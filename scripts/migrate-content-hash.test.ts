import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { computeContentHash } from "../lib/hash.ts";
import { encodeVectorIndexLine, readIndexLines } from "../lib/index-io.ts";
import { isMainModule, runContentHashMigration } from "./migrate-content-hash.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-migrate-content-hash-test-"));
}

function logger() {
	const stdout: string[] = [];
	const stderr: string[] = [];
	return {
		stdout,
		stderr,
		out: { log: (line: string) => stdout.push(line) },
		err: { error: (line: string) => stderr.push(line) },
	};
}

function roundFileName(userPrompt: string, responseSequence = "", toolCalls = undefined): string {
	return `${computeContentHash(userPrompt, responseSequence, toolCalls)}.json`;
}

describe("migrate-content-hash script", () => {
	it("reports missing and empty round directories without mutating state", () => {
		const logs = logger();
		const missing = path.join(tmpDir(), "missing");

		expect(runContentHashMigration({ roundsDir: missing, stdout: logs.out, stderr: logs.err })).toBe(1);
		expect(logs.stderr[0]).toContain("Rounds directory does not exist");

		const empty = tmpDir();
		expect(runContentHashMigration({ roundsDir: empty, stdout: logs.out, stderr: logs.err })).toBe(0);
		expect(logs.stdout.at(-1)).toContain("No round files found");
	});

	it("dry-runs id updates, renames, collisions, and index rewrites", () => {
		const roundsDir = tmpDir();
		const indexPath = path.join(roundsDir, "index.csv");
		const unchangedName = roundFileName("same", "answer");
		fs.writeFileSync(
			path.join(roundsDir, unchangedName),
			JSON.stringify({ id: "old", userPrompt: "same", responseSequence: "answer" }),
		);
		fs.writeFileSync(
			path.join(roundsDir, "legacy.json"),
			JSON.stringify({ userPrompt: "rename", responseSequence: "answer" }),
		);
		fs.writeFileSync(
			path.join(roundsDir, "target.json"),
			JSON.stringify({ userPrompt: "target", responseSequence: "answer" }),
		);
		fs.writeFileSync(path.join(roundsDir, "notes.txt"), "not json and not a round");
		fs.writeFileSync(indexPath, `${encodeVectorIndexLine([1], "legacy.json:prompt")}\n`);
		const logs = logger();

		expect(
			runContentHashMigration({
				args: ["--dry-run", "--backup"],
				roundsDir,
				indexPath,
				now: () => 123,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).toBe(0);

		expect(fs.existsSync(`${roundsDir}.bak-123`)).toBe(false);
		expect(fs.existsSync(path.join(roundsDir, "legacy.json"))).toBe(true);
		expect(JSON.parse(fs.readFileSync(path.join(roundsDir, unchangedName), "utf-8")).id).toBe("old");
		expect(readIndexLines(indexPath)).toEqual([encodeVectorIndexLine([1], "legacy.json:prompt")]);
		expect(logs.stdout.join("\n")).toContain("Found 3 round files");
		expect(logs.stdout.join("\n")).toContain(`would update id field: ${unchangedName}`);
		expect(logs.stdout.join("\n")).toContain("~ Would update index.csv (1 entries)");
		expect(logs.stdout.join("\n")).toContain("Hash changed:      2");
		expect(logs.stdout.join("\n")).toContain("Hash unchanged:    1");
		expect(logs.stdout.join("\n")).toContain("Renamed:           2");
		expect(logs.stdout.join("\n")).toContain("Dry run:           yes");
	});

	it("migrates changed hashes, updates ids, handles collisions, backs up, and reports parse errors", () => {
		const roundsDir = tmpDir();
		const indexPath = path.join(roundsDir, "index.csv");
		const renameNewName = roundFileName("rename", "answer");
		const collisionNewName = roundFileName("collision", "target");
		const unchangedName = roundFileName("same", "answer");

		fs.writeFileSync(
			path.join(roundsDir, "legacy.json"),
			JSON.stringify({ userPrompt: "rename", responseSequence: "answer" }),
		);
		fs.writeFileSync(
			path.join(roundsDir, "collision-source.json"),
			JSON.stringify({ userPrompt: "collision", responseSequence: "target" }),
		);
		fs.writeFileSync(
			path.join(roundsDir, collisionNewName),
			JSON.stringify({
				id: collisionNewName.replace(/\.json$/, ""),
				userPrompt: "collision",
				responseSequence: "target",
			}),
		);
		fs.writeFileSync(
			path.join(roundsDir, unchangedName),
			JSON.stringify({ id: "old", userPrompt: "same", responseSequence: "answer" }),
		);
		fs.writeFileSync(path.join(roundsDir, "bad.json"), "not json");
		fs.writeFileSync(path.join(roundsDir, "notes.txt"), "not json and not a round");
		fs.writeFileSync(
			indexPath,
			`${[
				encodeVectorIndexLine([1], "legacy.json:prompt"),
				encodeVectorIndexLine([2], "collision-source.json:response"),
				encodeVectorIndexLine([3], "retained.json:prompt"),
			].join("\n")}\n`,
		);
		const logs = logger();

		expect(
			runContentHashMigration({
				args: ["--backup"],
				roundsDir,
				indexPath,
				now: () => 123,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).toBe(1);

		expect(fs.existsSync(`${roundsDir}.bak-123`)).toBe(true);
		expect(fs.existsSync(path.join(roundsDir, "legacy.json"))).toBe(false);
		expect(fs.existsSync(path.join(roundsDir, renameNewName))).toBe(true);
		expect(JSON.parse(fs.readFileSync(path.join(roundsDir, renameNewName), "utf-8")).id).toBe(
			renameNewName.replace(/\.json$/, ""),
		);
		expect(fs.existsSync(path.join(roundsDir, "collision-source.json"))).toBe(false);
		expect(fs.existsSync(path.join(roundsDir, collisionNewName))).toBe(true);
		expect(JSON.parse(fs.readFileSync(path.join(roundsDir, unchangedName), "utf-8")).id).toBe(
			unchangedName.replace(/\.json$/, ""),
		);
		expect(readIndexLines(indexPath)).toEqual(
			expect.arrayContaining([
				encodeVectorIndexLine([3], "retained.json:prompt"),
				encodeVectorIndexLine([1], `${renameNewName}:prompt`),
				encodeVectorIndexLine([2], `${collisionNewName}:response`),
			]),
		);
		expect(readIndexLines(indexPath)).toHaveLength(3);
		expect(logs.stderr.join("\n")).toContain("Failed to parse bad.json");
		expect(logs.stdout.join("\n")).toContain("Found 5 round files");
		expect(logs.stdout.join("\n")).toContain("Backup created:");
		expect(logs.stdout.join("\n")).toContain("Hash changed:      2");
		expect(logs.stdout.join("\n")).toContain("Hash unchanged:    2");
		expect(logs.stdout.join("\n")).toContain("Renamed:           2");
		expect(logs.stdout.join("\n")).toContain("Skipped (errors):  1");
		expect(logs.stdout.join("\n")).toContain("Dry run:           no");
	});

	it("detects direct CLI execution", () => {
		expect(isMainModule("file:///tmp/script.ts", "/tmp/script.ts")).toBe(true);
		expect(isMainModule("file:///tmp/script.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/script.ts", undefined)).toBe(false);
	});
});
