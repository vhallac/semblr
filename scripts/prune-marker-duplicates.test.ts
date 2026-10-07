import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { computeContentHash, createRoundFilePath } from "../lib/hash.ts";
import { isMainModule, runPruneMarkerDuplicates, scanMarkerDuplicates } from "./prune-marker-duplicates.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-prune-dups-test-"));
}

const dirs: string[] = [];
afterEach(() => {
	for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
});

function markerFile(prefix: string): {
	preFixName: string;
	preFixData: object;
	twinName: string;
	twinData: object;
} {
	const raw = "answer\nround_needs_followup";
	const cleaned = "answer";
	const toolCalls = [{ arguments: "{}", result_summary: "ok" }];
	// Pre-fix file: raw marker text hashed (divergent name), needsFollowup absent
	const preFixName = `${computeContentHash(prefix, raw, toolCalls)}.json`;
	// Twin: stripped text hashed (live-path name), needsFollowup true
	const twinName = createRoundFilePath(prefix, cleaned, toolCalls);
	return {
		preFixName,
		preFixData: { userPrompt: prefix, responseSequence: raw, toolCalls },
		twinName,
		twinData: { userPrompt: prefix, responseSequence: cleaned, toolCalls, needsFollowup: true },
	};
}

describe("prune-marker-duplicates script", () => {
	it("deletes the pre-fix marker copy only when the stripped-hash twin exists with needsFollowup", () => {
		const roundsDir = tmpDir();
		dirs.push(roundsDir);
		const a = markerFile("prompt-a");
		const b = markerFile("prompt-b");
		// Duplicate pair: pre-fix + twin
		fs.writeFileSync(path.join(roundsDir, a.preFixName), JSON.stringify(a.preFixData));
		fs.writeFileSync(path.join(roundsDir, a.twinName), JSON.stringify(a.twinData));
		// Marker file with no twin — must be left alone
		fs.writeFileSync(path.join(roundsDir, b.preFixName), JSON.stringify(b.preFixData));

		const exit = runPruneMarkerDuplicates({ roundsDir, execute: true });

		expect(exit).toBe(0);
		expect(fs.existsSync(path.join(roundsDir, a.preFixName))).toBe(false);
		expect(fs.existsSync(path.join(roundsDir, a.twinName))).toBe(true);
		expect(fs.existsSync(path.join(roundsDir, b.preFixName))).toBe(true);
	});

	it("dry-run deletes nothing and lists the duplicates", () => {
		const roundsDir = tmpDir();
		dirs.push(roundsDir);
		const a = markerFile("prompt-a");
		fs.writeFileSync(path.join(roundsDir, a.preFixName), JSON.stringify(a.preFixData));
		fs.writeFileSync(path.join(roundsDir, a.twinName), JSON.stringify(a.twinData));

		const exit = runPruneMarkerDuplicates({ roundsDir });

		expect(exit).toBe(0);
		expect(fs.existsSync(path.join(roundsDir, a.preFixName))).toBe(true);
	});

	it("keeps a marker file whose twin exists without needsFollowup (not a verified twin)", () => {
		const roundsDir = tmpDir();
		dirs.push(roundsDir);
		const a = markerFile("prompt-a");
		fs.writeFileSync(path.join(roundsDir, a.preFixName), JSON.stringify(a.preFixData));
		fs.writeFileSync(
			path.join(roundsDir, a.twinName),
			JSON.stringify({
				userPrompt: "prompt-a",
				responseSequence: "answer",
				toolCalls: [{ arguments: "{}", result_summary: "ok" }],
			}),
		);

		const scan = scanMarkerDuplicates(roundsDir, fs);

		expect(scan.duplicates).toEqual([]);
		expect(scan.twinNotFollowup).toEqual([a.preFixName]);
		expect(fs.existsSync(path.join(roundsDir, a.preFixName))).toBe(true);
	});

	it("is idempotent: a second run on a clean store is a no-op", () => {
		const roundsDir = tmpDir();
		dirs.push(roundsDir);
		const a = markerFile("prompt-a");
		fs.writeFileSync(path.join(roundsDir, a.twinName), JSON.stringify(a.twinData));

		expect(runPruneMarkerDuplicates({ roundsDir, execute: true })).toBe(0);
		const scan = scanMarkerDuplicates(roundsDir, fs);
		expect(scan.duplicates).toEqual([]);
	});

	it("treats unparseable files as skipped, not duplicates", () => {
		const roundsDir = tmpDir();
		dirs.push(roundsDir);
		fs.writeFileSync(path.join(roundsDir, "broken.json"), "{not json");

		const scan = scanMarkerDuplicates(roundsDir, fs);

		expect(scan.unparseable).toEqual(["broken.json"]);
		expect(scan.duplicates).toEqual([]);
	});

	it("returns exit 1 for a missing rounds directory", () => {
		const logs = { out: [] as string[], err: [] as string[] };
		const missing = path.join(tmpDir(), "missing");
		dirs.push(missing);
		const exit = runPruneMarkerDuplicates({
			roundsDir: missing,
			stdout: { log: (l: string) => logs.out.push(l) },
			stderr: { error: (l: string) => logs.err.push(l), warn: (l: string) => logs.err.push(l) },
		});
		expect(exit).toBe(1);
		expect(logs.err[0]).toContain("Rounds directory not found");
	});

	it("isMainModule matches a script path module URL", () => {
		expect(isMainModule(pathToFileURLFixture(), "/x/prune-marker-duplicates.ts")).toBe(true);
		expect(isMainModule(pathToFileURLFixture(), "/x/other.ts")).toBe(false);
	});
});

function pathToFileURLFixture(): string {
	// eslint-disable-next-line @typescript-eslint/no-require-imports
	return new (require("node:url").URL)("file:///x/prune-marker-duplicates.ts").href;
}
