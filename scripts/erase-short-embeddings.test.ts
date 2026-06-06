import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { readIndexLines } from "../lib/index-io.ts";
import { runEraseShortEmbeddings } from "./erase-short-embeddings.ts";

describe("erase-short-embeddings", () => {
	function makeTempDir() {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-erase-test-"));
		return dir;
	}

	function createRoundFile(roundsDir: string, filename: string, userPrompt: string) {
		const roundPath = path.join(roundsDir, filename);
		fs.writeFileSync(
			roundPath,
			JSON.stringify({
				id: filename.replace(".json", ""),
				userPrompt,
				responseSequence: "Some response text here",
				promptEmbedding: [0.1, 0.2, 0.3],
				turnIndex: 0,
				toolCallCount: 1,
			}),
		);
		return roundPath;
	}

	function createIndexFile(roundsDir: string, entries: Array<{ roundFile: string; suffix: string }>) {
		const indexPath = path.join(roundsDir, "index.csv");
		const lines = entries.map(
			(e) => `${Buffer.from(JSON.stringify([0.1, 0.2])).toString("base64url")},${e.roundFile}:${e.suffix}`,
		);
		fs.writeFileSync(indexPath, `${lines.join("\n")}\n`);
		return indexPath;
	}

	it("removes :prompt and :response rows for short-prompt rounds", () => {
		const dir = makeTempDir();

		// Create a long-prompt round (should be kept)
		createRoundFile(
			dir,
			"long1.json",
			"this is a very clearly long prompt with more than enough words to completely pass the minimum threshold test for short embedding detection purposes",
		);
		// Create a short-prompt round (should be removed)
		createRoundFile(dir, "short1.json", "yes");

		// Create index entries
		const indexPath = createIndexFile(dir, [
			{ roundFile: "long1.json", suffix: "prompt" },
			{ roundFile: "long1.json", suffix: "response" },
			{ roundFile: "short1.json", suffix: "prompt" },
			{ roundFile: "short1.json", suffix: "response" },
		]);

		const exitCode = runEraseShortEmbeddings({ roundsDir: dir, indexPath });
		expect(exitCode).toBe(0);

		// Verify index: short entries should be removed
		const retained = readIndexLines(indexPath);
		expect(retained).toHaveLength(2);
		expect(retained.every((line) => line.includes("long1.json"))).toBe(true);
		expect(retained.some((line) => line.includes("short1.json"))).toBe(false);

		// Verify round JSON: short prompt has no promptEmbedding
		const shortData = JSON.parse(fs.readFileSync(path.join(dir, "short1.json"), "utf-8"));
		expect(shortData.promptEmbedding).toBeUndefined();

		// Long round still has its embedding
		const longData = JSON.parse(fs.readFileSync(path.join(dir, "long1.json"), "utf-8"));
		expect(longData.promptEmbedding).toEqual([0.1, 0.2, 0.3]);

		fs.rmSync(dir, { recursive: true, force: true });
	});

	it("handles empty index gracefully", () => {
		const dir = makeTempDir();
		const indexPath = path.join(dir, "index.csv");
		fs.writeFileSync(indexPath, "");

		const exitCode = runEraseShortEmbeddings({ roundsDir: dir, indexPath });
		expect(exitCode).toBe(0);

		fs.rmSync(dir, { recursive: true, force: true });
	});

	it("handles missing rounds directory", () => {
		const exitCode = runEraseShortEmbeddings({ roundsDir: "/nonexistent/semblr/rounds" });
		expect(exitCode).toBe(1);
	});

	it("respects custom minWords", () => {
		const dir = makeTempDir();

		// This prompt has 5 words — below default 20 but above custom 3
		createRoundFile(dir, "mid1.json", "this is a medium prompt");
		// This has 15 words — also medium
		createRoundFile(dir, "mid2.json", "this prompt has exactly fifteen words so it should be medium length prompt");
		// This has 2 words — below even the custom min
		createRoundFile(dir, "short1.json", "hi");

		const indexPath = createIndexFile(dir, [
			{ roundFile: "mid1.json", suffix: "prompt" },
			{ roundFile: "mid2.json", suffix: "prompt" },
			{ roundFile: "short1.json", suffix: "prompt" },
		]);

		const exitCode = runEraseShortEmbeddings({ roundsDir: dir, indexPath, minWords: 10 });
		expect(exitCode).toBe(0);

		const retained = readIndexLines(indexPath);
		// mid2 has 15 words >= 10, should be kept
		// mid1 has 5 < 10, should be removed
		// short1 has 2 < 10, should be removed
		expect(retained).toHaveLength(1);
		expect(retained[0]).toContain("mid2.json");

		fs.rmSync(dir, { recursive: true, force: true });
	});

	it("dry run does not modify files", () => {
		const dir = makeTempDir();

		createRoundFile(dir, "short1.json", "yes");
		const indexPath = createIndexFile(dir, [{ roundFile: "short1.json", suffix: "prompt" }]);

		const exitCode = runEraseShortEmbeddings({ roundsDir: dir, indexPath, args: ["--dry-run"] });
		expect(exitCode).toBe(0);

		// Index should be unchanged
		const retained = readIndexLines(indexPath);
		expect(retained).toHaveLength(1);
		expect(retained[0]).toContain("short1.json");

		// Round file should still have promptEmbedding
		const roundData = JSON.parse(fs.readFileSync(path.join(dir, "short1.json"), "utf-8"));
		expect(roundData.promptEmbedding).toEqual([0.1, 0.2, 0.3]);

		fs.rmSync(dir, { recursive: true, force: true });
	});

	it("returns 0 with no changes when all rounds are long enough", () => {
		const dir = makeTempDir();

		createRoundFile(
			dir,
			"long1.json",
			"this is a very clearly long prompt with more than enough words to completely pass the minimum threshold test for short embedding detection purposes",
		);
		createRoundFile(
			dir,
			"long2.json",
			"another very much longer prompt that should definitely never be filtered out because it has quite plenty of words included here now today",
		);
		const indexPath = createIndexFile(dir, [
			{ roundFile: "long1.json", suffix: "prompt" },
			{ roundFile: "long2.json", suffix: "prompt" },
		]);

		const exitCode = runEraseShortEmbeddings({ roundsDir: dir, indexPath });
		expect(exitCode).toBe(0);

		// Index should be unchanged
		const retained = readIndexLines(indexPath);
		expect(retained).toHaveLength(2);

		fs.rmSync(dir, { recursive: true, force: true });
	});
});
