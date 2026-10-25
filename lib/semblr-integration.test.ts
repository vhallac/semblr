import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { appendIndexRowToShards, embeddingClientDeps } from "../src/semblr.ts";

describe("embeddingClientDeps", () => {
	it("overrides only the embedding model for a model-specific query", () => {
		const modelRegistry = { marker: "registry" };
		const deps = embeddingClientDeps({ modelRegistry } as never, "historical-model");

		expect(deps.modelRegistry).toBe(modelRegistry);
		expect(deps.config.embeddingModel).toBe("historical-model");
		expect(deps.config.embeddingProvider).toBeTruthy();
		expect(deps.config.embedMaxRetries).toBeGreaterThan(0);
	});
});

describe("live append write surface (#149)", () => {
	let tempDir = "";

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-live-append-"));
	});

	afterEach(() => {
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	// The extension's appendToIndex routes through this seam: a live embed of a
	// round must land as a row in the host's own shard (index.<hostname>) and
	// nowhere else, with the legacy index.csv adopted on first write. Locked
	// with the shared index-writer protocol (decision 005) using the real fs.
	it("lands a live row in the host's own shard only (real fs, with lock)", () => {
		appendIndexRowToShards(
			tempDir,
			"rounds/abc.json:prompt",
			[0.1, 0.2, 0.3],
			{ hostname: os.hostname() },
			"test-model",
		);
		appendIndexRowToShards(
			tempDir,
			"rounds/abc.json:response",
			[0.4, 0.5, 0.6],
			{ hostname: os.hostname() },
			"test-model",
		);

		const files = fs.readdirSync(tempDir);
		expect(files).toEqual([`index.${os.hostname()}`]);
		const lines = fs
			.readFileSync(path.join(tempDir, `index.${os.hostname()}`), "utf-8")
			.trim()
			.split("\n");
		expect(lines).toHaveLength(2);
		expect(lines[0]).toContain("rounds/abc.json:prompt");
		expect(lines[1]).toContain("rounds/abc.json:response");
	});
});
