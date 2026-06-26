import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import type { SemblrConfig } from "../lib/semblr-config.ts";
import { runSnapshot } from "./snapshot.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-snapshot-test-"));
}

function testConfig(root: string): SemblrConfig {
	const agentDir = path.join(root, "agent");
	const roundsDir = path.join(agentDir, "semblr", "rounds");
	return {
		agentDir,
		embeddingProvider: "openrouter",
		embeddingModel: "openai/text-embedding-3-small",
		embeddingMaxTokens: 8000,
		roundsDir,
		indexPath: path.join(roundsDir, "index.csv"),
		groupThreshold: 0.77,
		minSimilarity: 0.3,
		embedTimeoutMs: 15_000,
		embedMaxRetries: 3,
		embedBackoffMs: 1000,
		summaryThresholdExtra: 40_000,
		multiModelRouting: {
			enabled: false,
			maxSwitches: 3,
			maxConsecutiveStuck: 2,
			phaseModelMap: {
				exploring: null,
				planning: "deepseek-v4-flash:cloud",
				executing: "glm-5.2:cloud",
				stuck: "kimi-k2.6:cloud",
				verifying: "minimax-m3:cloud",
				reporting: "gemma4:31b:cloud",
			},
		},
	};
}

describe("snapshot script", () => {
	it("copies rounds, index, stats, and sessions into a timestamped snapshot", async () => {
		const root = tmpDir();
		const config = testConfig(root);
		const semblrDir = path.dirname(config.roundsDir);
		const sessionFile = path.join(config.agentDir, "sessions", "--session-a", "rounds.jsonl");
		fs.mkdirSync(config.roundsDir, { recursive: true });
		fs.mkdirSync(path.dirname(sessionFile), { recursive: true });
		fs.writeFileSync(path.join(config.roundsDir, "round-a.json"), JSON.stringify({ userPrompt: "hello" }));
		fs.writeFileSync(config.indexPath, "vector,round-a.json:prompt\n");
		fs.writeFileSync(path.join(semblrDir, "chain-read-stats.json"), JSON.stringify({ version: 2 }));
		fs.writeFileSync(sessionFile, "{}\n");

		const stdout: string[] = [];
		await expect(
			runSnapshot({
				config,
				now: new Date("2026-06-12T06:24:11.000Z"),
				stdout: { log: (line: string) => stdout.push(line) },
			}),
		).resolves.toBe(0);

		const snapshotDir = path.join(semblrDir, "snapshots", "corpus-2026-06-12T06-24-11-000Z");
		expect(stdout).toEqual([`Snapshot written to ${snapshotDir}`]);
		expect(fs.readFileSync(path.join(snapshotDir, "rounds", "round-a.json"), "utf-8")).toContain("hello");
		expect(fs.readFileSync(path.join(snapshotDir, "rounds", "index.csv"), "utf-8")).toBe(
			"vector,round-a.json:prompt\n",
		);
		expect(fs.readFileSync(path.join(snapshotDir, "chain-read-stats.json"), "utf-8")).toBe(
			JSON.stringify({ version: 2 }),
		);
		expect(fs.readFileSync(path.join(snapshotDir, "sessions", "--session-a", "rounds.jsonl"), "utf-8")).toBe("{}\n");
	});

	it("requires the configured rounds directory", async () => {
		const root = tmpDir();
		const config = testConfig(root);
		const stderr: string[] = [];

		await expect(runSnapshot({ config, stderr: { error: (line: string) => stderr.push(line) } })).resolves.toBe(1);

		expect(stderr).toEqual([`Source rounds directory does not exist: ${config.roundsDir}`]);
	});

	it("requires the configured index file", async () => {
		const root = tmpDir();
		const config = testConfig(root);
		const stderr: string[] = [];
		fs.mkdirSync(config.roundsDir, { recursive: true });

		await expect(runSnapshot({ config, stderr: { error: (line: string) => stderr.push(line) } })).resolves.toBe(1);

		expect(stderr).toEqual([`Source index does not exist: ${config.indexPath}`]);
	});
});
