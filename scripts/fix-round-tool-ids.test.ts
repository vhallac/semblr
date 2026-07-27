import type * as fs from "node:fs";
import { describe, expect, it } from "vitest";
import type { SemblrConfig } from "../lib/semblr-config.js";
import { runFixRoundToolIds } from "./fix-round-tool-ids.ts";

function makeFS(files: Record<string, string>) {
	const allPaths = new Set<string>();
	for (const k of Object.keys(files)) {
		// Add each path component as an implicit directory
		const parts = k.split("/").filter(Boolean);
		let acc = "";
		for (const part of parts) {
			acc += `/${part}`;
			allPaths.add(acc);
		}
	}
	return {
		existsSync: (p: string) => allPaths.has(p),
		readFileSync: (p: string) => {
			if (!(p in files)) {
				const err = new Error(`ENOENT: ${p}`);
				(err as NodeJS.ErrnoException).code = "ENOENT";
				throw err;
			}
			return files[p];
		},
		readdirSync: (p: string) => {
			const prefix = p === "/" ? "/" : `${p}/`;
			const names = new Set<string>();
			for (const k of Object.keys(files)) {
				if (!k.startsWith(prefix)) continue;
				const rest = k.slice(prefix.length);
				const firstPart = rest.split("/")[0];
				if (firstPart) names.add(firstPart);
			}
			return [...names];
		},
		mkdirSync: () => {},
		writeFileSync: (p: string, data: string) => {
			files[p] = data;
		},
		appendFileSync: (p: string, data: string) => {
			files[p] = (files[p] ?? "") + data;
		},
		unlinkSync: (p: string) => {
			delete files[p];
		},
	};
}

const line = (v: unknown) => JSON.stringify(v);

function testConfig(overrides: Partial<SemblrConfig> = {}): SemblrConfig {
	return {
		agentDir: "/agent",
		embeddingProvider: "test",
		embeddingModel: "test-model",
		embeddingMaxTokens: 8000,
		roundsDir: "/rounds",
		indexPath: "/rounds/index.csv",
		groupThreshold: 0.77,
		minSimilarity: 0.3,
		embedTimeoutMs: 15000,
		embedMaxRetries: 3,
		embedBackoffMs: 1000,
		hybridSemanticWeight: 0.7,
		summaryThresholdExtra: 0,
		...overrides,
	};
}

describe("fix-round-tool-ids", () => {
	it("fixes a broken round file where tool results were assigned incorrectly", async () => {
		const sessionJSONL = [
			line({ type: "session", version: 3, id: "s1", timestamp: "2026-01-01T00:00:00Z", cwd: "/test" }),
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 1000, content: [{ type: "text", text: "Use tools" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "text", text: "Running tools for you now." },
						{ type: "toolCall", id: "call_a", name: "alpha", arguments: { x: 1 } },
						{ type: "toolCall", id: "call_b", name: "beta", arguments: { y: 2 } },
					],
				},
			}),
			// Results arrive reversed (parallel race), but with toolCallId for correct matching
			line({
				type: "message",
				message: {
					role: "toolResult",
					toolCallId: "call_b",
					toolName: "beta",
					content: [{ type: "text", text: "result of B" }],
				},
			}),
			line({
				type: "message",
				message: {
					role: "toolResult",
					toolCallId: "call_a",
					toolName: "alpha",
					content: [{ type: "text", text: "result of A" }],
				},
			}),
		].join("\n");

		// Create a broken round file (simulating what would have been produced
		// by the old reverse-sequential matcher — results swapped to wrong calls).
		const brokenRound = {
			id: "u1",
			userPrompt: "Use tools",
			responseSequence: "Running tools for you now.",
			responseEndTimestamp: 9999,
			toolCalls: [
				{
					index: 0,
					name: "alpha",
					arguments: '{"x":1}',
					result_summary: "result of B",
					result_full: "result of B",
					result_truncated: false,
				},
				{
					index: 1,
					name: "beta",
					arguments: '{"y":2}',
					result_summary: "result of A",
					result_full: "result of A",
					result_truncated: false,
				},
			],
		};

		const oldHash = "broken123.json";
		const files: Record<string, string> = {};
		files[`/sessions/--test--/s1.jsonl`] = sessionJSONL;
		files[`/rounds/${oldHash}`] = JSON.stringify(brokenRound);
		files["/rounds/index.csv"] = `base64vector,${oldHash}:prompt,test-model\n`;

		const fsImpl = makeFS(files) as unknown as typeof fs;
		const stdout = { log: () => {} } as Pick<typeof console, "log">;
		const stderr = { error: () => {} } as Pick<typeof console, "error">;

		await runFixRoundToolIds({
			config: testConfig({ roundsDir: "/rounds", agentDir: "/agent" }),
			sessionsDir: "/sessions",
			roundsDir: "/rounds",
			indexPath: "/rounds/index.csv",
			dryRun: false,
			stdout,
			stderr,
			fsImpl,
		});

		// Broken file should be deleted
		expect(files[`/rounds/${oldHash}`]).toBeUndefined();

		// Corrected file should exist with correct tool results
		const newFiles = Object.keys(files).filter(
			(k) => k.startsWith("/rounds/") && k.endsWith(".json") && !k.includes("index"),
		);
		expect(newFiles).toHaveLength(1);

		const corrected = JSON.parse(files[newFiles[0]]);
		expect(corrected.userPrompt).toBe("Use tools");
		expect(corrected.toolCalls).toHaveLength(2);
		expect(corrected.toolCalls[0].name).toBe("alpha");
		expect(corrected.toolCalls[0].result_full).toBe("result of A");
		expect(corrected.toolCalls[1].name).toBe("beta");
		expect(corrected.toolCalls[1].result_full).toBe("result of B");

		// Index should not reference the old hash
		const indexContent = files["/rounds/index.csv"] ?? "";
		expect(indexContent).not.toContain(oldHash);
	});

	it("skips rounds that are already correct", async () => {
		const sessionJSONL = [
			line({ type: "session", version: 3, id: "s1", timestamp: "2026-01-01T00:00:00Z", cwd: "/test" }),
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 1000, content: [{ type: "text", text: "Simple prompt" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Simple response with enough text to pass" }],
				},
			}),
		].join("\n");

		// Pre-compute the correct hash
		const { parsePiSessionJsonl } = await import("../lib/pi-session.js");
		const { computeContentHash } = await import("../lib/hash.js");
		const rounds = parsePiSessionJsonl(sessionJSONL, { skipShortFinalResponse: true });
		const correctHash = `${computeContentHash(rounds[0].userPrompt, rounds[0].responseSequence, rounds[0].toolCalls)}.json`;

		const files: Record<string, string> = {};
		files[`/sessions/--test--/s1.jsonl`] = sessionJSONL;
		files[`/rounds/${correctHash}`] = JSON.stringify(rounds[0]);
		files["/rounds/index.csv"] = `base64vector,${correctHash}:prompt,test-model\n`;

		const fsImpl = makeFS(files) as unknown as typeof fs;
		const stdout = { log: () => {} } as Pick<typeof console, "log">;
		const stderr = { error: () => {} } as Pick<typeof console, "error">;

		await runFixRoundToolIds({
			config: testConfig({ roundsDir: "/rounds", agentDir: "/agent" }),
			sessionsDir: "/sessions",
			roundsDir: "/rounds",
			indexPath: "/rounds/index.csv",
			dryRun: false,
			stdout,
			stderr,
			fsImpl,
		});

		expect(files[`/rounds/${correctHash}`]).toBeDefined();
	});

	it("dry-run does not modify files", async () => {
		const sessionJSONL = [
			line({ type: "session", version: 3, id: "s1", timestamp: "2026-01-01T00:00:00Z", cwd: "/test" }),
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 1000, content: [{ type: "text", text: "Use tools" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "text", text: "Some response text long enough to pass the filter" },
						{ type: "toolCall", id: "call_a", name: "alpha", arguments: { x: 1 } },
					],
				},
			}),
			line({
				type: "message",
				message: {
					role: "toolResult",
					toolCallId: "call_a",
					toolName: "alpha",
					content: [{ type: "text", text: "result of A" }],
				},
			}),
		].join("\n");

		const brokenRound = {
			id: "u1",
			userPrompt: "Use tools",
			responseSequence: "Some response text long enough to pass the filter",
			responseEndTimestamp: 9999,
			toolCalls: [
				{
					index: 0,
					name: "alpha",
					arguments: '{"x":1}',
					result_summary: "wrong result",
					result_full: "wrong result",
					result_truncated: false,
				},
			],
		};

		const oldHash = "broken_dry.json";
		const files: Record<string, string> = {};
		files[`/sessions/--test--/s1.jsonl`] = sessionJSONL;
		files[`/rounds/${oldHash}`] = JSON.stringify(brokenRound);
		files["/rounds/index.csv"] = `base64vector,${oldHash}:prompt,test-model\n`;

		const fsImpl = makeFS(files) as unknown as typeof fs;
		const logs: string[] = [];
		const stdout = { log: (msg: string) => logs.push(msg) } as Pick<typeof console, "log">;
		const stderr = { error: () => {} } as Pick<typeof console, "error">;

		await runFixRoundToolIds({
			config: testConfig({ roundsDir: "/rounds", agentDir: "/agent" }),
			sessionsDir: "/sessions",
			roundsDir: "/rounds",
			indexPath: "/rounds/index.csv",
			dryRun: true,
			stdout,
			stderr,
			fsImpl,
		});

		// Broken file must still exist
		expect(files[`/rounds/${oldHash}`]).toBeDefined();

		// Index must be unchanged
		expect(files["/rounds/index.csv"]).toContain(oldHash);

		// Should log dry-run message
		const joinedLogs = logs.join(" ");
		expect(joinedLogs).toContain("Would fix");
	});
});
