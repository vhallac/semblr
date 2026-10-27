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
	// File-descriptor surface for the chunked index reader (forEachIndexLine).
	const fds = new Map<number, { path: string; position: number }>();
	let nextFd = 1;
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
		renameSync: (from: string, to: string) => {
			if (!(from in files)) {
				const err = new Error(`ENOENT: ${from}`);
				(err as NodeJS.ErrnoException).code = "ENOENT";
				throw err;
			}
			files[to] = files[from];
			delete files[from];
		},
		appendFileSync: (p: string, data: string) => {
			files[p] = (files[p] ?? "") + data;
		},
		unlinkSync: (p: string) => {
			delete files[p];
		},
		openSync: (p: string) => {
			const fd = nextFd++;
			fds.set(fd, { path: p, position: 0 });
			return fd;
		},
		readSync: (fd: number, buffer: Buffer, offset: number, length: number, position: number | null) => {
			const entry = fds.get(fd);
			if (!entry) return 0;
			const data = Buffer.from(files[entry.path] ?? "", "utf-8");
			const start = position ?? entry.position;
			const chunk = data.subarray(start, start + length);
			chunk.copy(buffer, offset);
			entry.position = start + chunk.length;
			return chunk.length;
		},
		closeSync: (fd: number) => {
			fds.delete(fd);
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
		hostId: "test-host",
		groupThreshold: 0.77,
		minSimilarity: 0.3,
		embedTimeoutMs: 15000,
		embedMaxRetries: 3,
		embedBackoffMs: 1000,
		hybridSemanticWeight: 0.7,
		summaryThresholdExtra: 0,
		contextPromptHeadChars: 260,
		contextPromptTailChars: 140,
		contextBudgetRatio: 0.08,
		contextRelevanceMaxEntries: 20,
		contextRecencyMaxEntries: 20,
		promptNoiseFenceMaxChars: 600,
		promptNoiseJsonMaxChars: 600,
		promptNoiseRepeatMaxChars: 200,
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
			hostname: "test-host",
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

		// The legacy index was claimed by the own shard; the migrated row must
		// not reference the old hash (issue #149: the rewrite touches the own
		// shard only).
		expect(files["/rounds/index.csv"]).toBeUndefined();
		const indexContent = files["/rounds/index.test-host"] ?? "";
		expect(indexContent).not.toContain(oldHash);
		expect(indexContent).toContain(newFiles[0].split("/").pop() ?? "");
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
			hostname: "test-host",
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
			hostname: "test-host",
			dryRun: true,
			stdout,
			stderr,
			fsImpl,
		});

		// Broken file must still exist
		expect(files[`/rounds/${oldHash}`]).toBeDefined();

		// Index must be unchanged — a dry run claims nothing and writes nothing.
		expect(files["/rounds/index.csv"]).toContain(oldHash);
		expect(files["/rounds/index.test-host"]).toBeUndefined();

		// Should log dry-run message
		const joinedLogs = logs.join(" ");
		expect(joinedLogs).toContain("Would fix");
	});

	it("failure-injection: tmp write throws → broken file byte-identical, no corrected file, run reports errors", async () => {
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
			responseSequence: "Running tools for you now.",
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

		const oldHash = "broken_fail.json";
		const oldContent = JSON.stringify(brokenRound);
		const files: Record<string, string> = {};
		files[`/sessions/--test--/s1.jsonl`] = sessionJSONL;
		files[`/rounds/${oldHash}`] = oldContent;
		files["/rounds/index.csv"] = `base64vector,${oldHash}:prompt,test-model\n`;

		const fsImpl = makeFS(files) as unknown as typeof fs;
		// Fail only the atomic tmp write of the corrected round file, not the
		// (also atomic) index publish.
		const inner = fsImpl.writeFileSync.bind(fsImpl);
		(fsImpl as { writeFileSync: (p: string, data: string) => void }).writeFileSync = (p, data) => {
			if (p.includes(".json.tmp.")) throw new Error("disk full");
			inner(p, data);
		};
		const stdout = { log: () => {} } as Pick<typeof console, "log">;
		const stderr = { error: () => {} } as Pick<typeof console, "error">;

		const exitCode = await runFixRoundToolIds({
			config: testConfig({ roundsDir: "/rounds", agentDir: "/agent" }),
			sessionsDir: "/sessions",
			roundsDir: "/rounds",
			indexPath: "/rounds/index.csv",
			hostname: "test-host",
			dryRun: false,
			stdout,
			stderr,
			fsImpl,
		});

		// Run survived the per-file catch and reports the error
		expect(exitCode).toBe(1);

		// Broken file survives byte-identical (crash-safe ordering)
		expect(files[`/rounds/${oldHash}`]).toBe(oldContent);

		// No corrected file and no tmp leftovers
		const newFiles = Object.keys(files).filter(
			(k) => k.startsWith("/rounds/") && k.endsWith(".json") && !k.includes("index"),
		);
		expect(newFiles).toHaveLength(1); // only the broken file
		expect(Object.keys(files).some((k) => k.includes(".tmp."))).toBe(false);

		// Pre-existing ordering: index migration runs before the round-file write,
		// so a failed tmp write leaves the index already migrated (unit scope is the
		// atomic round write + write-new-then-delete-old, not index reordering).
		expect(files["/rounds/index.test-host"]).not.toContain(oldHash);
	});
});
