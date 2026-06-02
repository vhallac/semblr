import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { computeContentHash } from "../lib/hash.ts";
import { encodeVectorIndexLine, loadVectorIndex, readIndexLines } from "../lib/index-io.ts";
import { isMainModule, runDigestAll } from "./digest-all.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-digest-all-test-"));
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

function line(value: unknown): string {
	return JSON.stringify(value);
}

/**
 * Write a pi-style session JSONL file with user + assistant round pairs.
 * Each pair is one user message followed by one assistant message.
 */
function writeSession(filePath: string, pairs: Array<{ userPrompt: string; responseSequence: string }>): void {
	const entries: unknown[] = [];
	for (let i = 0; i < pairs.length; i++) {
		entries.push({
			type: "message",
			id: `u${i}`,
			message: { role: "user", content: [{ type: "text", text: pairs[i].userPrompt }] },
		});
		entries.push({
			type: "message",
			id: `a${i}`,
			message: {
				role: "assistant",
				content: [{ type: "text", text: pairs[i].responseSequence }],
			},
		});
	}
	fs.writeFileSync(filePath, entries.map(line).join("\n"));
}

function embeddingFetch(vectors: number[][], requests: unknown[] = []): typeof fetch {
	return vi.fn(async (input, init) => {
		requests.push({
			input,
			method: init?.method,
			headers: init?.headers,
			body: JSON.parse(String(init?.body)),
		});
		return new Response(JSON.stringify({ data: [{ embedding: vectors.shift() ?? [1] }] }), { status: 200 });
	}) as typeof fetch;
}

describe("digest-all script", () => {
	it("reports missing API key", async () => {
		const logs = logger();
		const sessionsDir = tmpDir();
		const roundsDir = tmpDir();

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				apiKey: "",
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(1);

		expect(logs.stderr).toContain("❌ OPENROUTER_API_KEY environment variable required");
	});

	it("handles empty or nonexistent sessions directory", async () => {
		const logs = logger();
		const sessionsDir = tmpDir();
		const roundsDir = tmpDir();

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				apiKey: "key",
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(logs.stdout.join("\n")).toContain("📂 Found 0 session files");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to embed: 0");
	});

	it("processes multiple sessions, embeds prompts and responses, and writes round files", async () => {
		const root = tmpDir();
		// Create a sessions directory structure: --session1/file.jsonl, --session2/file.jsonl
		const sessionsDir = path.join(root, "sessions");
		const s1Dir = path.join(sessionsDir, "--session1");
		const s2Dir = path.join(sessionsDir, "--session2");
		fs.mkdirSync(s1Dir, { recursive: true });
		fs.mkdirSync(s2Dir, { recursive: true });

		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const logs = logger();
		const requests: unknown[] = [];

		const userPrompt1 = `${"p".repeat(8100)}`;
		const resp1 = `${"r".repeat(8100)}`;
		const userPrompt2 = "What is the answer to life?";
		const resp2 = "The answer is forty two.";

		writeSession(path.join(s1Dir, "session.jsonl"), [{ userPrompt: userPrompt1, responseSequence: resp1 }]);
		writeSession(path.join(s2Dir, "session.jsonl"), [{ userPrompt: userPrompt2, responseSequence: resp2 }]);

		const fetchImpl = embeddingFetch(
			[
				[3, 4], // prompt1
				[0, 0], // resp1
				[1, 1], // prompt2
				[2, 2], // resp2
			],
			requests,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// Verify round files exist
		const roundFile1 = `${computeContentHash(userPrompt1, resp1, [])}.json`;
		const roundFile2 = `${computeContentHash(userPrompt2, resp2, [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, roundFile1))).toBe(true);
		expect(fs.existsSync(path.join(roundsDir, roundFile2))).toBe(true);

		// Verify index
		const index = loadVectorIndex(indexPath);
		expect(index).toHaveLength(4); // 2 prompts + 2 responses

		// Prompt text is truncated to 8000 chars
		expect(requests).toEqual([
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: "p".repeat(8000) },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: "r".repeat(8000) },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: userPrompt2 },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: resp2 },
			},
		]);

		expect(logs.stdout.join("\n")).toContain("📂 Found 2 session files across 2 directories");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to embed: 2");
		expect(logs.stdout.join("\n")).toContain("✅ Done");
	});

	it("skips already-indexed rounds on second run", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = "What is the capital of France?";
		const responseSequence = "The capital of France is Paris.";

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);

		// First run — embeds everything
		const firstFetch = embeddingFetch(
			[
				[1, 0],
				[0, 1],
			],
			[],
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: firstFetch,
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		expect(firstFetch).toHaveBeenCalledTimes(2);

		// Second run — skip all
		const secondLogs = logger();
		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: secondFetch,
				stdout: secondLogs.out,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);

		expect(secondFetch).not.toHaveBeenCalled();
		expect(secondLogs.stdout.join("\n")).toContain("📊 New rounds to embed: 0 (1 already indexed)");
		expect(secondLogs.stdout.join("\n")).toContain("✨ Nothing to do — all sessions already indexed!");
	});

	it("migrates stale round filenames before deciding a round is already indexed", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		fs.mkdirSync(roundsDir, { recursive: true });
		const indexPath = path.join(roundsDir, "index.csv");

		const userPrompt = "What is the same prompt?";
		const responseSequence = "The same prompt always returns the same answer.";

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt, responseSequence }]);

		// Pre-create a stale round file and index entry
		fs.writeFileSync(
			path.join(roundsDir, "legacy.json"),
			JSON.stringify({ userPrompt, responseSequence, toolCalls: [] }),
		);
		fs.writeFileSync(indexPath, `${encodeVectorIndexLine([1], "legacy.json:prompt")}\n`);

		const logs = logger();
		const fetchImpl = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, "legacy.json"))).toBe(false);
		expect(fs.existsSync(path.join(roundsDir, roundFile))).toBe(true);
		expect(readIndexLines(indexPath)).toEqual([encodeVectorIndexLine([1], `${roundFile}:prompt`)]);
		expect(logs.stderr.join("\n")).toContain(`Migrated stale: legacy.json → ${roundFile}`);
		expect(logs.stdout.join("\n")).toContain("1 rounds embedded, 0 errors");
		expect(fetchImpl).not.toHaveBeenCalled();
	});

	it("handles embedding API errors gracefully and continues processing", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const s1Dir = path.join(sessionsDir, "--session1");
		const s2Dir = path.join(sessionsDir, "--session2");
		fs.mkdirSync(s1Dir, { recursive: true });
		fs.mkdirSync(s2Dir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");

		const userPrompt1 = "What is a good prompt?";
		const resp1 = "A good prompt is clear and specific.";
		const userPrompt2 = "What is a failing prompt?";
		const resp2 = "A failing prompt causes an API error.";

		writeSession(path.join(s1Dir, "session.jsonl"), [{ userPrompt: userPrompt1, responseSequence: resp1 }]);
		writeSession(path.join(s2Dir, "session.jsonl"), [{ userPrompt: userPrompt2, responseSequence: resp2 }]);

		// First round: success. Second round: prompt fails, embeds response anyway.
		// But since our processRound tries prompt first, a prompt failure skips response embedding.
		// Let's make the first call (first round prompt) succeed, second call (first round response) succeed,
		// third call (second round prompt) fail.
		const logs = logger();
		const fetchImpl = vi.fn(async (_input: unknown, init: any) => {
			const body = JSON.parse(String((init as any).body));
			if (body.input === userPrompt2) {
				return new Response("server error", { status: 500 });
			}
			return new Response(JSON.stringify({ data: [{ embedding: [1, 2] }] }), { status: 200 });
		}) as typeof fetch;

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const roundFile1 = `${computeContentHash(userPrompt1, resp1, [])}.json`;
		const roundFile2 = `${computeContentHash(userPrompt2, resp2, [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, roundFile1))).toBe(true);
		expect(fs.existsSync(path.join(roundsDir, roundFile2))).toBe(true);

		expect(logs.stderr.join("\n")).toContain("[ERROR]");
		expect(logs.stdout.join("\n")).toContain("1 rounds embedded, 1 errors");
	});

	it("truncates long prompt and response text to MAX chars", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		const sDir = path.join(sessionsDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		const longPrompt = `${"A".repeat(10000)}`;
		const longResponse = `${"B".repeat(10000)}`;

		writeSession(path.join(sDir, "session.jsonl"), [{ userPrompt: longPrompt, responseSequence: longResponse }]);

		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
			],
			requests,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logger().out,
			}),
		).resolves.toBe(0);

		// Both prompt and response were truncated to 8000 chars
		expect(requests).toHaveLength(2);
		expect((requests[0] as any).body.input).toBe("A".repeat(8000));
		expect((requests[1] as any).body.input).toBe("B".repeat(8000));
	});

	it("skips non-session (non-dash-dash) directories", async () => {
		const root = tmpDir();
		const sessionsDir = path.join(root, "sessions");
		fs.mkdirSync(path.join(sessionsDir, "--real"), { recursive: true });
		fs.mkdirSync(path.join(sessionsDir, "not-a-session"), { recursive: true });

		// Put a session file in the real dir only
		writeSession(path.join(sessionsDir, "--real", "session.jsonl"), [
			{ userPrompt: "What is the capital?", responseSequence: "The capital of France is Paris." },
		]);
		// Put a file in the non-session dir that should be ignored
		writeSession(path.join(sessionsDir, "not-a-session", "session.jsonl"), [
			{ userPrompt: "ignored", responseSequence: "This response should never be seen." },
		]);

		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const logs = logger();
		const requests: unknown[] = [];

		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
			],
			requests,
		);

		await expect(
			runDigestAll({
				sessionsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		// Only the real session was picked up
		expect(logs.stdout.join("\n")).toContain("📂 Found 1 session files across 1 directories");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to embed: 1");
		expect(requests).toHaveLength(2);
	});

	it("uses environment and fetch defaults when options are not injected", async () => {
		const root = tmpDir();
		const _sessionsDir = tmpDir();
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");

		const oldKey = process.env.OPENROUTER_API_KEY;
		const oldFetch = globalThis.fetch;
		const fetchSpy = vi.fn(
			async () => new Response(JSON.stringify({ data: [{ embedding: [9, 9] }] }), { status: 200 }),
		) as typeof fetch;
		process.env.OPENROUTER_API_KEY = "env-key";
		globalThis.fetch = fetchSpy;
		const sessionsOnlyDir = path.join(root, "sessions");
		const sDir = path.join(sessionsOnlyDir, "--test");
		fs.mkdirSync(sDir, { recursive: true });
		writeSession(path.join(sDir, "session.jsonl"), [
			{ userPrompt: "prompt from env", responseSequence: "response from environment variable." },
		]);

		try {
			await expect(runDigestAll({ sessionsDir: sessionsOnlyDir, roundsDir, indexPath })).resolves.toBe(0);

			expect(fetchSpy).toHaveBeenCalledWith(
				"https://openrouter.ai/api/v1/embeddings",
				expect.objectContaining({
					headers: expect.objectContaining({ Authorization: "Bearer env-key" }),
				}),
			);
		} finally {
			if (oldKey === undefined) delete process.env.OPENROUTER_API_KEY;
			else process.env.OPENROUTER_API_KEY = oldKey;
			globalThis.fetch = oldFetch;
		}
	});

	it("detects direct CLI execution", () => {
		expect(isMainModule("file:///tmp/digest-all.ts", "/tmp/digest-all.ts")).toBe(true);
		expect(isMainModule("file:///tmp/digest-all.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/digest-all.ts", "")).toBe(false);
	});
});
