import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { computeContentHash } from "../lib/hash.ts";
import { encodeVectorIndexLine, loadVectorIndex, readIndexLines } from "../lib/index-io.ts";
import { embed, isMainModule, normalize, runDigestSession } from "./digest-session.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-digest-session-test-"));
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

function writeSession(filePath: string, userPrompt: string, responseSequence: string): void {
	fs.writeFileSync(
		filePath,
		[
			line({ type: "message", id: "u1", message: { role: "user", content: [{ type: "text", text: userPrompt }] } }),
			line({
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: responseSequence }] },
			}),
		].join("\n"),
	);
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

describe("digest-session script", () => {
	it("reports a missing session path without exiting the process", async () => {
		const logs = logger();

		await expect(runDigestSession({ stdout: logs.out, stderr: logs.err })).resolves.toBe(1);

		expect(logs.stderr).toEqual(["Usage: npx tsx scripts/digest-session.ts <session-file>"]);
		expect(logs.stdout).toEqual([]);

		const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
		try {
			await expect(runDigestSession({ stdout: logs.out })).resolves.toBe(1);
			expect(errorSpy).toHaveBeenCalledWith("Usage: npx tsx scripts/digest-session.ts <session-file>");
		} finally {
			errorSpy.mockRestore();
		}
	});

	it("writes parsed rounds and embeds prompt/response vectors", async () => {
		const root = tmpDir();
		const sessionFile = path.join(root, "session.jsonl");
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const userPrompt = `${"p".repeat(8100)}`;
		const responseSequence = `${"r".repeat(8100)}`;
		const requests: unknown[] = [];
		const logs = logger();
		writeSession(sessionFile, userPrompt, responseSequence);
		const fetchImpl = embeddingFetch(
			[
				[3, 4],
				[0, 0],
			],
			requests,
		);

		await expect(
			runDigestSession({ sessionFile, roundsDir, apiKey: "key", fetchImpl, stdout: logs.out, stderr: logs.err }),
		).resolves.toBe(0);

		const roundFile = `${computeContentHash(userPrompt, responseSequence, [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, roundFile))).toBe(true);
		expect(loadVectorIndex(indexPath)).toEqual([
			{ vector: [0.6, 0.8], filePath: `${roundFile}:prompt`, model: "openai/text-embedding-3-small" },
			{ vector: [0, 0], filePath: `${roundFile}:response`, model: "openai/text-embedding-3-small" },
		]);
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
		]);
		expect(logs.stdout.join("\n")).toContain(`📂 Session: ${sessionFile}`);
		expect(logs.stdout.join("\n")).toContain("📊 Parsed 1 rounds");
		expect(logs.stdout.join("\n")).toContain("🔄 Embedding round 1/1");
		expect(logs.stdout.join("\n")).toContain("1 new rounds embedded, 0 already in index");

		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;
		await expect(
			runDigestSession({ sessionFile, roundsDir, apiKey: "key", fetchImpl: secondFetch, stdout: logs.out }),
		).resolves.toBe(0);

		expect(secondFetch).not.toHaveBeenCalled();
		expect(readIndexLines(indexPath)).toHaveLength(2);
	});

	it("uses shared config for script rounds dir, embedding endpoint, model, and clipping", async () => {
		const root = tmpDir();
		const sessionFile = path.join(root, "session.jsonl");
		const configuredRoundsDir = path.join(root, "configured-rounds");
		fs.mkdirSync(path.join(root, ".pi"), { recursive: true });
		fs.writeFileSync(
			path.join(root, ".pi", "settings.json"),
			JSON.stringify({
				semblr: {
					roundsDir: "configured-rounds",
					embeddingModel: "configured-embedding-model",
					embeddingApiUrl: "https://embeddings.example/custom",
					embeddingMaxTokens: 4,
				},
			}),
		);
		writeSession(sessionFile, "123456", "abcdef");
		const requests: unknown[] = [];
		const fetchImpl = embeddingFetch([[1], [2]], requests);

		await expect(
			runDigestSession({
				sessionFile,
				apiKey: "key",
				fetchImpl,
				stdout: logger().out,
				configDeps: { cwd: root, agentDir: path.join(root, "agent"), env: {} },
			}),
		).resolves.toBe(0);

		const roundFile = `${computeContentHash("123456", "abcdef", [])}.json`;
		expect(fs.existsSync(path.join(configuredRoundsDir, roundFile))).toBe(true);
		expect(loadVectorIndex(path.join(configuredRoundsDir, "index.csv"))).toEqual([
			{ vector: [1], filePath: `${roundFile}:prompt`, model: "configured-embedding-model" },
			{ vector: [1], filePath: `${roundFile}:response`, model: "configured-embedding-model" },
		]);
		expect(requests).toEqual([
			{
				input: "https://embeddings.example/custom",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "configured-embedding-model", input: "1234" },
			},
			{
				input: "https://embeddings.example/custom",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "configured-embedding-model", input: "abcd" },
			},
		]);
	});

	it("migrates stale round filenames before deciding a round is already indexed", async () => {
		const root = tmpDir();
		const sessionFile = path.join(root, "session.jsonl");
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		fs.mkdirSync(roundsDir, { recursive: true });
		writeSession(sessionFile, "same prompt", "same response");
		fs.writeFileSync(
			path.join(roundsDir, "legacy.json"),
			JSON.stringify({ userPrompt: "same prompt", responseSequence: "same response", toolCalls: [] }),
		);
		fs.writeFileSync(indexPath, `${encodeVectorIndexLine([1], "legacy.json:prompt")}\n`);
		const logs = logger();
		const fetchImpl = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runDigestSession({
				sessionFile,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const roundFile = `${computeContentHash("same prompt", "same response", [])}.json`;
		expect(fs.existsSync(path.join(roundsDir, "legacy.json"))).toBe(false);
		expect(fs.existsSync(path.join(roundsDir, roundFile))).toBe(true);
		expect(readIndexLines(indexPath)).toEqual([encodeVectorIndexLine([1], `${roundFile}:prompt`)]);
		expect(logs.stdout.join("\n")).toContain(`Migrated stale: legacy.json → ${roundFile}`);
		expect(logs.stdout.join("\n")).toContain("0 new rounds embedded, 1 already in index");
		expect(fetchImpl).not.toHaveBeenCalled();
	});

	it("surfaces embedding setup and API failures", async () => {
		await expect(embed("text", { apiKey: "" })).rejects.toThrow("OPENROUTER_API_KEY");

		const fetchImpl = vi.fn(async () => new Response("bad", { status: 500 })) as typeof fetch;
		await expect(embed("text", { apiKey: "key", fetchImpl })).rejects.toThrow("Embedding API error 500: bad");
	});

	it("uses environment and fetch defaults for embedding", async () => {
		const oldKey = process.env.OPENROUTER_API_KEY;
		const fetchSpy = vi.fn(async () => new Response(JSON.stringify({ data: [{ embedding: [9] }] }))) as typeof fetch;
		const oldFetch = globalThis.fetch;
		process.env.OPENROUTER_API_KEY = "env-key";
		globalThis.fetch = fetchSpy;
		try {
			await expect(embed("text from env")).resolves.toEqual([9]);
		} finally {
			if (oldKey === undefined) delete process.env.OPENROUTER_API_KEY;
			else process.env.OPENROUTER_API_KEY = oldKey;
			globalThis.fetch = oldFetch;
		}
		expect(fetchSpy).toHaveBeenCalledWith(
			"https://openrouter.ai/api/v1/embeddings",
			expect.objectContaining({ headers: expect.objectContaining({ Authorization: "Bearer env-key" }) }),
		);
	});

	it("normalizes vectors and detects direct CLI execution", () => {
		expect(normalize([0, 0])).toEqual([0, 0]);
		expect(normalize([6, 8])).toEqual([0.6, 0.8]);
		expect(isMainModule("file:///tmp/digest-session.ts", "/tmp/digest-session.ts")).toBe(true);
		expect(isMainModule("file:///tmp/digest-session.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/digest-session.ts", "")).toBe(false);
	});
});
