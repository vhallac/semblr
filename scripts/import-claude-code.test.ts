import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { claudeRoundFileName } from "../lib/claude-code.ts";
import { loadVectorIndex } from "../lib/index-io.ts";
import { isMainModule, runImportClaudeCode } from "./import-claude-code.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-import-claude-test-"));
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

function writeClaudeJsonl(filePath: string, entries: Array<Record<string, any>>): void {
	fs.writeFileSync(filePath, entries.map((e) => line({ timestamp: e.timestamp ?? 1000, ...e })).join("\n"));
}

function validUserEntry(overrides: Record<string, any> = {}): Record<string, any> {
	return {
		type: "user",
		timestamp: 1000,
		cwd: "/home/user",
		sessionId: "abc",
		gitBranch: "main",
		message: { role: "user", content: [{ type: "text", text: "User prompt" }] },
		...overrides,
	};
}

function RESPONSE(mm = false): string {
	return mm
		? "This is a sufficiently long assistant response. It has enough characters."
		: "Assistant response — long enough to pass the minimum length check.";
}

function validAssistantEntry(overrides: Record<string, any> = {}): Record<string, any> {
	return {
		type: "assistant",
		timestamp: 2000,
		message: { role: "assistant", content: [{ type: "text", text: RESPONSE() }] },
		...overrides,
	};
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

describe("import-claude-code script", () => {
	it("reports missing API key when not in dry-run mode", async () => {
		const logs = logger();
		const projectsDir = tmpDir();
		writeClaudeJsonl(path.join(projectsDir, "test.jsonl"), [validUserEntry(), validAssistantEntry()]);

		const exitCode = await runImportClaudeCode({
			projectsDir,
			roundsDir: tmpDir(),
			dryRun: false,
			apiKey: "",
			stdout: logs.out,
			stderr: logs.err,
		});

		expect(exitCode).toBe(1);
		expect(logs.stderr).toContain("❌ OPENROUTER_API_KEY required");
	});

	it("dry-run counts rounds without writing or embedding", async () => {
		const logs = logger();
		const projectsDir = tmpDir();
		const roundsDir = tmpDir();
		const indexPath = path.join(roundsDir, "index.csv");

		writeClaudeJsonl(path.join(projectsDir, "session.jsonl"), [
			validUserEntry({ cwd: "/home/user/project", gitBranch: "feature/x" }),
			validAssistantEntry(),
			validUserEntry({
				cwd: "/home/user/project2",
				message: { role: "user", content: [{ type: "text", text: "A different user prompt" }] },
			}),
			validAssistantEntry(),
		]);

		const fetchSpy = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				indexPath,
				dryRun: true,
				fetchImpl: fetchSpy,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(fetchSpy).not.toHaveBeenCalled();
		expect(fs.existsSync(indexPath)).toBe(false);
		if (fs.existsSync(roundsDir)) {
			expect(fs.readdirSync(roundsDir).filter((f) => f.endsWith(".json"))).toHaveLength(0);
		}
		expect(logs.stdout.join("\n")).toContain("📊 Parsed rounds: 2");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to import: 2");
	});

	it("writes round files and embeds prompt/response vectors", async () => {
		const root = tmpDir();
		const projectsDir = path.join(root, "projects");
		fs.mkdirSync(projectsDir, { recursive: true });
		const roundsDir = path.join(root, "rounds");
		const indexPath = path.join(roundsDir, "index.csv");
		const logs = logger();
		const requests: unknown[] = [];

		const userPrompt = "What is TypeScript?";
		const responseSequence = "TypeScript is a typed superset of JavaScript.";

		writeClaudeJsonl(path.join(projectsDir, "session.jsonl"), [
			validUserEntry({
				cwd: "/home/user/work",
				gitBranch: "main",
				message: { role: "user", content: [{ type: "text", text: userPrompt }] },
			}),
			validAssistantEntry({
				message: { role: "assistant", content: [{ type: "text", text: responseSequence }] },
			}),
		]);

		const fetchImpl = embeddingFetch(
			[
				[3, 4],
				[6, 8],
			],
			requests,
		);

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl,
				concurrency: 1,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		const roundFile = claudeRoundFileName({ userPrompt, responseSequence });
		expect(fs.existsSync(path.join(roundsDir, roundFile))).toBe(true);

		const writtenRound = JSON.parse(fs.readFileSync(path.join(roundsDir, roundFile), "utf-8"));
		expect(writtenRound.source).toBe("claude-code");
		expect(writtenRound.userPrompt).toBe(userPrompt);
		expect(writtenRound.responseSequence).toBe(responseSequence);
		expect(writtenRound.cwd).toBe("/home/user/work");
		expect(writtenRound.gitBranch).toBe("main");

		expect(loadVectorIndex(indexPath)).toEqual([
			{ vector: [0.6, 0.8], filePath: `${roundFile}:prompt` },
			{ vector: [0.6, 0.8], filePath: `${roundFile}:response` },
		]);

		expect(requests).toEqual([
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: userPrompt },
			},
			{
				input: "https://openrouter.ai/api/v1/embeddings",
				method: "POST",
				headers: { Authorization: "Bearer key", "Content-Type": "application/json" },
				body: { model: "openai/text-embedding-3-small", input: responseSequence },
			},
		]);

		expect(logs.stdout.join("\n")).toContain("📂 Found 1 Claude Code JSONL files");
		expect(logs.stdout.join("\n")).toContain("📊 Parsed rounds: 1");
		expect(logs.stdout.join("\n")).toContain("✅ Done");
	});

	it("respects limit option", async () => {
		const logs = logger();
		const projectsDir = tmpDir();
		const roundsDir = tmpDir();
		const indexPath = path.join(roundsDir, "index.csv");
		const requests: unknown[] = [];

		const entries: Record<string, any>[] = [];
		for (let i = 0; i < 5; i++) {
			entries.push(
				validUserEntry({
					message: { role: "user", content: [{ type: "text", text: `Prompt ${i}` }] },
				}),
				validAssistantEntry({
					message: {
						role: "assistant",
						content: [{ type: "text", text: `Response ${i} is long enough to pass the check.` }],
					},
				}),
			);
		}

		writeClaudeJsonl(path.join(projectsDir, "session.jsonl"), entries);

		const fetchImpl = embeddingFetch(
			[
				[1, 0],
				[0, 1],
				[1, 1],
				[0, 0],
			],
			requests,
		);

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				indexPath,
				limit: 2,
				apiKey: "key",
				fetchImpl,
				concurrency: 1,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(logs.stdout.join("\n")).toContain("📊 Parsed rounds: 5");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to import: 2");
		expect(loadVectorIndex(indexPath)).toHaveLength(4); // 2 prompt + 2 response
	});

	it("skips already-indexed rounds on second run", async () => {
		const logs = logger();
		const projectsDir = tmpDir();
		const roundsDir = tmpDir();
		const indexPath = path.join(roundsDir, "index.csv");

		writeClaudeJsonl(path.join(projectsDir, "session.jsonl"), [
			validUserEntry({
				message: { role: "user", content: [{ type: "text", text: "Prompt A" }] },
			}),
			validAssistantEntry({
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Response A is the answer to Prompt A, long enough to pass." }],
				},
			}),
		]);

		const firstFetch = embeddingFetch(
			[
				[1, 0],
				[0, 1],
			],
			[],
		);

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: firstFetch,
				concurrency: 1,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(firstFetch).toHaveBeenCalledTimes(2);

		const secondLogs = logger();
		const secondFetch = vi.fn(async () => new Response("should not be called")) as typeof fetch;

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				indexPath,
				apiKey: "key",
				fetchImpl: secondFetch,
				stdout: secondLogs.out,
				stderr: secondLogs.err,
			}),
		).resolves.toBe(0);

		expect(secondFetch).not.toHaveBeenCalled();
		expect(secondLogs.stdout.join("\n")).toContain("📊 New rounds to import: 0 (1 already indexed");
	});

	it("skips sidechains by default", async () => {
		const logs = logger();
		const projectsDir = tmpDir();
		const roundsDir = tmpDir();

		writeClaudeJsonl(path.join(projectsDir, "session.jsonl"), [
			validUserEntry({
				message: { role: "user", content: [{ type: "text", text: "Main prompt — what does this do?" }] },
			}),
			validAssistantEntry(),
			{
				type: "user",
				timestamp: 3000,
				isSidechain: true,
				message: { role: "user", content: [{ type: "text", text: "Sidechain prompt" }] },
			},
			{
				type: "assistant",
				timestamp: 4000,
				isSidechain: true,
				message: { role: "assistant", content: [{ type: "text", text: "Sidechain response" }] },
			},
		]);

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				dryRun: true,
				includeSidechains: false,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(logs.stdout.join("\n")).toContain("📊 Parsed rounds: 1");
	});

	it("handles JSONL parse errors gracefully", async () => {
		const logs = logger();
		const projectsDir = tmpDir();
		const roundsDir = tmpDir();
		const indexPath = path.join(roundsDir, "index.csv");

		// Good file
		fs.writeFileSync(
			path.join(projectsDir, "good.jsonl"),
			[
				line(validUserEntry({ message: { role: "user", content: [{ type: "text", text: "Good prompt" }] } })),
				line(
					validAssistantEntry({
						message: {
							role: "assistant",
							content: [
								{ type: "text", text: "Good response — this is the helpful answer the assistant gave." },
							],
						},
					}),
				),
			].join("\n"),
		);
		// Corrupt file — parseClaudeCodeJsonl throws on invalid JSON, caught at file level
		fs.writeFileSync(path.join(projectsDir, "bad.jsonl"), "not valid json {{{\n");

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				indexPath,
				dryRun: true,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(logs.stdout.join("\n")).toContain("📊 Parsed rounds: 1");
		expect(logs.stdout.join("\n")).toContain("1 parse errors");
		expect(logs.stderr.some((l) => l.includes("⚠") && l.includes("bad.jsonl"))).toBe(true);
	});

	it("handles empty projects directory", async () => {
		const logs = logger();
		const projectsDir = tmpDir();
		const roundsDir = tmpDir();

		await expect(
			runImportClaudeCode({
				projectsDir,
				roundsDir,
				dryRun: true,
				stdout: logs.out,
				stderr: logs.err,
			}),
		).resolves.toBe(0);

		expect(logs.stdout.join("\n")).toContain("📂 Found 0 Claude Code JSONL files");
		expect(logs.stdout.join("\n")).toContain("📊 New rounds to import: 0");
	});

	it("detects direct CLI execution", () => {
		expect(isMainModule("file:///tmp/import-claude-code.ts", "/tmp/import-claude-code.ts")).toBe(true);
		expect(isMainModule("file:///tmp/import-claude-code.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/import-claude-code.ts", undefined)).toBe(false);
	});
});
