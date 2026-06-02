import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { isMainModule, runResponseSegmentsMigration } from "./migrate-rounds.ts";

function tmpDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "semblr-migrate-rounds-test-"));
}

function logger() {
	const stdout: string[] = [];
	const stderr: string[] = [];
	return {
		stdout,
		stderr,
		out: { log: (line: string) => stdout.push(line) },
		err: {
			error: (line: string) => stderr.push(line),
			warn: (line: string) => stderr.push(line),
		},
	};
}

function writeRound(roundsDir: string, name: string, data: unknown): void {
	fs.writeFileSync(path.join(roundsDir, name), JSON.stringify(data));
}

describe("migrate-rounds script", () => {
	it("reports a missing rounds directory without exiting the process", () => {
		const logs = logger();
		const missing = path.join(tmpDir(), "missing");

		expect(runResponseSegmentsMigration({ roundsDir: missing, stdout: logs.out, stderr: logs.err })).toBe(1);

		expect(logs.stderr).toEqual([expect.stringContaining("Rounds directory not found")]);
		expect(logs.stdout).toEqual([]);
	});

	it("uses console output defaults when loggers are not injected", () => {
		const roundsDir = tmpDir();
		const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

		try {
			expect(runResponseSegmentsMigration({ roundsDir })).toBe(0);
			expect(logSpy).toHaveBeenCalledWith(expect.stringContaining("0 round files migrated"));
		} finally {
			logSpy.mockRestore();
		}
	});

	it("migrates response text and tool calls while preserving existing segmented rounds", () => {
		const roundsDir = tmpDir();
		const logs = logger();
		writeRound(roundsDir, "text-and-tools.json", {
			userPrompt: "question",
			responseSequence: "answer",
			toolCalls: [
				{ index: 0, name: "read", arguments: "{}", result_summary: "ok" },
				{ index: 1, name: "bash", arguments: "{}", result_summary: "done" },
			],
		});
		writeRound(roundsDir, "tools-only.json", {
			userPrompt: "question",
			responseSequence: "   ",
			toolCalls: [{ index: 0, name: "read", arguments: "{}", result_summary: "ok" }],
		});
		writeRound(roundsDir, "already.json", {
			userPrompt: "question",
			responseSequence: "answer",
			responseSegments: [{ type: "text", text: "existing" }],
		});
		writeRound(roundsDir, "empty-segments.json", {
			userPrompt: "question",
			responseSequence: "answer after empty segments",
			responseSegments: [],
		});
		writeRound(roundsDir, "non-array-segments.json", {
			userPrompt: "question",
			responseSequence: "answer after malformed segments",
			responseSegments: { type: "text", text: "not an array" },
		});
		writeRound(roundsDir, "missing-response.json", {
			userPrompt: "question",
			toolCalls: [{ index: 0, name: "read", arguments: "{}", result_summary: "ok" }],
		});
		fs.writeFileSync(path.join(roundsDir, "notes.txt"), "not a round");

		expect(runResponseSegmentsMigration({ roundsDir, stdout: logs.out, stderr: logs.err })).toBe(0);

		expect(
			JSON.parse(fs.readFileSync(path.join(roundsDir, "text-and-tools.json"), "utf-8")).responseSegments,
		).toEqual([
			{ type: "text", text: "answer" },
			{ type: "toolCall", toolCallIndex: 0 },
			{ type: "toolCall", toolCallIndex: 1 },
		]);
		expect(JSON.parse(fs.readFileSync(path.join(roundsDir, "tools-only.json"), "utf-8")).responseSegments).toEqual([
			{ type: "toolCall", toolCallIndex: 0 },
		]);
		expect(JSON.parse(fs.readFileSync(path.join(roundsDir, "already.json"), "utf-8")).responseSegments).toEqual([
			{ type: "text", text: "existing" },
		]);
		expect(
			JSON.parse(fs.readFileSync(path.join(roundsDir, "empty-segments.json"), "utf-8")).responseSegments,
		).toEqual([{ type: "text", text: "answer after empty segments" }]);
		expect(
			JSON.parse(fs.readFileSync(path.join(roundsDir, "non-array-segments.json"), "utf-8")).responseSegments,
		).toEqual([{ type: "text", text: "answer after malformed segments" }]);
		expect(
			JSON.parse(fs.readFileSync(path.join(roundsDir, "missing-response.json"), "utf-8")).responseSegments,
		).toEqual([{ type: "toolCall", toolCallIndex: 0 }]);
		expect(logs.stdout).toEqual(["\n✅ Done. 5 round files migrated, 1 already had responseSegments."]);
		expect(logs.stderr).toEqual([]);
	});

	it("skips unparseable and empty rounds without rewriting them", () => {
		const roundsDir = tmpDir();
		const logs = logger();
		writeRound(roundsDir, "empty.json", { userPrompt: "question", responseSequence: "" });
		fs.writeFileSync(path.join(roundsDir, "bad.json"), "not json");

		expect(runResponseSegmentsMigration({ roundsDir, stdout: logs.out, stderr: logs.err })).toBe(0);

		expect(JSON.parse(fs.readFileSync(path.join(roundsDir, "empty.json"), "utf-8"))).not.toHaveProperty(
			"responseSegments",
		);
		expect(fs.readFileSync(path.join(roundsDir, "bad.json"), "utf-8")).toBe("not json");
		expect(logs.stderr.join("\n")).toContain("Skipping empty: empty.json");
		expect(logs.stderr.join("\n")).toContain("Skipping unparseable: bad.json");
		expect(logs.stdout.join("\n")).toContain("0 round files migrated, 0 already had responseSegments");
	});

	it("detects direct CLI execution", () => {
		expect(isMainModule("file:///tmp/script.ts", "/tmp/script.ts")).toBe(true);
		expect(isMainModule("file:///tmp/script.ts", "/tmp/other.ts")).toBe(false);
		expect(isMainModule("file:///tmp/script.ts", undefined)).toBe(false);
	});
});
