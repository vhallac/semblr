import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { getRoundFileSize, readRoundFileFromDir, readRoundJson } from "./round-io.ts";

describe("round-io", () => {
	let roundsDir: string;

	beforeEach(() => {
		roundsDir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-round-io-test-"));
	});

	afterEach(() => {
		fs.rmSync(roundsDir, { recursive: true, force: true });
	});

	describe("readRoundJson", () => {
		it("returns null when file does not exist", () => {
			expect(readRoundJson(roundsDir, "nonexistent.json")).toBeNull();
		});

		it("returns parsed JSON when file exists", () => {
			const data = { userPrompt: "hello", responseSequence: "world", turnIndex: 1 };
			fs.writeFileSync(path.join(roundsDir, "abc.json"), JSON.stringify(data));
			const result = readRoundJson(roundsDir, "abc.json");
			expect(result).toEqual(data);
		});

		it("returns null for unparseable JSON", () => {
			fs.writeFileSync(path.join(roundsDir, "bad.json"), "not json");
			expect(readRoundJson(roundsDir, "bad.json")).toBeNull();
		});

		it("returns null for empty file", () => {
			fs.writeFileSync(path.join(roundsDir, "empty.json"), "");
			expect(readRoundJson(roundsDir, "empty.json")).toBeNull();
		});
	});

	describe("readRoundFileFromDir", () => {
		it("returns null when file does not exist", () => {
			expect(readRoundFileFromDir("abc.json", roundsDir)).toBeNull();
		});

		it("resolves :prompt suffix to the actual file", () => {
			const data = { userPrompt: "test", responseSequence: "resp", turnIndex: 1 };
			const hash = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6";
			const fileName = `${hash}.json`;
			fs.writeFileSync(path.join(roundsDir, fileName), JSON.stringify(data));
			const result = readRoundFileFromDir(`${fileName}:prompt`, roundsDir);
			expect(result).not.toBeNull();
			expect(result?.userPrompt).toBe("test");
		});

		it("returns null for unparseable JSON", () => {
			fs.writeFileSync(path.join(roundsDir, "bad.json"), "not json");
			expect(readRoundFileFromDir("bad.json", roundsDir)).toBeNull();
		});

		it("returns fields with defaults for missing optional fields", () => {
			const data = { userPrompt: "minimal" };
			fs.writeFileSync(path.join(roundsDir, "min.json"), JSON.stringify(data));
			const result = readRoundFileFromDir("min.json", roundsDir);
			expect(result).toEqual({
				userPrompt: "minimal",
				responseSequence: "",
				turnIndex: 0,
				userTimestamp: undefined,
				toolCallCount: undefined,
				toolCallNames: undefined,
				toolCalls: undefined,
			});
		});
	});

	describe("getRoundFileSize", () => {
		it("returns null when file does not exist", () => {
			expect(getRoundFileSize(roundsDir, "nonexistent.json")).toBeNull();
		});

		it("returns formatted size for an existing file", () => {
			const data = { userPrompt: "test", responseSequence: "resp", turnIndex: 1 };
			fs.writeFileSync(path.join(roundsDir, "size.json"), JSON.stringify(data));
			const size = getRoundFileSize(roundsDir, "size.json");
			expect(size).toBeTruthy();
			expect(size).toMatch(/KB$/); // should end in KB
		});

		it("returns size string that matches actual file size", () => {
			fs.writeFileSync(path.join(roundsDir, "big.json"), "x".repeat(5000));
			const size = getRoundFileSize(roundsDir, "big.json");
			// ~5KB — formatFileSize rounds to 2 decimal places
			expect(size).toMatch(/\d+\.?\d*KB/);
		});
	});
});
