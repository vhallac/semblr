import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { atomicWriteFileSync } from "./atomic-write.ts";

describe("atomicWriteFileSync", () => {
	let dir: string;

	beforeEach(() => {
		dir = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-atomic-write-test-"));
	});

	afterEach(() => {
		fs.rmSync(dir, { recursive: true, force: true });
	});

	it("writes the target content atomically (tmp+rename ordering)", () => {
		const target = path.join(dir, "round.json");
		atomicWriteFileSync(target, '{"a":1}');
		expect(fs.readFileSync(target, "utf-8")).toBe('{"a":1}');
		// No tmp leftovers after success
		expect(fs.readdirSync(dir).filter((f) => f.includes(".tmp."))).toEqual([]);
	});

	it("overwrites an existing target byte-for-byte", () => {
		const target = path.join(dir, "round.json");
		fs.writeFileSync(target, "old-content-that-must-be-replaced");
		atomicWriteFileSync(target, "new-content");
		expect(fs.readFileSync(target, "utf-8")).toBe("new-content");
	});

	it("leaves the target byte-identical when the tmp write throws", () => {
		const target = path.join(dir, "round.json");
		const original = "original-round-bytes";
		fs.writeFileSync(target, original);

		const failing = {
			writeFileSync(tmpPath: string) {
				// Simulate a crash mid-write: tmp may even contain partial bytes
				fs.writeFileSync(tmpPath, "partial");
				throw new Error("simulated write failure");
			},
			renameSync: fs.renameSync.bind(fs),
		};

		expect(() => atomicWriteFileSync(target, "new", failing)).toThrow("simulated write failure");
		expect(fs.readFileSync(target, "utf-8")).toBe(original);
	});

	it("leaves the target byte-identical when the rename throws", () => {
		const target = path.join(dir, "round.json");
		const original = "original-round-bytes";
		fs.writeFileSync(target, original);

		const failing = {
			writeFileSync: fs.writeFileSync.bind(fs),
			renameSync() {
				throw new Error("simulated rename failure");
			},
		};

		expect(() => atomicWriteFileSync(target, "new", failing)).toThrow("simulated rename failure");
		expect(fs.readFileSync(target, "utf-8")).toBe(original);
	});

	it("never touches the target when it did not exist and the write throws", () => {
		const target = path.join(dir, "round.json");
		const failing = {
			writeFileSync() {
				throw new Error("boom");
			},
			renameSync: fs.renameSync.bind(fs),
		};
		expect(() => atomicWriteFileSync(target, "new", failing)).toThrow("boom");
		expect(fs.existsSync(target)).toBe(false);
	});
});
