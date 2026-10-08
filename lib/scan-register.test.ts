import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { computeMaxMtime, loadScanCutoff, saveScanCutoff, scanRegisterPath } from "./scan-register.ts";

describe("scan-register", () => {
	let tmp: string;
	let stateDir: string;
	let sessionDir: string;

	beforeEach(() => {
		tmp = fs.mkdtempSync(path.join(os.tmpdir(), "scan-register-"));
		stateDir = path.join(tmp, "state");
		sessionDir = path.join(tmp, "sessions");
		fs.mkdirSync(sessionDir);
	});
	afterEach(() => {
		fs.rmSync(tmp, { recursive: true, force: true });
	});

	it("round-trips the cutoff as epoch milliseconds", () => {
		expect(loadScanCutoff(stateDir, sessionDir)).toBeNull();
		saveScanCutoff(stateDir, sessionDir, 1_700_000_000_123);
		expect(loadScanCutoff(stateDir, sessionDir)).toBe(1_700_000_000_123);
	});

	it("keys the register by the resolved session dir path", () => {
		saveScanCutoff(stateDir, sessionDir, 42);
		const other = path.join(tmp, "other-sessions");
		fs.mkdirSync(other);
		expect(loadScanCutoff(stateDir, other)).toBeNull();
		expect(path.basename(scanRegisterPath(stateDir, sessionDir))).toMatch(/^scan-register-[0-9a-f]{16}\.json$/);
	});

	it("treats a corrupt register as no cutoff (full-scan fallback)", () => {
		fs.mkdirSync(stateDir, { recursive: true });
		fs.writeFileSync(scanRegisterPath(stateDir, sessionDir), "{not json");
		expect(loadScanCutoff(stateDir, sessionDir)).toBeNull();
	});

	it("rejects a register whose lastFullScanMtime is not a finite number", () => {
		fs.mkdirSync(stateDir, { recursive: true });
		for (const bad of [null, "123", { nested: true }, "abc"]) {
			fs.writeFileSync(scanRegisterPath(stateDir, sessionDir), JSON.stringify({ lastFullScanMtime: bad }));
			expect(loadScanCutoff(stateDir, sessionDir), String(bad)).toBeNull();
		}
	});

	describe("computeMaxMtime", () => {
		it("returns the max finite mtime across files", () => {
			const a = path.join(sessionDir, "a.jsonl");
			const b = path.join(sessionDir, "b.jsonl");
			fs.writeFileSync(a, "x");
			fs.writeFileSync(b, "y");
			fs.utimesSync(a, new Date(1_000), new Date(1_000));
			fs.utimesSync(b, new Date(2_000), new Date(2_000));
			expect(computeMaxMtime([a, b])).toBe(2_000);
			expect(computeMaxMtime([b, a])).toBe(2_000);
		});

		it("skips unstatable files and returns null when nothing stats", () => {
			expect(computeMaxMtime([path.join(sessionDir, "missing.jsonl")])).toBeNull();
			const existing = path.join(sessionDir, "here.jsonl");
			fs.writeFileSync(existing, "x");
			fs.utimesSync(existing, new Date(5_000), new Date(5_000));
			expect(computeMaxMtime([path.join(sessionDir, "missing.jsonl"), existing])).toBe(5_000);
		});

		it("ignores non-finite mtimes from a stubbed fsImpl", () => {
			const stub = {
				statSync: (file: fs.PathLike) => {
					if (file === "bad") return { mtimeMs: Number.NaN } as fs.Stats;
					if (file === "inf") return { mtimeMs: Number.POSITIVE_INFINITY } as fs.Stats;
					throw new Error("enoent");
				},
			};
			expect(computeMaxMtime(["bad", "inf"], stub as unknown as Pick<typeof fs, "statSync">)).toBeNull();
		});
	});

	it("saves best-effort: a read-only stateDir does not throw", () => {
		const ro = fs.mkdtempSync(path.join(os.tmpdir(), "scan-register-ro-"));
		fs.chmodSync(ro, 0o555);
		try {
			expect(() => saveScanCutoff(ro, sessionDir, 42)).not.toThrow();
		} finally {
			fs.chmodSync(ro, 0o755);
			fs.rmSync(ro, { recursive: true, force: true });
		}
	});
});
