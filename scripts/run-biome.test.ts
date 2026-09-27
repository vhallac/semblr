import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const shim = fileURLToPath(new URL("./run-biome.sh", import.meta.url));

const tempRoots: string[] = [];

afterEach(() => {
	for (const root of tempRoots.splice(0)) {
		rmSync(root, { recursive: true, force: true });
	}
});

/**
 * Creates a directory containing a fake `biome` executable that appends its
 * arguments to `logPath` on every invocation. `exitsZero` simulates a runnable
 * binary (e.g. the npm-installed one on CI, or the environment one on NixOS);
 * otherwise a binary that cannot run (e.g. the npm one on NixOS).
 */
function makeFakeBiome(logPath: string, exitsZero: boolean): string {
	const root = mkdtempSync(join(tmpdir(), "run-biome-test-"));
	tempRoots.push(root);
	const bin = join(root, "biome");
	writeFileSync(bin, `#!/bin/sh\nprintf '%s\\n' "$*" >> '${logPath}'\n${exitsZero ? "exit 0" : "exit 1"}\n`);
	chmodSync(bin, 0o755);
	return root;
}

function logLines(logPath: string): string[] {
	try {
		return readFileSync(logPath, "utf8").split("\n").filter(Boolean);
	} catch {
		return [];
	}
}

function runShim(pathDirs: string[], args: string[]) {
	return spawnSync("/bin/sh", [shim, ...args], {
		encoding: "utf8",
		env: { ...process.env, PATH: pathDirs.join(":") },
	});
}

describe("scripts/run-biome.sh", () => {
	it("skips a biome that cannot run and delegates to the next candidate, forwarding arguments", () => {
		const root = mkdtempSync(join(tmpdir(), "run-biome-test-"));
		tempRoots.push(root);
		const brokenLog = join(root, "broken.log");
		const workingLog = join(root, "working.log");
		const brokenDir = makeFakeBiome(brokenLog, false);
		const workingDir = makeFakeBiome(workingLog, true);

		const result = runShim([brokenDir, workingDir], ["check", "--write"]);

		expect(result.status).toBe(0);
		// The working biome receives a `--version` probe, then the real arguments.
		expect(logLines(workingLog)).toEqual(["--version", "check --write"]);
		// The broken biome is probed with `--version` only, never delegated to.
		expect(logLines(brokenLog)).toEqual(["--version"]);
	});

	it("uses the first runnable biome on PATH", () => {
		const root = mkdtempSync(join(tmpdir(), "run-biome-test-"));
		tempRoots.push(root);
		const firstLog = join(root, "first.log");
		const secondLog = join(root, "second.log");
		const firstDir = makeFakeBiome(firstLog, true);
		const secondDir = makeFakeBiome(secondLog, true);

		const result = runShim([firstDir, secondDir], ["check"]);

		expect(result.status).toBe(0);
		expect(logLines(firstLog)).toContain("check");
		expect(logLines(secondLog)).toEqual([]);
	});

	it("fails with exit code 127 when no runnable biome exists", () => {
		const root = mkdtempSync(join(tmpdir(), "run-biome-test-"));
		tempRoots.push(root);
		const brokenLog = join(root, "broken.log");
		const brokenDir = makeFakeBiome(brokenLog, false);

		const result = runShim([brokenDir], ["check"]);

		expect(result.status).toBe(127);
		expect(result.stderr).toContain("no runnable 'biome'");
	});
});
