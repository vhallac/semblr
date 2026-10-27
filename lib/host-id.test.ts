import { describe, expect, it } from "vitest";
import { resolveHostId, sanitizeHostId } from "./host-id.ts";

describe("sanitizeHostId", () => {
	it("rejects empty, dot-only and path-bearing ids", () => {
		expect(sanitizeHostId(undefined)).toBeUndefined();
		expect(sanitizeHostId("  ")).toBeUndefined();
		expect(sanitizeHostId(".")).toBeUndefined();
		expect(sanitizeHostId("..")).toBeUndefined();
		expect(sanitizeHostId("a/b")).toBeUndefined();
		expect(sanitizeHostId("a\\b")).toBeUndefined();
	});

	it("trims and keeps a usable id", () => {
		expect(sanitizeHostId("  workstation-1 ")).toBe("workstation-1");
	});
});

describe("resolveHostId", () => {
	it("prefers a configured id", () => {
		expect(resolveHostId({ configuredHostId: "workstation-1", hostname: "machine-a" })).toEqual({
			hostId: "workstation-1",
			source: "configured",
		});
	});

	it("warns and falls back to the machine name for an unusable configured id", () => {
		const warnings: string[] = [];
		const resolved = resolveHostId({
			configuredHostId: "../evil",
			hostname: "machine-a",
			warn: (message) => warnings.push(message),
		});

		expect(resolved).toEqual({ hostId: "machine-a", source: "hostname" });
		expect(warnings.join("\n")).toContain("host id");
	});

	it("resolves the machine name with no configuration", () => {
		expect(resolveHostId({ hostname: "machine-a" })).toEqual({ hostId: "machine-a", source: "hostname" });
	});

	it("does not touch the filesystem (PR!151 F2: no host-id file)", () => {
		// The deps carry no fs seam at all — resolution cannot persist anything.
		const deps = { configuredHostId: "workstation-1", hostname: "machine-a" } as const;
		expect(Object.keys(deps)).toEqual(["configuredHostId", "hostname"]);
	});
});
