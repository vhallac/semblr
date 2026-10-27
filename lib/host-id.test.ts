import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { type HostIdFs, hostIdFilePath, resolveHostId, sanitizeHostId } from "./host-id.ts";

function fakeFs(files: Record<string, string> = {}) {
	const store = new Map<string, string>(Object.entries(files).map(([key, value]) => [path.normalize(key), value]));
	return {
		store,
		fsImpl: {
			existsSync: (filePath: string) => store.has(path.normalize(filePath)),
			readFileSync: (filePath: string) => {
				const content = store.get(path.normalize(filePath));
				if (content === undefined) throw new Error(`Missing file: ${filePath}`);
				return content;
			},
			writeFileSync: (filePath: string, data: string) => {
				store.set(path.normalize(filePath), data);
			},
			mkdirSync: () => {},
			renameSync: (from: string, to: string) => {
				const content = store.get(path.normalize(from));
				if (content === undefined) throw new Error(`Missing file: ${from}`);
				store.delete(path.normalize(from));
				store.set(path.normalize(to), content);
			},
		} satisfies HostIdFs,
	};
}

describe("hostIdFilePath", () => {
	it("keeps the host id file host-local (under the agent dir, not the rounds dir)", () => {
		expect(hostIdFilePath("/agent")).toBe(path.join("/agent", "semblr", "host-id"));
	});
});

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
	it("prefers a configured id and does not persist it", () => {
		const { fsImpl, store } = fakeFs();
		const resolved = resolveHostId({
			configuredHostId: "workstation-1",
			agentDir: "/agent",
			hostname: "machine-a",
			fsImpl,
		});

		expect(resolved).toEqual({ hostId: "workstation-1", source: "configured", persisted: false });
		expect(store.size).toBe(0);
	});

	it("warns and falls back to the machine name for an unusable configured id", () => {
		const { fsImpl } = fakeFs();
		const warnings: string[] = [];
		const resolved = resolveHostId({
			configuredHostId: "../evil",
			agentDir: "/agent",
			hostname: "machine-a",
			fsImpl,
			warn: (message) => warnings.push(message),
		});

		expect(resolved.hostId).toBe("machine-a");
		expect(warnings.join("\n")).toContain("host id");
	});

	it("uses the saved host id file over the machine name", () => {
		const { fsImpl, store } = fakeFs({ [hostIdFilePath("/agent")]: "saved-host\n" });
		const resolved = resolveHostId({ agentDir: "/agent", hostname: "machine-b", fsImpl });

		expect(resolved).toEqual({ hostId: "saved-host", source: "file", persisted: false });
		expect(store.get(path.normalize(hostIdFilePath("/agent")))).toBe("saved-host\n");
	});

	it("saves the machine name on first use and reuses it afterwards", () => {
		const { fsImpl, store } = fakeFs();
		const first = resolveHostId({ agentDir: "/agent", hostname: "machine-a", fsImpl });

		expect(first).toEqual({ hostId: "machine-a", source: "hostname", persisted: true });
		expect(store.get(path.normalize(hostIdFilePath("/agent")))).toBe("machine-a\n");

		// A later run on a host with a different machine name resolves the
		// saved id — the shard name does not change.
		const second = resolveHostId({ agentDir: "/agent", hostname: "machine-b", fsImpl });
		expect(second).toEqual({ hostId: "machine-a", source: "file", persisted: false });
	});

	it("reports a failed persist but still resolves the id", () => {
		const warnings: string[] = [];
		const fsImpl = {
			existsSync: () => false,
			readFileSync: () => "",
			writeFileSync: () => {
				throw new Error("disk full");
			},
			mkdirSync: () => {},
			renameSync: () => {},
		} satisfies HostIdFs;

		const resolved = resolveHostId({
			agentDir: "/agent",
			hostname: "machine-a",
			fsImpl,
			warn: (message) => warnings.push(message),
		});

		expect(resolved).toEqual({ hostId: "machine-a", source: "hostname", persisted: false });
		expect(warnings.join("\n")).toContain("host id");
	});

	it("resolves the id without persisting when the fs seam cannot write", () => {
		const warnings: string[] = [];
		const resolved = resolveHostId({
			agentDir: "/agent",
			hostname: "machine-a",
			fsImpl: {
				existsSync: () => false,
				readFileSync: () => "",
			},
			warn: (message) => warnings.push(message),
		});

		expect(resolved).toEqual({ hostId: "machine-a", source: "hostname", persisted: false });
		expect(warnings).toEqual([]);
	});
});
