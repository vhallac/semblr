import { describe, expect, it } from "vitest";
import {
	type AgentEndPersistDeps,
	type AgentEndPersistFs,
	defaultAgentEndPersistFs,
	type PersistAgentEndResult,
	persistAgentEndRound,
} from "./agent-end-persist.ts";

import { createRoundFilePath } from "./hash.ts";
import type { ToolCallDetail } from "./round-data.ts";

function makeFs(overrides: Partial<AgentEndPersistFs> = {}): AgentEndPersistFs & { files: Map<string, string> } {
	const files = new Map<string, string>();
	return {
		files,
		mkdirSync() {},
		existsSync: (path) => files.has(path) || overrides.existsSync?.(path) === true,
		writeFileSync: (path, data) => {
			files.set(path, data);
			overrides.writeFileSync?.(path, data);
		},
		readFileSync: (path) => {
			const data = files.get(path);
			if (data === undefined) throw new Error("ENOENT");
			return data;
		},
		...overrides,
	};
}

const toolCall: ToolCallDetail = {
	index: 0,
	name: "bash",
	arguments: "ls",
	result_summary: "ok",
};

const baseInput = {
	cachedUserPrompt: "fix the bug",
	accumulatedText: ["did the thing"],
	messages: [{ role: "user", content: "fix the bug" }],
	turnIndex: 0,
	toolCallCount: 1,
	toolCallNames: ["bash"],
	toolCalls: [toolCall],
	responseSegments: [],
	parentId: null,
};

const FS_KEYS = ["mkdirSync", "existsSync", "writeFileSync", "readFileSync"] as const;

function makeDeps(
	overrides: Partial<AgentEndPersistDeps> & Partial<AgentEndPersistFs> = {},
): AgentEndPersistDeps & { fs: ReturnType<typeof makeFs> } {
	const fsOverrides = Object.fromEntries(
		Object.entries(overrides).filter(([k]) => (FS_KEYS as readonly string[]).includes(k)),
	);
	const fs = makeFs(fsOverrides);
	const { mkdirSync: _m, existsSync: _e, writeFileSync: _w, readFileSync: _r, ...rest } = overrides;
	return { ...rest, fs, roundsDir: "/tmp/rounds" };
}

function kindOf(result: PersistAgentEndResult): string {
	return result.kind;
}

describe("persistAgentEndRound (issue #130 handler-ordering)", () => {
	it("default fs surface is node:fs", () => {
		expect(typeof defaultAgentEndPersistFs.writeFileSync).toBe("function");
	});

	it("saved: writes the round file and returns its data", () => {
		const deps = makeDeps();
		const result = persistAgentEndRound(deps, baseInput);
		expect(result.kind).toBe("saved");
		if (result.kind !== "saved") return;
		expect(result.saved.userPrompt).toBe("fix the bug");
		expect(result.saved.fileName.endsWith(".json")).toBe(true);
		expect(JSON.parse(deps.fs.files.get(`/tmp/rounds/${result.saved.fileName}`)!).userPrompt).toBe("fix the bug");
	});

	it("no-prompt: empty prompt produces no file", () => {
		const deps = makeDeps();
		const result = persistAgentEndRound(deps, { ...baseInput, cachedUserPrompt: "", messages: [] });
		expect(kindOf(result)).toBe("no-prompt");
		expect(deps.fs.files.size).toBe(0);
	});

	it("emergency: assembly failure still writes a raw round from in-memory state", () => {
		const deps = makeDeps({
			buildRoundFile: () => {
				throw new Error("assembly blew up");
			},
		});
		const result = persistAgentEndRound(deps, baseInput);
		expect(kindOf(result)).toBe("emergency");
		if (result.kind !== "emergency") return;
		expect(result.message).toBe("assembly blew up");
		const [name, data] = [...deps.fs.files.entries()][0];
		expect(name.startsWith("/tmp/rounds/")).toBe(true);
		expect(JSON.parse(data).userPrompt).toBe("fix the bug");
	});

	it("emergency: filename parity with the normal path — toolCalls hashed, followup marker stripped (F2)", () => {
		const deps = makeDeps({
			buildRoundFile: () => {
				throw new Error("assembly blew up");
			},
		});
		const withMarker = {
			...baseInput,
			accumulatedText: ["did the thing\n\nround_needs_followup"],
		};
		const result = persistAgentEndRound(deps, withMarker);
		if (result.kind !== "emergency") return;
		// The emergency filename must equal what the normal path computes for the
		// same round: prompt + marker-stripped text + toolCalls feed the hash.
		expect(result.fileName).toBe(createRoundFilePath("fix the bug", "did the thing", [toolCall]));
		const [name, data] = [...deps.fs.files.entries()][0];
		const round = JSON.parse(data);
		expect(name.endsWith(result.fileName)).toBe(true);
		expect(round.responseSequence).not.toContain("round_needs_followup");
		expect(round.needsFollowup).toBe(true);
	});

	it("failed: emergency write failing loses nothing silently — reports failure", () => {
		const deps = makeDeps({
			buildRoundFile: () => {
				throw new Error("assembly blew up");
			},
			mkdirSync: () => {
				throw new Error("disk full");
			},
		});
		const result = persistAgentEndRound(deps, baseInput);
		expect(result.kind).toBe("failed");
		if (result.kind !== "failed") return;
		expect(result.message).toBe("disk full");
	});

	it("dedup: an existing content-hash file is not overwritten", () => {
		const deps = makeDeps({ existsSync: (path: string) => path.endsWith(".json") });
		const result = persistAgentEndRound(deps, baseInput);
		expect(result.kind).toBe("dedup");
		expect(deps.fs.files.size).toBe(0);
	});

	it("dedup: onDedup step runs with the existing round file, without postWrite", () => {
		const deps = makeDeps({ existsSync: (path: string) => path.endsWith(".json") });
		const seen: string[] = [];
		const result = persistAgentEndRound(
			deps,
			baseInput,
			() => seen.push("postWrite"),
			(saved) => seen.push(`onDedup:${saved.fileName}`),
		);
		expect(result.kind).toBe("dedup");
		if (result.kind !== "dedup") return;
		// onDedup receives the existing saved file; postWrite never runs on dedup
		expect(seen).toEqual([`onDedup:${result.saved.fileName}`]);
		expect(deps.fs.files.size).toBe(0);
	});

	it("dedup: onDedup throwing reports dedupError and stays dedup", () => {
		const deps = makeDeps({ existsSync: (path: string) => path.endsWith(".json") });
		const result = persistAgentEndRound(deps, baseInput, undefined, () => {
			throw new Error("chain push exploded");
		});
		expect(result.kind).toBe("dedup");
		if (result.kind !== "dedup") return;
		expect(result.dedupError).toBe("chain push exploded");
	});

	it("write-first: postWrite step throwing does not lose the round file", () => {
		const deps = makeDeps();
		const result = persistAgentEndRound(deps, baseInput, () => {
			throw new Error("bm25 exploded");
		});
		expect(result.kind).toBe("saved");
		if (result.kind !== "saved") return;
		expect(result.postWriteError).toBe("bm25 exploded");
		// The round file is on disk despite the downstream exception
		const onDisk = JSON.parse(deps.fs.files.get(`/tmp/rounds/${result.saved.fileName}`)!);
		expect(onDisk.userPrompt).toBe("fix the bug");
		expect(onDisk.parentId).toBeNull();
	});

	it("saved: round data carries parentId, toolCalls, and needsFollowup marker stripping", () => {
		const deps = makeDeps();
		const result = persistAgentEndRound(deps, {
			...baseInput,
			accumulatedText: ["done\n\nround_needs_followup"],
			parentId: "prev.json",
		});
		if (result.kind !== "saved") throw new Error(`expected saved, got ${result.kind}`);
		expect(result.saved.needsFollowup).toBe(true);
		expect(result.roundData.parentId).toBe("prev.json");
	});
});
