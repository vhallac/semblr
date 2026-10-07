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
	// arguments is JSON.stringify'd by the session-parse derivation — keep the
	// live ToolCallDetail in the same shape so content-hash names agree.
	arguments: "{}",
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

	it("emergency: parent linkage is preserved — parentId persists instead of null (F7)", () => {
		const deps = makeDeps({
			buildRoundFile: () => {
				throw new Error("assembly blew up");
			},
		});
		const linked = { ...baseInput, parentId: "parent.json" };
		const result = persistAgentEndRound(deps, linked);
		if (result.kind !== "emergency") return;
		const round = JSON.parse([...deps.fs.files.values()][0]);
		expect(round.parentId).toBe("parent.json");
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

	it("F1: write failure sets the gate to the failed round", () => {
		const gate = { blockedBy: null as string | null };
		const deps = makeDeps({
			writeFileSync: () => {
				throw new Error("ENOSPC");
			},
			gate,
		});
		const result = persistAgentEndRound(deps, baseInput);
		expect(result.kind).toBe("failed");
		if (result.kind !== "failed") return;
		expect(gate.blockedBy).toBe(createRoundFilePath("fix the bug", "did the thing", [toolCall]));
	});

	it("F1: blocked gate persists no later round ahead of the gap", () => {
		const gate = { blockedBy: "gap.json" };
		const deps = makeDeps({ gate });
		const result = persistAgentEndRound(deps, { ...baseInput, cachedUserPrompt: "later round" });
		expect(result.kind).toBe("blocked");
		if (result.kind !== "blocked") return;
		expect(result.blockedBy).toBe("gap.json");
		expect(deps.fs.files.size).toBe(0); // nothing was written
	});

	it("F1: composed — failed write at K, blocked K+1, fresh process + backfill recovers K", async () => {
		const { backfillMissingRounds } = await import("./session-backfill.ts");
		// Round K's write fails (disk full) — gate records the gap.
		const gate = { blockedBy: null as string | null };
		const failingFs = makeDeps({
			writeFileSync: () => {
				throw new Error("ENOSPC");
			},
			gate,
		});
		const kResult = persistAgentEndRound(failingFs, baseInput);
		expect(kResult.kind).toBe("failed");
		const kFileName = createRoundFilePath("fix the bug", "did the thing", [toolCall]);
		expect(gate.blockedBy).toBe(kFileName);

		// Next round (K+1) is blocked — never persisted ahead of K.
		const liveFs = makeDeps({ gate });
		const k1Result = persistAgentEndRound(liveFs, {
			...baseInput,
			cachedUserPrompt: "second round",
			accumulatedText: ["second answer"],
		});
		expect(k1Result.kind).toBe("blocked");
		expect(liveFs.fs.files.size).toBe(0);

		// Fresh process (new gate): nothing was persisted for K or K+1 — the
		// gap-free invariant means the rounds dir has no round ahead of the gap.
		const nodeFs = await import("node:fs");
		const os = await import("node:os");
		const nodePath = await import("node:path");
		const tmp = nodeFs.mkdtempSync(nodePath.join(os.tmpdir(), "f1-gap-"));
		const roundsDir = nodePath.join(tmp, "rounds");
		const k1FileName = createRoundFilePath("second round", "second answer", []); // fixture round K+1 has no tool calls
		expect(nodeFs.existsSync(nodePath.join(roundsDir, kFileName))).toBe(false);
		expect(nodeFs.existsSync(nodePath.join(roundsDir, k1FileName))).toBe(false);

		// Startup backfill over the session JSONL recovers K and K+1.
		try {
			const sessionFile = nodePath.join(tmp, "session.jsonl");
			nodeFs.writeFileSync(
				sessionFile,
				`${[
					JSON.stringify({
						type: "message",
						message: { role: "user", content: [{ type: "text", text: "fix the bug" }] },
					}),
					// K's live write carried the tool call — the fixture must mirror it so
					// the content-hash filename derivation agrees (shared derivation, F1).
					JSON.stringify({
						type: "message",
						message: {
							role: "assistant",
							content: [{ type: "toolCall", name: "bash", arguments: {}, id: "t1" }],
						},
					}),
					JSON.stringify({
						type: "message",
						message: {
							role: "toolResult",
							toolName: "bash",
							toolCallId: "t1",
							content: [{ type: "text", text: "ok" }],
						},
					}),
					JSON.stringify({
						type: "message",
						message: {
							role: "assistant",
							content: [{ type: "text", text: "did the thing" }],
							stopReason: "stop",
						},
					}),
					JSON.stringify({
						type: "message",
						message: { role: "user", content: [{ type: "text", text: "second round" }] },
					}),
					JSON.stringify({
						type: "message",
						message: {
							role: "assistant",
							content: [{ type: "text", text: "second answer" }],
							stopReason: "stop",
						},
					}),
				].join("\n")}\n`,
			);
			const outcome = backfillMissingRounds(sessionFile, roundsDir, nodeFs, { liveWindowMs: 0 });
			expect(outcome.recoveredFiles).toEqual([kFileName, k1FileName]);
			// Both rounds now on disk via backfill.
			expect(JSON.parse(nodeFs.readFileSync(nodePath.join(roundsDir, kFileName), "utf-8")).userPrompt).toBe(
				"fix the bug",
			);
			expect(nodeFs.existsSync(nodePath.join(roundsDir, k1FileName))).toBe(true);
			expect(JSON.parse(nodeFs.readFileSync(nodePath.join(roundsDir, k1FileName), "utf-8")).userPrompt).toBe(
				"second round",
			);
		} finally {
			nodeFs.rmSync(tmp, { recursive: true, force: true });
		}
	});
});
