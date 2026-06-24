import { describe, expect, it } from "vitest";
import {
	contextCacheSnapshot,
	contextCacheStore,
	contextCacheValid,
	createContextCache,
	createRound,
	createSession,
} from "./state.ts";

describe("createSession", () => {
	it("initialises all fields with sensible defaults", () => {
		const s = createSession();

		expect(s.miniMemStore).toBeDefined();
		expect(Array.isArray(s.causalChain)).toBe(true);
		expect(s.causalChain.length).toBe(0);
		expect(Array.isArray(s.roundGroups)).toBe(true);
		expect(s.roundGroups.length).toBe(0);
		expect(s.lastFollowupGroupIdx).toBeNull();
		expect(s.injectedFollowupRounds).toBeInstanceOf(Set);
		expect(s.injectedFollowupRounds.size).toBe(0);
		expect(s.injectedCheckpointRounds).toBeInstanceOf(Set);
		expect(s.injectedCheckpointRounds.size).toBe(0);
	});

	it("creates independent sessions (no shared references)", () => {
		const s1 = createSession();
		const s2 = createSession();

		// Mutate s1 and verify s2 is unaffected
		s1.causalChain.push({ fileName: "abc.json", userPrompt: "test", responseSequence: "resp", toolSummary: "" });
		s1.roundGroups.push({ centroid: [0.5], rounds: [] });
		s1.injectedFollowupRounds.add("x.json");
		s1.injectedCheckpointRounds.add("y.json");

		expect(s2.causalChain.length).toBe(0);
		expect(s2.roundGroups.length).toBe(0);
		expect(s2.injectedFollowupRounds.size).toBe(0);
		expect(s2.injectedCheckpointRounds.size).toBe(0);
	});
});

describe("ContextCache", () => {
	it("createContextCache returns nulls", () => {
		const cc = createContextCache();
		expect(cc.envPreamble).toBeNull();
		expect(cc.messages).toBeNull();
		expect(cc.userPrompt).toBeNull();
	});

	it("contextCacheValid returns false when any field is null", () => {
		const cc = createContextCache();
		expect(contextCacheValid(cc, "prompt")).toBe(false);

		cc.envPreamble = "env";
		expect(contextCacheValid(cc, "prompt")).toBe(false);

		cc.messages = [];
		expect(contextCacheValid(cc, "prompt")).toBe(false);

		cc.userPrompt = "wrong";
		expect(contextCacheValid(cc, "prompt")).toBe(false);

		cc.userPrompt = "prompt";
		expect(contextCacheValid(cc, "prompt")).toBe(true);
	});

	it("contextCacheStore sets all three at once", () => {
		const cc = createContextCache();
		contextCacheStore(cc, "env", ["msg"], "prompt");
		expect(cc.envPreamble).toBe("env");
		expect(cc.messages).toEqual(["msg"]);
		expect(cc.userPrompt).toBe("prompt");
		expect(contextCacheValid(cc, "prompt")).toBe(true);
	});

	it("contextCacheSnapshot updates only messages", () => {
		const cc = createContextCache();
		cc.envPreamble = "env";
		cc.userPrompt = "prompt";
		contextCacheSnapshot(cc, ["updated"]);
		expect(cc.envPreamble).toBe("env");
		expect(cc.messages).toEqual(["updated"]);
		expect(cc.userPrompt).toBe("prompt");
	});
});

describe("createRound", () => {
	it("initialises all fields with sensible defaults", () => {
		const r = createRound();

		expect(r.userPrompt).toBeNull();
		expect(r.turnIndex).toBeNull();
		expect(Array.isArray(r.accumulatedText)).toBe(true);
		expect(r.accumulatedText.length).toBe(0);
		expect(r.toolCallCount).toBe(0);
		expect(Array.isArray(r.toolCallNames)).toBe(true);
		expect(r.toolCallNames.length).toBe(0);
		expect(Array.isArray(r.toolCalls)).toBe(true);
		expect(r.toolCalls.length).toBe(0);
		expect(Array.isArray(r.responseSegments)).toBe(true);
		expect(r.responseSegments.length).toBe(0);

		// embedding cache
		expect(r.lastContextUserPrompt).toBeNull();
		expect(Array.isArray(r.lastContextVec)).toBe(true);
		expect(r.lastContextVec.length).toBe(0);
		expect(r.promptVec).toBeNull();
		expect(r.skipPromptEmbedding).toBe(false);
		expect(r.presentedRecorded).toBe(false);

		// full-message cache
		expect(r.contextCache.envPreamble).toBeNull();
		expect(r.contextCache.messages).toBeNull();
		expect(r.contextCache.userPrompt).toBeNull();

		// checkpoint
		expect(r.lastCheckpointSummary).toBeNull();
		expect(r.contextWarningIssued).toBe(0);

		// multi-model routing
		expect(r.switchCounter).toBe(0);
		expect(r.currentPhase).toBeNull();
		expect(r.pendingModelSwitch).toBeNull();
	});

	it("creates independent rounds (no shared array references)", () => {
		const r1 = createRound();
		const r2 = createRound();

		// Mutate r1 arrays and verify r2 is unaffected
		r1.accumulatedText.push("some text");
		r1.toolCallNames.push("read");
		r1.lastContextVec.push(0.42);

		expect(r2.accumulatedText.length).toBe(0);
		expect(r2.toolCallNames.length).toBe(0);
		expect(r2.lastContextVec.length).toBe(0);
	});

	it("independent rounds have independent boolean fields", () => {
		const r1 = createRound();
		const r2 = createRound();

		r1.skipPromptEmbedding = true;
		r1.presentedRecorded = true;
		r1.contextWarningIssued = 3;

		expect(r2.skipPromptEmbedding).toBe(false);
		expect(r2.presentedRecorded).toBe(false);
		expect(r2.contextWarningIssued).toBe(0);
	});

	it("independent rounds have independent object references", () => {
		const r1 = createRound();
		const r2 = createRound();

		r1.toolCalls.push({ index: 0, name: "read", arguments: "{}", result_summary: "ok" });
		r1.responseSegments.push({ type: "text", text: "hello" });

		expect(r2.toolCalls.length).toBe(0);
		expect(r2.responseSegments.length).toBe(0);
	});

	it("independent rounds have independent context caches", () => {
		const r1 = createRound();
		const r2 = createRound();

		contextCacheStore(r1.contextCache, "env", ["msg"], "p");
		expect(r1.contextCache.messages).toEqual(["msg"]);
		expect(r2.contextCache.messages).toBeNull();
	});

	it("independent rounds have independent routing fields", () => {
		const r1 = createRound();
		const r2 = createRound();

		r1.switchCounter = 2;
		r1.currentPhase = "executing";
		r1.pendingModelSwitch = "glm-5.2:cloud";

		expect(r2.switchCounter).toBe(0);
		expect(r2.currentPhase).toBeNull();
		expect(r2.pendingModelSwitch).toBeNull();
	});

	it("routing fields start with sensible defaults", () => {
		const r = createRound();

		expect(r.switchCounter).toBe(0);
		expect(r.currentPhase).toBeNull();
		expect(r.pendingModelSwitch).toBeNull();
	});
});
