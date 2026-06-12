import { describe, expect, it } from "vitest";
import {
	buildContextPreamble,
	buildFollowUpSectionContent,
	buildGroupedRecencyList,
	buildRelevanceList,
	buildSessionArchitecture,
	buildWorkingMemorySection,
	formatFileSize,
	formatGroupedRoundEntry,
	formatRoundEntry,
	splitCommandArgs,
} from "./context-format.ts";
import { addSlot, createMiniMemStore } from "./working-memory.ts";

describe("follow-up section", () => {
	it("builds the follow-up section with round content", () => {
		const result = buildFollowUpSectionContent("abc.json", "What is the capital of France?", "The capital is Paris.");
		expect(result).toContain("--- PREVIOUS ROUND FOLLOW-UP ---");
		expect(result).toContain("abc.json");
		expect(result).toContain("What is the capital of France?");
		expect(result).toContain("The capital is Paris.");
	});

	it("includes both user prompt and assistant response in the section", () => {
		const result = buildFollowUpSectionContent("x.json", "multi\nline\nprompt", "multi\nline\nresponse");
		expect(result).toContain("USER PROMPT:");
		expect(result).toContain("ASSISTANT RESPONSE:");
		expect(result).toContain("multi\nline\nprompt");
		expect(result).toContain("multi\nline\nresponse");
	});
});

describe("session architecture", () => {
	it("builds session architecture section with heading", () => {
		const result = buildSessionArchitecture();
		expect(result).toContain("[SESSION ARCHITECTURE]");
	});

	it("describes round boundary amnesia", () => {
		const result = buildSessionArchitecture();
		expect(result).toContain("Each conversation round starts fresh by default");
		expect(result).toContain("NOT automatic — it exists only through semblr's explicit survival mechanisms");
	});

	it("lists follow-up injection as a survival mechanism (stage 1)", () => {
		const result = buildSessionArchitecture();
		expect(result).toContain("Follow-up injection");
		expect(result).toContain("round_needs_followup");
	});

	it("lists checkpoint as a survival mechanism (stage 1)", () => {
		const result = buildSessionArchitecture();
		expect(result).toContain("Checkpoint");
		expect(result).toContain("semblr_checkpoint");
	});

	it("lists working memory as a survival mechanism (stage 2)", () => {
		const result = buildSessionArchitecture();
		expect(result).toContain("Working Memory");
		expect(result).toContain("mini_mem__add");
		expect(result).toContain("mini_mem__get");
		expect(result).toContain("mini_mem__update");
		expect(result).toContain("mini_mem__delete");
		expect(result).toContain("mini_mem__get_and_delete");
		expect(result).toContain("short-term notes that survive round boundaries");
	});
});

describe("working memory section", () => {
	it("returns null for empty store", () => {
		const store = createMiniMemStore();
		expect(buildWorkingMemorySection(store)).toBeNull();
	});

	it("outputs correct single-entry list", () => {
		const store = createMiniMemStore();
		addSlot(store, "Fix the authentication flow", "Detailed plan here");
		const result = buildWorkingMemorySection(store);
		expect(result).toContain("[WORKING MEMORY]");
		expect(result).toContain("Use mini_mem__xxx tools");
		expect(result).toContain("- [id: 1] Fix the authentication flow");
	});

	it("outputs ordered list with correct ids and summaries for multiple slots", () => {
		const store = createMiniMemStore();
		addSlot(store, "First plan", "Content 1");
		addSlot(store, "Second note", "Content 2");
		addSlot(store, "Third decision", "Content 3");
		const result = buildWorkingMemorySection(store);
		expect(result).toContain("- [id: 1] First plan");
		expect(result).toContain("- [id: 2] Second note");
		expect(result).toContain("- [id: 3] Third decision");
		// Verify order — id 1 before id 2 before id 3
		const idx1 = result?.indexOf("[id: 1]") ?? -1;
		const idx2 = result?.indexOf("[id: 2]") ?? -1;
		const idx3 = result?.indexOf("[id: 3]") ?? -1;
		expect(idx1).toBeLessThan(idx2);
		expect(idx2).toBeLessThan(idx3);
	});

	it("does not expose full slot content in the section", () => {
		const store = createMiniMemStore();
		addSlot(store, "Secret plan", "TOP SECRET DETAILS HERE");
		const result = buildWorkingMemorySection(store);
		expect(result).not.toContain("TOP SECRET DETAILS HERE");
	});
});

describe("context formatting", () => {
	it("formats relevance and grouped entries with multiline prompts and optional size", () => {
		expect(formatRoundEntry(2, "abc.json", "0.91", "1 tools", "first\nsecond", "2KB")).toEqual([
			"2. abc.json [0.91 | 1 tools | 2KB]:",
			"  user: first",
			"  second",
			"  ---",
		]);
		expect(formatGroupedRoundEntry(1, "abc.json", "0 tools", "prompt")).toEqual([
			"- [index: 1] abc.json [n/a | 0 tools]:",
			"  user: prompt",
			"  ---",
		]);
	});

	it("builds grouped recency lists newest topic first with stable global indices", () => {
		const older = { fileName: "older.json", userPrompt: "older", responseSequence: "", toolSummary: "0 tools" };
		const newerA = { fileName: "new-a.json", userPrompt: "newer A", responseSequence: "", toolSummary: "1 tools" };
		const newerB = { fileName: "new-b.json", userPrompt: "newer B", responseSequence: "", toolSummary: "2 tools" };
		const causalChain = [older, newerA, newerB];
		const list = buildGroupedRecencyList([{ rounds: [older] }, { rounds: [newerA, newerB] }], causalChain, (file) =>
			file === "new-b.json" ? "4KB" : null,
		);

		expect(list).toContain("--- RECENCY LIST (current session, by topic) ---");
		expect(list).toContain("asks about past work, decisions, code, or findings from earlier in this\n  session");
		expect(list).not.toContain("Higher score = stronger match.");
		expect(list).toContain("Scan the group topics and prompts for relevance to the current prompt.");
		expect(list).toContain("Prefer the most recent entry (lowest index) in the most related group.");
		expect(list).not.toContain("cross-session");
		expect(list).toContain("**Group 1**\n\n- [index: 1] new-b.json [n/a | 2 tools | 4KB]:");
		expect(list).toContain("- [index: 2] new-a.json [n/a | 1 tools]:");
		expect(list).toContain("**Group 2**\n\n- [index: 3] older.json [n/a | 0 tools]:");
		expect(buildGroupedRecencyList([], [])).toBeNull();
	});

	it("builds relevance lists with score, size, and per-tool result sizes", () => {
		const list = buildRelevanceList(
			[
				{
					fileName: "round.json",
					bestScore: 0.876,
					data: {
						userPrompt: "Prompt",
						toolCallCount: 2,
						toolCalls: [
							{ name: "read", result_full: "x".repeat(1024) },
							{ name: "grep", result_summary: "" },
						],
					},
				},
			],
			() => "10KB",
		);

		expect(list).toContain("--- RELEVANCE LIST (all sessions, by similarity) ---");
		expect(list).toContain(
			"Use this list when the prompt asks about past work, decisions, or findings\nfrom prior sessions, or requires cross-session continuity (same project,\nrecurring topic, long-running task).",
		);
		expect(list).toContain(
			"If nothing here matches but the query\nclearly needs past context, use search_interactions.",
		);
		expect(list).toContain("1. round.json [0.88 | 2 tools (read×1 (1KB), grep×1) | 10KB]:");
		expect(buildRelevanceList([])).toBeNull();
	});

	it("builds context preamble only when at least one list exists", () => {
		expect(buildContextPreamble(false, false)).toBeNull();
		expect(buildContextPreamble(true, false)).toContain("[CONTEXT BUILDING REFERENCES]");
		expect(buildContextPreamble(false, true)).toContain("get_round_details");
	});

	it("context preamble includes survival mechanism hint", () => {
		const result = buildContextPreamble(true, false);
		expect(result).toContain("These tools fill in what the context summaries leave out");
		expect(result).toContain("See the SESSION ARCHITECTURE section for details.");
	});

	it("formats byte counts", () => {
		expect(formatFileSize(512)).toBe("0.5KB");
		expect(formatFileSize(5 * 1024)).toBe("5KB");
		expect(formatFileSize(54 * 1024)).toBe("54KB");
		expect(formatFileSize(1_153_434)).toBe("1.1MB");
	});

	it("splits command arguments with quotes and escapes", () => {
		expect(splitCommandArgs("--dry-run --limit 10")).toEqual(["--dry-run", "--limit", "10"]);
		expect(splitCommandArgs("--name 'two words' \"more words\"")).toEqual(["--name", "two words", "more words"]);
		expect(splitCommandArgs("escaped\\ space trailing\\")).toEqual(["escaped space", "trailing\\"]);
		expect(splitCommandArgs("   ")).toEqual([]);
	});
});
