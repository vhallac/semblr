import { describe, expect, it } from "vitest";
import {
	buildContextPreamble,
	buildFollowUpSectionContent,
	buildGroupedRecencyList,
	buildRelevanceList,
	buildSessionArchitecture,
	buildWorkingMemorySection,
	DEFAULT_PROMPT_TRUNCATION,
	formatFileSize,
	formatGroupedRoundEntry,
	formatRoundEntry,
	RECENCY_LIST_HEADER,
	splitCommandArgs,
	truncateUserPrompt,
} from "./context-format.ts";
import { addSlot, createMiniMemStore } from "./working-memory.ts";

const LONG_PROMPT = [
	"Fix the auth flow in lib/auth.ts.",
	"The spec below is the source of truth:",
	...Array.from({ length: 80 }, (_, i) => `spec line ${i} with some content to pad`),
	"Final instruction: keep the token budget small.",
].join("\n");

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

describe("truncateUserPrompt", () => {
	it("returns prompts at or below the head+tail budget unchanged", () => {
		const budget = DEFAULT_PROMPT_TRUNCATION.headChars + DEFAULT_PROMPT_TRUNCATION.tailChars;
		const shortPrompt = "a".repeat(budget);
		expect(truncateUserPrompt(shortPrompt)).toBe(shortPrompt);
	});

	it("keeps head and tail of long prompts and elides the middle", () => {
		const result = truncateUserPrompt(LONG_PROMPT);
		expect(result).toContain("Fix the auth flow in lib/auth.ts.");
		expect(result).toContain("Final instruction: keep the token budget small.");
		const match = result.match(/… \[(\d+) chars elided\] …/);
		expect(match).not.toBeNull();
		const kept = result.replace(`\n${match?.[0]}\n`, "");
		expect(LONG_PROMPT.length - Number(match?.[1])).toBe(kept.length);
	});

	it("clamps to complete lines when the head/tail windows allow it", () => {
		const lines = Array.from({ length: 40 }, (_, i) => "x".repeat(28) + String(i).padStart(2, "0"));
		const prompt = lines.join("\n"); // 40 lines × 30 chars + 39 newlines = 1239
		const result = truncateUserPrompt(prompt, { headChars: 260, tailChars: 140 });
		expect(result.split("\n")).toEqual([...lines.slice(0, 8), "… [869 chars elided] …", ...lines.slice(36)]);
	});

	it("truncates prompts one char over the budget without line snapping", () => {
		const budget = DEFAULT_PROMPT_TRUNCATION.headChars + DEFAULT_PROMPT_TRUNCATION.tailChars;
		const prompt = "a".repeat(budget + 1);
		const result = truncateUserPrompt(prompt);
		expect(result).toBe(
			`${"a".repeat(DEFAULT_PROMPT_TRUNCATION.headChars)}\n… [1 chars elided] …\n${"a".repeat(
				DEFAULT_PROMPT_TRUNCATION.tailChars,
			)}`,
		);
	});

	it("is disabled when head or tail chars are not positive", () => {
		expect(truncateUserPrompt(LONG_PROMPT, { headChars: 0, tailChars: 140 })).toBe(LONG_PROMPT);
		expect(truncateUserPrompt(LONG_PROMPT, { headChars: 260, tailChars: 0 })).toBe(LONG_PROMPT);
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

	it("truncates long prompts in relevance entries with an elision marker", () => {
		const entry = formatRoundEntry(1, "round.json", "0.90", "0 tools", LONG_PROMPT, "2KB");
		expect(entry).toContain("  user: Fix the auth flow in lib/auth.ts.");
		expect(entry).toContain("  Final instruction: keep the token budget small.");
		expect(entry.some((line) => /… \[\d+ chars elided\] …/.test(line))).toBe(true);
		expect(entry.some((line) => line.includes("spec line 40 with some content to pad"))).toBe(false);
	});

	it("honors custom truncation options in grouped entries", () => {
		const entry = formatGroupedRoundEntry(1, "round.json", "0 tools", LONG_PROMPT, undefined, {
			headChars: 30,
			tailChars: 20,
		});
		expect(entry.join("\n")).toContain("… [");
		expect(entry.join("\n")).toContain("chars elided] …");
		expect(entry.some((line) => line.includes("Final instruction"))).toBe(false);
		expect(entry[1]).toBe("  user: Fix the auth flow in lib/auth.");
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

	it("selects retained rounds by global chronology across interleaved topics, then renders grouped (issue #107 F2)", () => {
		// Causal order A1, B1, A2, B2 — topics A and B interleave. Under a
		// 2-entry cap the globally newest two rounds are B2 and A2; a group-major
		// walk would keep B2 and B1 instead, discarding A2 although it is newer
		// than B1. Rendering stays grouped by topic, groups ordered by their
		// newest retained round.
		const a1 = { fileName: "a1.json", userPrompt: "A1", responseSequence: "", toolSummary: "0 tools" };
		const b1 = { fileName: "b1.json", userPrompt: "B1", responseSequence: "", toolSummary: "0 tools" };
		const a2 = { fileName: "a2.json", userPrompt: "A2", responseSequence: "", toolSummary: "0 tools" };
		const b2 = { fileName: "b2.json", userPrompt: "B2", responseSequence: "", toolSummary: "0 tools" };
		const causalChain = [a1, b1, a2, b2];
		const groups = [{ rounds: [a1, a2] }, { rounds: [b1, b2] }];
		const list = buildGroupedRecencyList(groups, causalChain, () => null, DEFAULT_PROMPT_TRUNCATION, {
			maxEntries: 2,
			budgetTokens: 1_000_000,
			estimateTokensFn: (text) => text.length,
		});
		expect(list).toContain("**Group 1**\n\n- [index: 1] b2.json [n/a | 0 tools]:");
		expect(list).toContain("**Group 2**\n\n- [index: 2] a2.json [n/a | 0 tools]:");
		expect(list).not.toContain("a1.json");
		expect(list).not.toContain("b1.json");
	});

	it("renders rounds that belong to no topic group as singleton groups (failed agent_end embedding)", () => {
		// u1.json never reached assignToGroup (embedding unavailable at agent_end):
		// it stays in the causal chain but in no topic group. It must still lead
		// the recency list — group-based selection would have dropped it entirely.
		const groupedOld = { fileName: "g.json", userPrompt: "grouped", responseSequence: "", toolSummary: "0 tools" };
		const ungroupedNew = {
			fileName: "u1.json",
			userPrompt: "ungrouped",
			responseSequence: "",
			toolSummary: "0 tools",
		};
		const list = buildGroupedRecencyList([{ rounds: [groupedOld] }], [groupedOld, ungroupedNew]);
		expect(list).toContain("**Group 1**\n\n- [index: 1] u1.json [n/a | 0 tools]:");
		expect(list).toContain("**Group 2**\n\n- [index: 2] g.json [n/a | 0 tools]:");
	});

	it("places singleton groups at their global recency position with globally ordered indices", () => {
		// Chain: gOld (topic G), uMid (ungrouped), gNew (topic G). Rendering is
		// group-major, but indices follow the global recency walk — Group 1 holds
		// gNew (1) and gOld (3), while the singleton holds uMid (2).
		const gOld = { fileName: "g-old.json", userPrompt: "g old", responseSequence: "", toolSummary: "0 tools" };
		const uMid = { fileName: "u-mid.json", userPrompt: "u mid", responseSequence: "", toolSummary: "0 tools" };
		const gNew = { fileName: "g-new.json", userPrompt: "g new", responseSequence: "", toolSummary: "0 tools" };
		const list = buildGroupedRecencyList([{ rounds: [gOld, gNew] }], [gOld, uMid, gNew]);
		expect(list).toContain("**Group 1**\n\n- [index: 1] g-new.json");
		expect(list).toContain("- [index: 3] g-old.json");
		expect(list).toContain("**Group 2**\n\n- [index: 2] u-mid.json");
		// group-major rendering: Group 1's rounds both come before Group 2
		const text = list ?? "";
		expect(text.indexOf("[index: 1] g-new.json")).toBeLessThan(text.indexOf("**Group 2**"));
		expect(text.indexOf("[index: 3] g-old.json")).toBeLessThan(text.indexOf("**Group 2**"));
	});

	it("truncates long prompts in grouped recency entries", () => {
		const entry = {
			fileName: "long.json",
			userPrompt: LONG_PROMPT,
			responseSequence: "",
			toolSummary: "0 tools",
		};
		const list = buildGroupedRecencyList([{ rounds: [entry] }], [entry], () => null);
		expect(list).toContain("- [index: 1] long.json [n/a | 0 tools]:");
		expect(list).toContain("  user: Fix the auth flow in lib/auth.ts.");
		expect(list).toContain("chars elided] …");
		expect(list).not.toContain("spec line 40 with some content to pad");
	});

	describe("recency list bounds (issue #106)", () => {
		const entry = (name: string, prompt = "short") => ({
			fileName: name,
			userPrompt: prompt,
			responseSequence: "",
			toolSummary: "0 tools",
		});
		// Entry cost is the rendered injection (entry + group header), measured
		// with a char-counting estimator like the relevance-list tests.
		const lenCost = (text: string) => text.length;
		const bounds = { budgetTokens: 1_000_000, estimateTokensFn: lenCost } as const;

		it("hard-caps entries across groups, keeping the most recent rounds", () => {
			const rounds = Array.from({ length: 9 }, (_, i) => entry(`r${i}.json`, `round ${i}`));
			const groups = [
				{ rounds: [rounds[0]] },
				{ rounds: [rounds[1], rounds[2], rounds[3]] },
				{ rounds: [rounds[4], rounds[5], rounds[6], rounds[7], rounds[8]] },
			];
			const list = buildGroupedRecencyList(groups, rounds, () => null, DEFAULT_PROMPT_TRUNCATION, {
				...bounds,
				maxEntries: 4,
			});
			// The newest group renders first: r8..r5 kept, everything older dropped.
			expect(list).toContain("[index: 1] r8.json");
			expect(list).toContain("[index: 4] r5.json");
			expect(list).not.toContain("r4.json");
			expect(list).not.toContain("r0.json");
		});

		it("drops emptied groups without leaving a stray header", () => {
			const rounds = Array.from({ length: 6 }, (_, i) => entry(`r${i}.json`, `round ${i}`));
			const groups = [{ rounds: [rounds[0], rounds[1], rounds[2]] }, { rounds: [rounds[3], rounds[4], rounds[5]] }];
			const list = buildGroupedRecencyList(groups, rounds, () => null, DEFAULT_PROMPT_TRUNCATION, {
				...bounds,
				maxEntries: 4,
			});
			// Newest group keeps r5, r4, r3; the older group keeps only r2 — r1
			// and r0 are dropped, and no group header renders without entries.
			expect(list).toContain("**Group 1**");
			expect(list).toContain("**Group 2**");
			expect(list).toContain("[index: 4] r2.json");
			expect(list).not.toContain("r1.json");
			expect(list).not.toContain("r0.json");
		});

		it("defaults to a 20-entry cap when no options are passed", () => {
			const rounds = Array.from({ length: 30 }, (_, i) => entry(`r${i}.json`, "hi"));
			const list = buildGroupedRecencyList([{ rounds }], rounds);
			expect(list).toContain("[index: 20] r10.json");
			expect(list).not.toContain("r9.json");
		});

		it("stops at the token budget after the header charge", () => {
			const rounds = [entry("old.json", "x".repeat(2000)), entry("new.json", "x".repeat(2000))];
			// The header is charged up-front; a truncated entry costs ~490 chars,
			// so 100 chars of slack admits exactly one (the newest, kept
			// unconditionally as the always-on causal context).
			const list = buildGroupedRecencyList([{ rounds }], rounds, () => null, DEFAULT_PROMPT_TRUNCATION, {
				budgetTokens: RECENCY_LIST_HEADER.length + 100,
				estimateTokensFn: lenCost,
			});
			expect(list).toContain("[index: 1] new.json");
			expect(list).not.toContain("old.json");
		});

		it("always keeps the most recent round even when the budget cannot fit it", () => {
			const rounds = [entry("old.json", "hi"), entry("new.json", "hi")];
			const list = buildGroupedRecencyList([{ rounds }], rounds, () => null, DEFAULT_PROMPT_TRUNCATION, {
				budgetTokens: 10,
				estimateTokensFn: lenCost,
			});
			expect(list).toContain("[index: 1] new.json");
			expect(list).not.toContain("old.json");
		});

		it("charges group headers against the budget", () => {
			// a.json is the newest round (renders as index 1); b.json sits in an older
			// group. An entry costs ~52 chars; a new group adds ~18 chars of
			// separator + header, so 60 chars of slack fits one more entry but not a
			// new group — b.json and its header are both dropped.
			const groups = [{ rounds: [entry("b.json", "hi")] }, { rounds: [entry("a.json", "hi")] }];
			const rounds = [...groups[0].rounds, ...groups[1].rounds];
			const list = buildGroupedRecencyList(groups, rounds, () => null, DEFAULT_PROMPT_TRUNCATION, {
				budgetTokens: RECENCY_LIST_HEADER.length + 60,
				estimateTokensFn: lenCost,
			});
			expect(list).toContain("[index: 1] a.json");
			expect(list).not.toContain("b.json");
			expect(list).not.toContain("**Group 2**");
		});

		it("charges singleton-group headers for ungrouped rounds against the budget", () => {
			// a.json is the newest round (grouped); b.json is older and belongs to no
			// group. An entry costs ~53 chars; a new group adds ~18 chars of
			// separator + header, so 60 chars of slack fits a's entry + header but
			// not b's singleton — b and its header are both dropped.
			const a = entry("a.json", "hi");
			const b = entry("b.json", "hi");
			const list = buildGroupedRecencyList([{ rounds: [a] }], [b, a], () => null, DEFAULT_PROMPT_TRUNCATION, {
				...bounds,
				budgetTokens: RECENCY_LIST_HEADER.length + 60,
			});
			expect(list).toContain("[index: 1] a.json");
			expect(list).not.toContain("b.json");
			expect(list).not.toContain("**Group 2**");
		});

		it("treats a non-positive budget as unbounded", () => {
			const rounds = Array.from({ length: 25 }, (_, i) => entry(`r${i}.json`, "hi"));
			const list = buildGroupedRecencyList([{ rounds }], rounds, () => null, DEFAULT_PROMPT_TRUNCATION, {
				maxEntries: 30,
				budgetTokens: 0,
			});
			expect(list).toContain("[index: 25] r0.json");
			expect(list).toContain("[index: 1] r24.json");
		});

		it("returns null when the entry cap is zero", () => {
			const rounds = [entry("a.json", "hi")];
			expect(
				buildGroupedRecencyList([{ rounds }], rounds, () => null, DEFAULT_PROMPT_TRUNCATION, { maxEntries: 0 }),
			).toBeNull();
		});
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

		expect(list).toContain("--- RELEVANCE LIST (all sessions, by hybrid relevance) ---");
		expect(list).toContain("They combine semantic vector similarity with exact keyword matching.");
		expect(list).toContain(
			"Use this list when the prompt asks about past work, decisions, or findings\nfrom prior sessions, or requires cross-session continuity (same project,\nrecurring topic, long-running task).",
		);
		expect(list).toContain(
			"If nothing here matches but the query\nclearly needs past context, use search_interactions.",
		);
		expect(list).toContain("1. round.json [0.88 | 2 tools (read×1 (1KB), grep×1) | 10KB]:");
		expect(buildRelevanceList([])).toBeNull();
	});

	it("truncates long prompts in relevance list entries", () => {
		const list = buildRelevanceList(
			[
				{
					fileName: "round.json",
					bestScore: 0.9,
					data: { userPrompt: LONG_PROMPT },
				},
			],
			() => null,
		);
		expect(list).toContain("  user: Fix the auth flow in lib/auth.ts.");
		expect(list).toContain("chars elided] …");
		expect(list).not.toContain("spec line 40 with some content to pad");
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

	it("context preamble notes mid-section elision of long prompts", () => {
		const result = buildContextPreamble(true, false);
		expect(result).toContain("Very long prompts are elided mid-section");
		expect(result).not.toContain("the full user prompt");
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
