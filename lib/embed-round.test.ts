import { describe, expect, it } from "vitest";
import {
	classifyRoundCoverage,
	type EmbedRoundDeps,
	embedRound,
	forcingReproducibleSuffixes,
	reproducedIndexSuffixes,
} from "./embed-round.ts";
import { indexRowSuffix } from "./index-io.ts";
import { buildPromptEmbeddingInput } from "./round-capture.ts";

describe("reproducedIndexSuffixes", () => {
	it("reports :prompt only when the prompt survives the short-prompt drop", () => {
		const prev = process.env.RELEVANCE_LIST_MIN_WORDS;
		process.env.RELEVANCE_LIST_MIN_WORDS = "20";
		try {
			const short = reproducedIndexSuffixes("only three words", false);
			expect([...short].sort()).toEqual([":response"]);
			const long = reproducedIndexSuffixes(
				"a prompt with well over twenty ordinary words so the shared policy will certainly not drop it from the index",
				false,
			);
			expect([...long].sort()).toEqual([":prompt", ":response"]);
		} finally {
			if (prev === undefined) delete process.env.RELEVANCE_LIST_MIN_WORDS;
			else process.env.RELEVANCE_LIST_MIN_WORDS = prev;
		}
	});

	it("adds :summary only when a summary is present", () => {
		const withSummary = reproducedIndexSuffixes("short one", true);
		expect(withSummary.has(":summary")).toBe(true);
		expect(withSummary.has(":response")).toBe(true);
		expect(withSummary.has(":prompt")).toBe(false);
	});
});

describe("forcingReproducibleSuffixes", () => {
	it("restricts a long prompt + summary to :prompt/:response, never :summary", () => {
		const prev = process.env.RELEVANCE_LIST_MIN_WORDS;
		process.env.RELEVANCE_LIST_MIN_WORDS = "20";
		try {
			const long = forcingReproducibleSuffixes(
				"a prompt with well over twenty ordinary words so the shared policy will certainly not drop it from the index",
				true,
			);
			expect([...long].sort()).toEqual([":prompt", ":response"]);
		} finally {
			if (prev === undefined) delete process.env.RELEVANCE_LIST_MIN_WORDS;
			else process.env.RELEVANCE_LIST_MIN_WORDS = prev;
		}
	});

	it("drops :prompt for a short prompt", () => {
		expect([...forcingReproducibleSuffixes("only three words", true)].sort()).toEqual([":response"]);
	});
});

describe("classifyRoundCoverage", () => {
	const forcing = new Set([":prompt", ":response"]);
	const classify = (entries: Array<{ filePath: string; model?: string }>, current: string) =>
		classifyRoundCoverage(entries, current, forcing, indexRowSuffix);

	it("is not covered when no forcing row exists (summary/round only)", () => {
		expect(classify([{ filePath: "r.json:summary", model: "m" }], "m").covered).toBe(false);
		expect(classify([{ filePath: "r.json:round", model: "m" }], "m").covered).toBe(false);
	});

	it("is covered when a forcing row is current, ignoring a foreign non-forcing row", () => {
		const c = classify(
			[
				{ filePath: "r.json:prompt", model: "m" },
				{ filePath: "r.json:summary", model: "old" },
			],
			"m",
		);
		expect(c.covered).toBe(true);
		expect(c.hasStaleReproducibleRow).toBe(false);
	});

	it("is not covered when a forcing row is stale", () => {
		const c = classify(
			[
				{ filePath: "r.json:prompt", model: "old" },
				{ filePath: "r.json:response", model: "m" },
			],
			"m",
		);
		expect(c.covered).toBe(false);
		expect(c.hasStaleReproducibleRow).toBe(true);
	});

	it("treats a legacy model-less row as current (issue #62)", () => {
		expect(classify([{ filePath: "r.json:response" }], "m").covered).toBe(true);
	});
});

function makeDeps(overrides: Partial<EmbedRoundDeps> = {}) {
	const embeddedTexts: string[] = [];
	const rows: Array<{ label: string; vec: number[]; hash?: string }> = [];
	const stored = new Map<string, number[]>();
	const deps: EmbedRoundDeps = {
		embed: (text) => {
			embeddedTexts.push(text);
			return Promise.resolve([text.length, 7]);
		},
		appendIndexRow: (label, vec, hash) => rows.push({ label, vec, hash }),
		writeRoundEmbedding: (fileName, vec) => stored.set(fileName, vec),
		...overrides,
	};
	return { deps, embeddedTexts, rows, stored };
}

describe("embedRound (shared F4+F6 policy core)", () => {
	it("full path: cleans and stamps the prompt, clips the response, stores the raw combined vector", async () => {
		const { deps, embeddedTexts, rows, stored } = makeDeps();
		const prompt = "explain the siege\n```python\n" + "y = 2\n".repeat(120) + "```";
		const result = await embedRound(
			{
				fileName: "r.json",
				userPrompt: prompt,
				responseText: "the answer",
				maxResponseBytes: 40,
			},
			deps,
		);
		expect(result.promptDropped).toBe(false);
		// prompt, response, combined — one embedding each
		expect(embeddedTexts).toHaveLength(3);
		const { text, hash } = buildPromptEmbeddingInput(prompt);
		expect(embeddedTexts[0]).toBe(text);
		expect(embeddedTexts[1].length).toBeLessThanOrEqual(40);
		expect(embeddedTexts[2]).toBe(`${text}\n\n${embeddedTexts[1]}`);
		expect(rows.map((r) => r.label)).toEqual(["r.json:prompt", "r.json:response"]);
		expect(rows[0].hash).toBe(hash);
		// live scale convention: rows normalized, stored vector raw
		const rowMag = Math.sqrt(rows[1].vec.reduce((s, x) => s + x * x, 0));
		expect(rowMag).toBeCloseTo(1, 5);
		expect(stored.get("r.json")).toEqual([text.length + 2 + embeddedTexts[1].length, 7]);
		expect(result.promptEmbedding).toEqual(stored.get("r.json"));
	});

	it("short-prompt drop: embeds only the response, skips :prompt, stores the normalized response vector", async () => {
		const { deps, embeddedTexts, rows, stored } = makeDeps();
		const result = await embedRound(
			{
				fileName: "s.json",
				userPrompt: "yes, do it",
				responseText: "then it is done",
			},
			deps,
		);
		expect(result.promptDropped).toBe(true);
		expect(embeddedTexts).toEqual(["then it is done"]);
		expect(rows.map((r) => r.label)).toEqual(["s.json:response"]);
		const mag = Math.sqrt(stored.get("s.json")!.reduce((s, x) => s + x * x, 0));
		expect(mag).toBeCloseTo(1, 5);
		expect(result.promptEmbedding).toEqual(stored.get("s.json"));
	});

	it("checkpoint summary: embeds a :summary row when summary text is provided", async () => {
		const { deps, embeddedTexts, rows } = makeDeps();
		await embedRound(
			{
				fileName: "t.json",
				userPrompt: "yes",
				responseText: "done",
				checkpointSummaryText: "Current Task: unit tests",
			},
			deps,
		);
		expect(rows.map((r) => r.label)).toEqual(["t.json:response", "t.json:summary"]);
		expect(embeddedTexts[1]).toBe("Current Task: unit tests");
	});

	it("hasIndexRow guard skips already-present labels without skipping the rest of the round", async () => {
		const appended: string[] = [];
		const { deps } = makeDeps({
			hasIndexRow: (label) => label === "u.json:prompt",
			appendIndexRow: (label) => appended.push(label),
		});
		const prompt =
			"a prompt with well over twenty ordinary words so the shared policy will not drop it from the index at all";
		const result = await embedRound({ fileName: "u.json", userPrompt: prompt, responseText: "the reply" }, deps);
		expect(result.promptDropped).toBe(false);
		expect(appended).toEqual(["u.json:response"]);
	});

	it("embedPrompt hook allows the live call site to reuse a hash-gated stashed prompt vector", async () => {
		const promptEmbeds: string[] = [];
		const stashHash = buildPromptEmbeddingInput(
			"continue the march to the sea and hold the stone bridge until the whole army has crossed over at dawn",
		).hash;
		const { deps, embeddedTexts, rows } = makeDeps({
			embedPrompt: (text, hash) => {
				promptEmbeds.push(text);
				if (hash === stashHash) return Promise.resolve([1, 2, 3]);
				return Promise.resolve([text.length, 7]);
			},
		});
		const result = await embedRound(
			{
				fileName: "v.json",
				userPrompt:
					"continue the march to the sea and hold the stone bridge until the whole army has crossed over at dawn",
				responseText: "the bridge holds",
			},
			deps,
		);
		expect(result.promptDropped).toBe(false);
		expect(promptEmbeds).toHaveLength(1);
		const promptRow = rows.find((r) => r.label === "v.json:prompt")!;
		// the core normalizes the prompt vector before appending, as agent_end does
		const mag = Math.sqrt(promptRow.vec.reduce((sum, x) => sum + x * x, 0));
		expect(mag).toBeCloseTo(1, 5);
		expect(promptRow.vec).toEqual([1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14)]);
		// the response and combined inputs still go through the plain embed dep
		expect(embeddedTexts).toHaveLength(2);
	});
});
