import { describe, expect, it } from "vitest";
import {
	buildContextMessagePrefix,
	countWordsInMessageContent,
	extractContextPrompt,
	prepareContextMessages,
	shouldDropRelevanceList,
	startsWithEnvironmentPreamble,
} from "./semblr.ts";

const env = "[ENVIRONMENT]\nHost: test\nCWD: /repo\nCurrent date/time: 20260101T000000Z";

describe("semblr context hook join points", () => {
	it("prepares current-round messages from the last user prompt and preserves the system message", () => {
		const system = { role: "system", content: "rules" };
		const prepared = prepareContextMessages(
			[
				system,
				{ role: "user", content: "old prompt" },
				{ role: "assistant", content: "old answer" },
				{ role: "user", content: "current prompt" },
				{ role: "assistant", content: "partial answer" },
			],
			env,
		);

		expect(prepared.systemMsg).toBe(system);
		expect(prepared.hasUserMessage).toBe(true);
		expect(prepared.rawPromptWordCount).toBe(2);
		expect(prepared.currentMessages).toHaveLength(2);
		expect((prepared.currentMessages[0] as { content: string }).content).toBe(`${env}\n\ncurrent prompt`);
		expect(prepared.userPrompt).toBe(`${env}\n\ncurrent prompt`);
	});

	it("does not prepend duplicate environment blocks to replayed string prompts", () => {
		const content = `  ${env}\n\nalready augmented`;
		const prepared = prepareContextMessages([{ role: "user", content }], env);

		expect((prepared.augmentedMessages[0] as { content: string }).content).toBe(content);
		expect(prepared.userPrompt).toBe(content);
		expect(startsWithEnvironmentPreamble(content)).toBe(true);
	});

	it("prepends environment blocks to text-array prompts and extracts text-only prompt content", () => {
		const prepared = prepareContextMessages(
			[
				{
					role: "user",
					content: [
						{ type: "text", text: "look here" },
						{ type: "image", image_url: "ignored" },
						{ type: "text", text: "and there" },
					],
				},
			],
			env,
		);

		const content = (prepared.augmentedMessages[0] as { content: Array<{ text?: string }> }).content;
		expect(content[0].text).toBe(`${env}\n\nlook here`);
		expect(prepared.rawPromptWordCount).toBe(4);
		expect(prepared.userPrompt).toBe(`${env}\n\nlook here and there`);
	});

	it("adds an environment text block before non-text array prompts", () => {
		const prepared = prepareContextMessages(
			[
				{
					role: "user",
					content: [
						{ type: "image", image_url: "ignored" },
						{ type: "text", text: "describe this" },
					],
				},
			],
			env,
		);

		const content = (prepared.augmentedMessages[0] as { content: Array<{ type: string; text?: string }> }).content;
		expect(content[0]).toEqual({ type: "text", text: `${env}\n\n` });
		expect(prepared.userPrompt).toBe(`${env}\n\n describe this`);
	});

	it("distinguishes no-user messages from unsupported user content", () => {
		const noUser = prepareContextMessages([{ role: "assistant", content: "hello" }], env);
		expect(noUser.hasUserMessage).toBe(false);
		expect(noUser.userPrompt).toBeNull();

		const unsupported = prepareContextMessages([{ role: "user", content: 42 }], env);
		expect(unsupported.hasUserMessage).toBe(true);
		expect(unsupported.userPrompt).toBeNull();
		expect(unsupported.rawPromptWordCount).toBe(0);
	});

	it("counts words and clips string prompts to 200 space-separated tokens", () => {
		expect(countWordsInMessageContent(" one\n two   three ")).toBe(3);
		expect(
			countWordsInMessageContent([
				{ type: "text", text: "one two" },
				{ type: "text", text: "three" },
			]),
		).toBe(3);
		expect(countWordsInMessageContent([{ type: "image" }])).toBe(0);

		const longPrompt = Array.from({ length: 205 }, (_, i) => `w${i}`).join(" ");
		expect(extractContextPrompt(longPrompt)?.split(" ")).toHaveLength(200);
		expect(extractContextPrompt([{ type: "text", text: "array prompt" }])).toBe("array prompt");
		expect(extractContextPrompt({ nope: true })).toBeNull();
	});

	it("applies relevance-list suppression gates", () => {
		expect(shouldDropRelevanceList(100, { DROP_RELEVANCE_LIST: "1" })).toBe(true);
		expect(shouldDropRelevanceList(100, { DROP_RELEVANCE_LIST: "true" })).toBe(true);
		expect(shouldDropRelevanceList(19, { RELEVANCE_LIST_MIN_WORDS: "20" })).toBe(true);
		expect(shouldDropRelevanceList(20, { RELEVANCE_LIST_MIN_WORDS: "20" })).toBe(false);
		expect(
			shouldDropRelevanceList(20, { DROP_RELEVANCE_LIST: "false", RELEVANCE_LIST_MIN_WORDS: "not-a-number" }),
		).toBe(false);
	});

	it("builds the stable context message prefix in display order", () => {
		const system = { role: "developer", content: "dev rules" };
		const prefix = buildContextMessagePrefix(system, "preamble", "recency", "relevance");

		expect(prefix).toHaveLength(4);
		expect(prefix[0]).toBe(system);
		expect((prefix[1] as { content: Array<{ text: string }> }).content[0].text).toBe("preamble");
		expect((prefix[2] as { content: Array<{ text: string }> }).content[0].text).toBe("recency");
		expect((prefix[3] as { content: Array<{ text: string }> }).content[0].text).toBe("relevance");
		expect(buildContextMessagePrefix(null, null, null, null)).toEqual([]);
	});
});
