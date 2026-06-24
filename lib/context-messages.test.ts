import { describe, expect, it } from "vitest";
import {
	assembleContextPrefix,
	type ContextBlocks,
	countWordsInMessageContent,
	extractContextPrompt,
	prepareContextMessages,
	shouldDropEmbedding,
	shouldDropRelevanceList,
	startsWithEnvironmentPreamble,
} from "./context-messages.ts";

describe("startsWithEnvironmentPreamble", () => {
	it("returns true when content starts with [ENVIRONMENT]", () => {
		expect(startsWithEnvironmentPreamble("[ENVIRONMENT]\nfoo")).toBe(true);
	});

	it("returns true with leading whitespace", () => {
		expect(startsWithEnvironmentPreamble("  [ENVIRONMENT]\nfoo")).toBe(true);
	});

	it("returns false when content does not start with [ENVIRONMENT]", () => {
		expect(startsWithEnvironmentPreamble("Hello world")).toBe(false);
	});

	it("returns false for empty string", () => {
		expect(startsWithEnvironmentPreamble("")).toBe(false);
	});
});

describe("countWordsInMessageContent", () => {
	it("counts words in a string", () => {
		expect(countWordsInMessageContent("hello world foo")).toBe(3);
	});

	it("counts words from extracted text of content blocks", () => {
		const blocks = [
			{ type: "text", text: "hello world" },
			{ type: "text", text: "foo bar" },
		];
		expect(countWordsInMessageContent(blocks)).toBe(4);
	});

	it("returns 0 for non-string non-array content", () => {
		expect(countWordsInMessageContent(42)).toBe(0);
		expect(countWordsInMessageContent(null)).toBe(0);
	});

	it("returns 0 for empty string", () => {
		expect(countWordsInMessageContent("")).toBe(0);
	});
});

describe("extractContextPrompt", () => {
	it("returns first 200 words of string content", () => {
		const longText = Array.from({ length: 300 }, (_, i) => `word${i}`).join(" ");
		const result = extractContextPrompt(longText);
		const wordCount = result?.split(" ").length;
		expect(wordCount).toBe(200);
	});

	it("returns full content when under 200 words", () => {
		expect(extractContextPrompt("short text")).toBe("short text");
	});

	it("returns extracted text from array content", () => {
		const blocks = [{ type: "text", text: "hello world" }];
		expect(extractContextPrompt(blocks)).toBe("hello world");
	});

	it("returns null for other types", () => {
		expect(extractContextPrompt(42)).toBeNull();
		expect(extractContextPrompt(null)).toBeNull();
	});
});

describe("prepareContextMessages", () => {
	const envPreamble = "[ENVIRONMENT]\nSome env info";

	it("finds system message", () => {
		const messages = [
			{ role: "system", content: "You are helpful" },
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there" },
		];
		const result = prepareContextMessages(messages, envPreamble);
		expect(result.systemMsg).toEqual({ role: "system", content: "You are helpful" });
	});

	it("finds developer message as system message", () => {
		const messages = [
			{ role: "developer", content: "System instructions" },
			{ role: "user", content: "Hello" },
		];
		const result = prepareContextMessages(messages, envPreamble);
		expect(result.systemMsg).toEqual({ role: "developer", content: "System instructions" });
	});

	it("returns null systemMsg when no system message present", () => {
		const messages = [
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi" },
		];
		const result = prepareContextMessages(messages, envPreamble);
		expect(result.systemMsg).toBeNull();
	});

	it("augments string user message with env preamble when no [ENVIRONMENT] present", () => {
		const messages = [{ role: "user", content: "do something" }];
		const result = prepareContextMessages(messages, envPreamble);
		const lastMsg = result.augmentedMessages[result.augmentedMessages.length - 1] as { content: string };
		expect(lastMsg.content).toContain("[ENVIRONMENT]");
		expect(lastMsg.content).toContain("[ACTIONABLE PROMPT]");
		expect(lastMsg.content).toContain("do something");
	});

	it("does not augment user message that already has [ENVIRONMENT]", () => {
		const messages = [{ role: "user", content: "[ENVIRONMENT]\nSome env\n\n[ACTIONABLE PROMPT]\ndo something" }];
		const result = prepareContextMessages(messages, envPreamble);
		const lastMsg = result.augmentedMessages[result.augmentedMessages.length - 1] as { content: string };
		// Should NOT be doubly-prepended
		expect(lastMsg.content).toBe("[ENVIRONMENT]\nSome env\n\n[ACTIONABLE PROMPT]\ndo something");
	});

	it("augments array user message with text first block", () => {
		const messages = [
			{
				role: "user",
				content: [{ type: "text", text: "do something" }],
			},
		];
		const result = prepareContextMessages(messages, envPreamble);
		const lastMsg = result.augmentedMessages[result.augmentedMessages.length - 1] as {
			content: Array<{ type: string; text: string }>;
		};
		expect(lastMsg.content[0].text).toContain("[ENVIRONMENT]");
		expect(lastMsg.content[0].text).toContain("do something");
	});

	it("augments array user message with non-text first block by prepending text block", () => {
		const messages = [
			{
				role: "user",
				content: [{ type: "image", url: "http://example.com/img.png" }],
			},
		];
		const result = prepareContextMessages(messages, envPreamble);
		const lastMsg = result.augmentedMessages[result.augmentedMessages.length - 1] as {
			content: Array<{ type: string; text?: string }>;
		};
		// Should prepend a text block before the image
		expect(lastMsg.content[0].type).toBe("text");
		expect((lastMsg.content[0] as { text: string }).text).toContain("[ENVIRONMENT]");
	});

	it("does not augment array with [ENVIRONMENT] already present in first text block", () => {
		const messages = [
			{
				role: "user",
				content: [{ type: "text", text: "[ENVIRONMENT]\nstuff\n\ndo something" }],
			},
		];
		const result = prepareContextMessages(messages, envPreamble);
		const lastMsg = result.augmentedMessages[result.augmentedMessages.length - 1] as {
			content: Array<{ type: string; text: string }>;
		};
		// Should not be modified
		expect(lastMsg.content[0].text).toBe("[ENVIRONMENT]\nstuff\n\ndo something");
	});

	it("computes currentMessages from last user message onward", () => {
		const messages = [
			{ role: "system", content: "sys" },
			{ role: "user", content: "hi" },
			{ role: "assistant", content: "hey" },
			{ role: "user", content: "what up" },
		];
		const result = prepareContextMessages(messages, envPreamble);
		// currentMessages should start from the last user message
		expect(result.currentMessages.length).toBeGreaterThanOrEqual(1);
		const firstUser = result.currentMessages.find((m) => (m as { role: string }).role === "user");
		expect(firstUser).toBeDefined();
	});

	it("returns empty currentMessages when no user message exists", () => {
		const messages = [
			{ role: "system", content: "sys" },
			{ role: "assistant", content: "hey" },
		];
		const result = prepareContextMessages(messages, envPreamble);
		expect(result.hasUserMessage).toBe(false);
		expect(result.userPrompt).toBeNull();
	});

	it("computes rawPromptWordCount", () => {
		const messages = [{ role: "user", content: "hello world foo" }];
		const result = prepareContextMessages(messages, envPreamble);
		expect(result.rawPromptWordCount).toBe(3);
	});

	it("computes rawPromptWordCount as 0 when no user message", () => {
		const messages = [{ role: "assistant", content: "hey" }];
		const result = prepareContextMessages(messages, envPreamble);
		expect(result.rawPromptWordCount).toBe(0);
	});

	it("extracts userPrompt from augmented string content", () => {
		const messages = [{ role: "user", content: "do the thing" }];
		const result = prepareContextMessages(messages, envPreamble);
		expect(result.hasUserMessage).toBe(true);
		// userPrompt is extracted from augmented messages which include the preamble
		expect(result.userPrompt).toContain("do the thing");
	});

	it("handles empty messages array", () => {
		const result = prepareContextMessages([], envPreamble);
		expect(result.systemMsg).toBeNull();
		expect(result.hasUserMessage).toBe(false);
		expect(result.rawPromptWordCount).toBe(0);
	});
});

describe("shouldDropRelevanceList", () => {
	it("returns true when DROP_RELEVANCE_LIST is '1'", () => {
		expect(shouldDropRelevanceList(50, { DROP_RELEVANCE_LIST: "1" })).toBe(true);
	});

	it("returns true when DROP_RELEVANCE_LIST is 'true'", () => {
		expect(shouldDropRelevanceList(50, { DROP_RELEVANCE_LIST: "true" })).toBe(true);
	});

	it("returns true when word count is below minimum", () => {
		expect(shouldDropRelevanceList(10, {})).toBe(true);
	});

	it("returns false when word count is at or above minimum", () => {
		expect(shouldDropRelevanceList(20, {})).toBe(false);
		expect(shouldDropRelevanceList(25, {})).toBe(false);
	});

	it("respects custom RELEVANCE_LIST_MIN_WORDS", () => {
		expect(shouldDropRelevanceList(5, { RELEVANCE_LIST_MIN_WORDS: "3" })).toBe(false);
		expect(shouldDropRelevanceList(2, { RELEVANCE_LIST_MIN_WORDS: "3" })).toBe(true);
	});
});

describe("shouldDropEmbedding", () => {
	it("returns true when word count is below minimum", () => {
		expect(shouldDropEmbedding(10, {})).toBe(true);
	});

	it("returns false when word count is at or above minimum", () => {
		expect(shouldDropEmbedding(20, {})).toBe(false);
		expect(shouldDropEmbedding(25, {})).toBe(false);
	});

	it("respects custom RELEVANCE_LIST_MIN_WORDS", () => {
		expect(shouldDropEmbedding(5, { RELEVANCE_LIST_MIN_WORDS: "3" })).toBe(false);
		expect(shouldDropEmbedding(2, { RELEVANCE_LIST_MIN_WORDS: "3" })).toBe(true);
	});

	it("returns true for zero words", () => {
		expect(shouldDropEmbedding(0, {})).toBe(true);
	});
});

function emptyBlocks(): ContextBlocks {
	return {
		systemMsg: null,
		sessionArchitecture: null,
		workingMemory: null,
		routingInstructions: null,
		preamble: null,
		recencyList: null,
		relevanceList: null,
		followUpMsg: null,
		checkpointMsg: null,
		contractMsg: null,
	};
}

describe("assembleContextPrefix", () => {
	it("returns empty array when all blocks are null", () => {
		expect(assembleContextPrefix(emptyBlocks())).toEqual([]);
	});

	it("includes system message when provided", () => {
		const systemMsg = { role: "system", content: "You are helpful" };
		const result = assembleContextPrefix({ ...emptyBlocks(), systemMsg });
		expect(result).toHaveLength(1);
		expect(result[0]).toEqual(systemMsg);
	});

	it("includes session architecture when provided", () => {
		const arch = "[SESSION ARCHITECTURE]\nsome text";
		const result = assembleContextPrefix({ ...emptyBlocks(), sessionArchitecture: arch });
		expect(result).toHaveLength(1);
		expect((result[0] as { content: Array<{ type: string; text: string }> }).content[0].text).toBe(arch);
	});

	it("injects session architecture after system and before preamble", () => {
		const systemMsg = { role: "system", content: "sys" };
		const arch = "[SESSION ARCHITECTURE]";
		const preamble = "preamble text";
		const result = assembleContextPrefix({ ...emptyBlocks(), systemMsg, sessionArchitecture: arch, preamble });
		expect(result).toHaveLength(3);
		expect(result[0]).toEqual(systemMsg);
		expect((result[1] as { content: Array<{ type: string; text: string }> }).content[0].text).toBe(arch);
		expect((result[2] as { content: Array<{ type: string; text: string }> }).content[0].text).toBe(preamble);
	});

	it("includes working memory after session architecture, before preamble", () => {
		const arch = "[SESSION ARCHITECTURE]";
		const wm = "[WORKING MEMORY]";
		const preamble = "preamble";
		const result = assembleContextPrefix({
			...emptyBlocks(),
			sessionArchitecture: arch,
			workingMemory: wm,
			preamble,
		});
		expect(result).toHaveLength(3);
		const texts = result.map((m) => (m as { content: Array<{ type: string; text: string }> }).content[0].text);
		expect(texts).toEqual([arch, wm, preamble]);
	});

	it("includes all components in correct fixed order", () => {
		const systemMsg = { role: "system", content: "sys" };
		const arch = "sa";
		const wm = "wm";
		const preamble = "pr";
		const recency = "rc";
		const relevance = "rl";
		const followUp = { role: "user", content: [{ type: "text", text: "fu" }] };
		const checkpoint = { role: "user", content: [{ type: "text", text: "cp" }] };
		const contract = { role: "user", content: [{ type: "text", text: "ct" }] };
		const result = assembleContextPrefix({
			systemMsg,
			sessionArchitecture: arch,
			workingMemory: wm,
			routingInstructions: null,
			preamble,
			recencyList: recency,
			relevanceList: relevance,
			followUpMsg: followUp,
			checkpointMsg: checkpoint,
			contractMsg: contract,
		});
		expect(result).toHaveLength(9);
		const roles = result.map((m) => (m as { role: string }).role);
		expect(roles).toEqual(["system", "user", "user", "user", "user", "user", "user", "user", "user"]);
	});

	it("includes recency, sessionArchitecture, preamble, and contract when relevance is null (short-prompt scenario)", () => {
		const systemMsg = { role: "system", content: "You are helpful" };
		const arch = "[SESSION ARCHITECTURE]";
		const preamble = "[CONTEXT BUILDING REFERENCES]\n...";
		const recencyList = "--- RECENCY LIST ---\n...";
		const contract = {
			role: "user",
			content: [{ type: "text", text: "[FINAL RESPONSE CONTRACT]" }],
		};

		const result = assembleContextPrefix({
			...emptyBlocks(),
			systemMsg,
			sessionArchitecture: arch,
			preamble,
			recencyList,
			contractMsg: contract,
		});

		expect(result).toHaveLength(5);
		expect((result[0] as { role: string }).role).toBe("system");
		const userTexts = result
			.slice(1)
			.map((m) => (m as { content: Array<{ type: string; text: string }> }).content[0].text);
		expect(userTexts[0]).toBe("[SESSION ARCHITECTURE]");
		expect(userTexts[1]).toBe("[CONTEXT BUILDING REFERENCES]\n...");
		expect(userTexts[2]).toBe("--- RECENCY LIST ---\n...");
		expect(userTexts[3]).toBe("[FINAL RESPONSE CONTRACT]");
	});

	it("produces only system + sessionArchitecture + contract when recency and preamble are null", () => {
		const systemMsg = { role: "system", content: "You are helpful" };
		const arch = "[SESSION ARCHITECTURE]";
		const contract = {
			role: "user",
			content: [{ type: "text", text: "[FINAL RESPONSE CONTRACT]" }],
		};

		const result = assembleContextPrefix({
			...emptyBlocks(),
			systemMsg,
			sessionArchitecture: arch,
			contractMsg: contract,
		});

		expect(result).toHaveLength(3);
		expect(result[0]).toEqual(systemMsg);
		expect((result[1] as { content: Array<{ type: string; text: string }> }).content[0].text).toBe(arch);
		expect(result[2]).toEqual(contract);
	});

	it("followUp and checkpoint appear after relevance, before contract", () => {
		const arch = "sa";
		const relevance = "rl";
		const followUp = { role: "user", content: [{ type: "text", text: "fu" }] };
		const checkpoint = { role: "user", content: [{ type: "text", text: "cp" }] };
		const contract = { role: "user", content: [{ type: "text", text: "ct" }] };
		const result = assembleContextPrefix({
			...emptyBlocks(),
			sessionArchitecture: arch,
			relevanceList: relevance,
			followUpMsg: followUp,
			checkpointMsg: checkpoint,
			contractMsg: contract,
		});
		const texts = result.map((m) => (m as { content: Array<{ type: string; text: string }> }).content[0].text);
		expect(texts).toEqual([arch, relevance, "fu", "cp", "ct"]);
	});
});
