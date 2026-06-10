import { extractText } from "./message-content.ts";

export interface PreparedContextMessages {
	systemMsg: unknown | null;
	augmentedMessages: unknown[];
	currentMessages: unknown[];
	hasUserMessage: boolean;
	userPrompt: string | null;
	rawPromptWordCount: number;
}

export function startsWithEnvironmentPreamble(content: string): boolean {
	return content.trimStart().startsWith("[ENVIRONMENT]");
}

export function countWordsInMessageContent(content: unknown): number {
	const text = typeof content === "string" ? content : Array.isArray(content) ? extractText(content) : "";
	return text.split(/\s+/).filter((word) => word.length > 0).length;
}

export function extractContextPrompt(content: unknown): string | null {
	if (typeof content === "string") return content.split(" ").slice(0, 200).join(" ");
	if (Array.isArray(content)) return extractText(content);
	return null;
}

export function prepareContextMessages(messages: readonly unknown[], envPreamble: string): PreparedContextMessages {
	const systemMsg =
		messages.find(
			(m) => (m as { role?: string }).role === "system" || (m as { role?: string }).role === "developer",
		) ?? null;
	const lastUserIdx = messages.reduce<number>(
		(last, m, i) => ((m as { role?: string }).role === "user" ? i : last),
		-1,
	);

	const rawPromptWordCount =
		lastUserIdx >= 0 ? countWordsInMessageContent((messages[lastUserIdx] as { content?: unknown }).content) : 0;

	const augmentedMessages = [...messages];
	if (lastUserIdx >= 0) {
		const userMsg = augmentedMessages[lastUserIdx];
		const userContent = (userMsg as { content?: unknown }).content;
		const userMsgAny = userMsg as Record<string, unknown>;
		if (typeof userContent === "string") {
			if (!startsWithEnvironmentPreamble(userContent)) {
				augmentedMessages[lastUserIdx] = {
					...userMsgAny,
					content: `${envPreamble}\n\n[ACTIONABLE PROMPT]\n${userContent}`,
				};
			}
		} else if (
			Array.isArray(userContent) &&
			userContent.length > 0 &&
			(userContent[0] as Record<string, unknown>).type === "text"
		) {
			const firstBlock = userContent[0] as { type: string; text: string };
			if (!startsWithEnvironmentPreamble(firstBlock.text)) {
				const newContent = [...userContent];
				newContent[0] = { ...firstBlock, text: `${envPreamble}\n\n[ACTIONABLE PROMPT]\n${firstBlock.text}` };
				augmentedMessages[lastUserIdx] = { ...userMsgAny, content: newContent };
			}
		} else if (Array.isArray(userContent)) {
			augmentedMessages[lastUserIdx] = {
				...userMsgAny,
				content: [{ type: "text", text: `${envPreamble}\n\n[ACTIONABLE PROMPT]\n` }, ...userContent],
			};
		}
	}

	const currentMessages = lastUserIdx >= 0 ? augmentedMessages.slice(lastUserIdx) : [...augmentedMessages];
	const userMessages = currentMessages.filter((m) => (m as { role?: string }).role === "user");
	if (userMessages.length === 0) {
		return {
			systemMsg,
			augmentedMessages,
			currentMessages,
			hasUserMessage: false,
			userPrompt: null,
			rawPromptWordCount,
		};
	}

	const lastUserContent = (userMessages[userMessages.length - 1] as { content?: unknown }).content;
	return {
		systemMsg,
		augmentedMessages,
		currentMessages,
		hasUserMessage: true,
		userPrompt: extractContextPrompt(lastUserContent),
		rawPromptWordCount,
	};
}

export function shouldDropRelevanceList(
	rawPromptWordCount: number,
	env: { DROP_RELEVANCE_LIST?: string; RELEVANCE_LIST_MIN_WORDS?: string } = process.env,
): boolean {
	const minWords = Number.parseInt(env.RELEVANCE_LIST_MIN_WORDS ?? "20", 10);
	return env.DROP_RELEVANCE_LIST === "1" || env.DROP_RELEVANCE_LIST === "true" || rawPromptWordCount < minWords;
}

/**
 * Check whether the prompt embedding should be skipped for short prompts.
 * Uses the same RELEVANCE_LIST_MIN_WORDS threshold (default: 20 words).
 * Short prompts produce noisy embeddings that waste API credits, pollute the
 * vector index, and cause false-positive search matches.
 *
 * When this returns true:
 * - No embedText() call for the prompt vector
 * - No :prompt row written to index.csv
 * - Response embedding is still computed and written as :response
 * - promptEmbedding in round JSON uses the response vector (not prompt+response)
 * - assignToGroup() uses the response vector (not a null signal)
 */
export function shouldDropEmbedding(
	rawPromptWordCount: number,
	env: { RELEVANCE_LIST_MIN_WORDS?: string } = process.env,
): boolean {
	const minWords = Number.parseInt(env.RELEVANCE_LIST_MIN_WORDS ?? "20", 10);
	return rawPromptWordCount < minWords;
}

/**
 * Blocks that form the context prefix — each is optional. The assembler produces
 * a fixed-order message array: system → sessionArchitecture → workingMemory →
 * preamble → recency → relevance → followUp → checkpoint → contract.
 */
export interface ContextBlocks {
	systemMsg: unknown | null;
	sessionArchitecture: string | null;
	workingMemory: string | null;
	preamble: string | null;
	recencyList: string | null;
	relevanceList: string | null;
	followUpMsg: unknown | null;
	checkpointMsg: unknown | null;
	contractMsg: { role: string; content: Array<{ type: string; text: string }> } | null;
}

/** Build a {role:"user", content: [{type:"text", text}]} message. */
function userTextMsg(text: string): unknown {
	return { role: "user" as const, content: [{ type: "text" as const, text }] };
}

/**
 * Assemble the context prefix from typed blocks. Every block is optional; only
 * non-null values are pushed. Order: system → sessionArchitecture →
 * workingMemory → preamble → recency → relevance → followUp → checkpoint → contract.
 *
 * This replaces the 5 duplicated inline assembly sites in src/semblr.ts.
 */
export function assembleContextPrefix(blocks: ContextBlocks): unknown[] {
	const out: unknown[] = [];
	if (blocks.systemMsg) out.push(blocks.systemMsg);
	if (blocks.sessionArchitecture) out.push(userTextMsg(blocks.sessionArchitecture));
	if (blocks.workingMemory) out.push(userTextMsg(blocks.workingMemory));
	if (blocks.preamble) out.push(userTextMsg(blocks.preamble));
	if (blocks.recencyList) out.push(userTextMsg(blocks.recencyList));
	if (blocks.relevanceList) out.push(userTextMsg(blocks.relevanceList));
	if (blocks.followUpMsg) out.push(blocks.followUpMsg);
	if (blocks.checkpointMsg) out.push(blocks.checkpointMsg);
	if (blocks.contractMsg) out.push(blocks.contractMsg);
	return out;
}
