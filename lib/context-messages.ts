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
 * Build the context message prefix block: system + preamble + recency + relevance + contract.
 * The contract message is always last, immediately before the actionable prompt.
 */
export function buildContextMessagePrefix(
	systemMsg: unknown | null,
	preamble: string | null,
	recencyList: string | null,
	relevanceList: string | null,
	contractMsg?: { role: string; content: Array<{ type: string; text: string }> } | null,
): unknown[] {
	const finalMessages: unknown[] = [];
	if (systemMsg) finalMessages.push(systemMsg);
	if (preamble) finalMessages.push({ role: "user" as const, content: [{ type: "text" as const, text: preamble }] });
	if (recencyList)
		finalMessages.push({ role: "user" as const, content: [{ type: "text" as const, text: recencyList }] });
	if (relevanceList)
		finalMessages.push({ role: "user" as const, content: [{ type: "text" as const, text: relevanceList }] });
	if (contractMsg) finalMessages.push(contractMsg);
	return finalMessages;
}
