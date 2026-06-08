export function estimateTokens(text: string): number {
	return Math.ceil(text.length / 4);
}

/**
 * Estimate total tokens in an array of LLM messages.
 * Handles both string content and content-block arrays.
 */
export function estimateMessagesTokens(messages: unknown[]): number {
	let total = 0;
	for (const msg of messages) {
		const content = (msg as { content?: unknown }).content;
		if (typeof content === "string") {
			total += estimateTokens(content);
		} else if (Array.isArray(content)) {
			for (const block of content) {
				if (typeof block === "string") {
					total += estimateTokens(block);
				} else if ((block as { text?: string }).text) {
					total += estimateTokens((block as { text: string }).text);
				}
			}
		}
	}
	return total;
}
