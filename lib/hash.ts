import crypto from "node:crypto";

export interface HashToolCallDetail {
	arguments: string;
	result_summary?: string;
	result_full?: string;
}

export function computeContentHash(userPrompt: string, responseText: string, toolCalls?: HashToolCallDetail[]): string {
	const parts: string[] = [userPrompt, responseText];
	if (toolCalls) {
		for (const tc of toolCalls) {
			parts.push(tc.arguments);
			parts.push(tc.result_full ?? tc.result_summary ?? "");
		}
	}
	return crypto.createHash("md5").update(parts.join("")).digest("hex");
}

export function createRoundFilePath(
	userPrompt: string,
	responseText: string,
	toolCalls?: HashToolCallDetail[],
): string {
	return `${computeContentHash(userPrompt, responseText, toolCalls)}.json`;
}
