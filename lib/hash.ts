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

export interface DuplicateKeyToolCall {
	name?: string;
	arguments?: string;
	result_summary?: string;
	result_full?: string;
}

export interface DuplicateKeyRound {
	userPrompt?: string;
	responseSequence?: string;
	toolCalls?: DuplicateKeyToolCall[];
}

/**
 * Whitespace-insensitive content key for duplicate detection (shared with
 * scripts/prune-recovery-duplicates.ts): prompts trimmed, tool results
 * trimEnd'ed. Rounds whose content differs only in trailing whitespace are
 * the same round — this absorbs the serialization drift between legacy live
 * saves (older semblr code stripped trailing newlines and omitted tool-call
 * ids) and the current recovery parser, so the backfill can recognize those
 * legacy rounds by content instead of by unreproducible historical filenames.
 */
export function canonicalDuplicateKey(round: DuplicateKeyRound): string {
	const tools = (round.toolCalls ?? []).map((tc) =>
		[tc.name ?? "", tc.arguments ?? "", (tc.result_full ?? tc.result_summary ?? "").trimEnd()].join("\u0000"),
	);
	return JSON.stringify([(round.userPrompt ?? "").trim(), (round.responseSequence ?? "").trim(), tools]);
}
