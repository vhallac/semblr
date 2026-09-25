import { createHash } from "node:crypto";
import * as fs from "node:fs";
import { computeContentHash } from "./hash.ts";
import { extractText } from "./message-content.ts";
import type { ChainEntry, CheckpointSummary, ResponseSegment, RoundData, ToolCallDetail } from "./round-data.ts";

export function extractAgentEndUserPrompt(cachedPrompt: string | null, messages?: readonly unknown[]): string {
	let userPrompt = cachedPrompt ?? "";
	if (!userPrompt && messages) {
		const lastUser = [...messages].reverse().find((m) => (m as { role: string }).role === "user");
		if (lastUser) {
			const content = (lastUser as { content: unknown }).content;
			if (typeof content === "string") {
				userPrompt = content;
			} else if (Array.isArray(content)) {
				userPrompt = extractText(content as Array<{ type: string; text?: string }>);
			}
		}
	}
	return userPrompt;
}

export function extractAgentEndResponseText(accumulatedText: readonly string[], messages?: readonly unknown[]): string {
	let responseText = accumulatedText.join("\n\n").trim();
	if (!responseText) {
		const lastAssistant = messages
			? [...messages].reverse().find((m) => (m as { role: string }).role === "assistant")
			: null;
		if (lastAssistant) {
			const content = (lastAssistant as { content: unknown }).content;
			if (typeof content === "string") {
				responseText = content;
			} else if (Array.isArray(content)) {
				responseText = extractText(content as Array<{ type: string; text?: string }>);
			}
		}
	}
	return responseText;
}

export function buildAgentEndToolSummary(toolCallCount: number, toolCallNames: readonly string[]): string {
	return toolCallCount > 0 ? `${toolCallCount} tools (${toolCallNames.join(", ")})` : "0 tools (discussion)";
}

export function buildAgentEndChainEntry(
	fileName: string,
	userPrompt: string,
	responseText: string,
	toolCallCount: number,
	toolCallNames: readonly string[],
): ChainEntry {
	return {
		fileName,
		userPrompt,
		responseSequence: responseText,
		toolSummary: buildAgentEndToolSummary(toolCallCount, toolCallNames),
	};
}

export function getAgentEndParentId(chain: readonly { fileName: string }[]): string | null {
	return chain.length >= 2 ? chain[chain.length - 2].fileName : null;
}

export function buildAgentEndRoundData(args: {
	userPrompt: string;
	responseText: string;
	turnIndex: number | null;
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
	responseSegments: ResponseSegment[];
	parentId: string | null;
	userTimestamp?: number;
	needsFollowup?: boolean;
	summary?: CheckpointSummary;
}): Record<string, unknown> {
	return {
		id: computeContentHash(args.userPrompt, args.responseText, args.toolCalls),
		userPrompt: args.userPrompt,
		responseSequence: args.responseText,
		turnIndex: args.turnIndex ?? 0,
		userTimestamp: args.userTimestamp ?? Date.now(),
		toolCallCount: args.toolCallCount,
		toolCallNames: args.toolCallNames,
		toolCalls: args.toolCalls,
		responseSegments: args.responseSegments,
		promptEmbedding: undefined,
		parentId: args.parentId,
		relatedParentId: null,
		needsFollowup: args.needsFollowup ?? false,
		...(args.summary ? { summary: args.summary } : {}),
	};
}

export function embeddingMaxTokensToResponseBytes(embeddingMaxTokens: number): number {
	return Math.max(0, Math.floor(embeddingMaxTokens * 3));
}

/** Thresholds for embedding-input prompt cleanup; non-positive values disable the collapse. */
export interface PromptNoiseOptions {
	/** Code fences whose body exceeds this many chars collapse to a placeholder; non-positive disables. */
	fenceMaxChars: number;
	/** JSON dumps whose text exceeds this many chars collapse to a placeholder; non-positive disables. */
	jsonMaxChars: number;
	/** Repeat runs (same char hammered, or a single no-space token) longer than this collapse; non-positive disables. */
	repeatMaxChars: number;
}

/** Default cleanup thresholds (issue #106 Stage 1): collapse fences/JSON dumps past ~600 chars, repetition past ~200. */
export const DEFAULT_PROMPT_NOISE_CLEANUP: PromptNoiseOptions = {
	fenceMaxChars: 600,
	jsonMaxChars: 600,
	repeatMaxChars: 200,
};

const PLACEHOLDER_LINE_CLIP = 80;
const JSON_PARSE_LENGTH_CAP = 1_000_000;

function clipPlaceholderLine(line: string): string {
	const trimmed = line.trim();
	return trimmed.length > PLACEHOLDER_LINE_CLIP ? `${trimmed.slice(0, PLACEHOLDER_LINE_CLIP)}…` : trimmed;
}

function collapseCodeFences(prompt: string, fenceMaxChars: number): string {
	if (fenceMaxChars <= 0) return prompt;
	// A fence opens with 3+ backticks/tildes at line start (optional indent) and closes with the
	// same marker at line start; the lazy body ends the block at the first closing line.
	const fenceRe = /^([ \t]*)(`{3,}|~{3})[^\n]*\n([\s\S]*?)^\1\2[^\n]*$/gm;
	return prompt.replace(fenceRe, (match, indent: string, marker: string, body: string) => {
		if (body.length <= fenceMaxChars) return match;
		const infoLine = match.slice(indent.length + marker.length);
		const langToken = infoLine.trim().split(/\s+/)[0]?.slice(0, 24) ?? "";
		const lang = /^[A-Za-z0-9_+#.-]{1,24}$/.test(langToken) ? langToken : "plain";
		const bodyText = body.endsWith("\n") ? body.slice(0, -1) : body;
		const bodyLines = bodyText.split("\n");
		const first = clipPlaceholderLine(bodyLines[0] ?? "");
		const last = clipPlaceholderLine(bodyLines[bodyLines.length - 1] ?? "");
		return `[CODE_BLOCK: ~${body.length} chars, ${lang}, first line: "${first}", last line: "${last}"]`;
	});
}

/** Cheap prefilter: is the bracket run at `open` plausibly the start of a JSON value? */
function looksLikeJsonStart(text: string, open: number): boolean {
	let i = open + 1;
	while (i < text.length && /\s/.test(text[i])) i++;
	if (i >= text.length) return false;
	const ch = text[i];
	if (text[open] === "{") {
		return ch === '"' || ch === "}";
	}
	return (
		ch === '"' ||
		ch === "{" ||
		ch === "[" ||
		ch === "]" ||
		ch === "-" ||
		ch === "t" ||
		ch === "f" ||
		ch === "n" ||
		/\d/.test(ch)
	);
}

/** Find the bracket matching the opener at `open`, honoring JSON string escapes; -1 if unbalanced. */
function findMatchingBracket(text: string, open: number): number {
	const openCh = text[open];
	const closeCh = openCh === "{" ? "}" : "]";
	let depth = 0;
	let inString = false;
	let escaped = false;
	for (let i = open; i < text.length; i++) {
		const ch = text[i];
		if (inString) {
			if (escaped) escaped = false;
			else if (ch === "\\") escaped = true;
			else if (ch === '"') inString = false;
			continue;
		}
		if (ch === '"') {
			inString = true;
		} else if (ch === openCh) {
			depth++;
		} else if (ch === closeCh) {
			depth--;
			if (depth === 0) return i;
		}
	}
	return -1;
}

function isJsonObjectOrArray(candidate: string): boolean {
	if (candidate.length > JSON_PARSE_LENGTH_CAP) return false;
	try {
		const parsed: unknown = JSON.parse(candidate);
		return typeof parsed === "object" && parsed !== null;
	} catch {
		return false;
	}
}

function collapseJsonDumps(text: string, jsonMaxChars: number): string {
	if (jsonMaxChars <= 0) return text;
	let result = "";
	let i = 0;
	while (i < text.length) {
		const ch = text[i];
		if (ch !== "{" && ch !== "[") {
			result += ch;
			i++;
			continue;
		}
		const close = looksLikeJsonStart(text, i) ? findMatchingBracket(text, i) : -1;
		if (close === -1) {
			result += ch;
			i++;
			continue;
		}
		const candidate = text.slice(i, close + 1);
		if (candidate.length > jsonMaxChars && isJsonObjectOrArray(candidate)) {
			result += `[JSON_DUMP: ~${candidate.length} chars]`;
			i = close + 1;
			continue;
		}
		result += ch;
		i++;
	}
	return result;
}

/** Escaped, single-line rendering of the repeated character for the REPEAT placeholder. */
function describeRepeatChar(ch: string): string {
	return JSON.stringify(ch).slice(1, -1);
}

/**
 * Collapse zero-variety repetition runs: the same character hammered past `repeatMaxChars`
 * (paste glitches like `pppp…`, oversized rules `----…`, huge whitespace) becomes
 * `[REPEAT: 'c' × M]`. Runs at or below the threshold are left verbatim.
 */
function collapseRepeatRuns(prompt: string, repeatMaxChars: number): string {
	if (repeatMaxChars <= 0) return prompt;
	// Match from 11 chars up so the replacement callback only runs on plausible noise runs;
	// dotAll so line-terminator runs (huge newline/whitespace gaps) are guarded too.
	return prompt.replace(/(.)\1{10,}/gs, (run, ch: string) =>
		run.length > repeatMaxChars ? `[REPEAT: '${describeRepeatChar(ch)}' × ${run.length}]` : run,
	);
}

/**
 * Collapse solid no-space tokens (base64 blobs, hex dumps, minified one-liners, periodic junk
 * like `ababab…`) longer than `repeatMaxChars` to `[LONG_TOKEN: ~M chars]`. Runs after the
 * single-char collapse, so it only claims mixed-content runs the char guard did not.
 */
function collapseLongTokens(prompt: string, repeatMaxChars: number): string {
	if (repeatMaxChars <= 0) return prompt;
	const longTokenRe = new RegExp(`[^\\s]{${repeatMaxChars + 1},}`, "g");
	return prompt.replace(longTokenRe, (token) => `[LONG_TOKEN: ~${token.length} chars]`);
}

/**
 * Collapse embedding noise in a user prompt (issue #106 Stage 1, derived-not-stored):
 * large code fences become `[CODE_BLOCK: ~M chars, lang, first/last line]` placeholders,
 * parse-validated JSON dumps become `[JSON_DUMP: ~M chars]` placeholders, and zero-variety
 * repetition runs (hammered chars, solid long tokens) become `[REPEAT: …]` / `[LONG_TOKEN: …]`
 * placeholders. The raw prompt stays verbatim in the round file; the cleaned view is only an
 * embedding-input transform.
 */
export function cleanPromptNoise(prompt: string, options: PromptNoiseOptions = DEFAULT_PROMPT_NOISE_CLEANUP): string {
	// Fences first: fenced JSON would otherwise be matched (and mis-labeled) by the JSON scan.
	let cleaned = collapseCodeFences(prompt, options.fenceMaxChars);
	cleaned = collapseJsonDumps(cleaned, options.jsonMaxChars);
	// Repetition last: it only claims residual runs the structural collapses did not absorb.
	cleaned = collapseRepeatRuns(cleaned, options.repeatMaxChars);
	cleaned = collapseLongTokens(cleaned, options.repeatMaxChars);
	return cleaned;
}

/**
 * Assemble the combined embedding text for a saved round.
 * The prompt is expected pre-cleaned via `cleanPromptNoise` (embedding-input transform);
 * the response is clipped to the configured embedding budget with REDACTED markers stripped.
 */
export function buildAgentEndEmbeddingTexts(
	userPrompt: string,
	responseText: string,
	maxResponseBytes = embeddingMaxTokensToResponseBytes(8000),
): { clippedResponse: string; combinedText: string } {
	const strippedResponse = responseText.replace(/\[(?:Tool call )?REDACTED[^\]]*\]\n?/g, "");
	const responseBuf = Buffer.from(strippedResponse, "utf-8");
	const clippedResponse =
		responseBuf.length > maxResponseBytes
			? responseBuf.slice(0, maxResponseBytes).toString("utf-8")
			: strippedResponse;
	return {
		clippedResponse,
		combinedText: `${userPrompt}\n\n${clippedResponse}`,
	};
}

/**
 * Hash the exact embedding input text (sha256, base64url). Used as the 4th index
 * CSV column on :prompt rows so the re-embed migration can detect when cleanup
 * heuristics or thresholds change what would be embedded (issue #106).
 */
export function hashEmbeddingInput(text: string): string {
	return createHash("sha256").update(text, "utf-8").digest("base64url");
}

/**
 * The current prompt-side embedding-input convention (issue #106, single source
 * of truth for capture, the digest scripts, and the re-embed migration): the
 * noise-cleaned prompt, clipped to `maxTokens` characters AFTER cleanup, plus
 * its `hashEmbeddingInput` stamp over the exact final input.
 *
 * Cleanup runs first so placeholders shrink high-entropy spans before the budget
 * is applied; the clip then still bounds the input actually sent to the embedding
 * API for prompts cleanup cannot shrink (issue #107 F4 — the unbounded full-text
 * embedding that replaced the legacy raw `slice(0, embeddingMaxTokens)` clip is
 * not restored verbatim, but bounded post-cleanup). The clip budget comes from
 * the configured `embeddingMaxTokens`; the default mirrors the config default
 * (same precedent as `buildAgentEndEmbeddingTexts`). Non-positive `maxTokens`
 * disables the clip (same convention as the noise options).
 */
export function buildPromptEmbeddingInput(
	userPrompt: string,
	options: PromptNoiseOptions = DEFAULT_PROMPT_NOISE_CLEANUP,
	maxTokens = 8000,
): { text: string; hash: string } {
	const cleaned = cleanPromptNoise(userPrompt, options);
	const text = maxTokens > 0 && cleaned.length > maxTokens ? cleaned.slice(0, maxTokens) : cleaned;
	return { text, hash: hashEmbeddingInput(text) };
}

/**
 * Detect and strip the round_needs_followup marker from the end of a response.
 * Returns the stripped text and whether the marker was present.
 */
export function extractAndStripFollowupMarker(responseText: string): {
	cleanedText: string;
	needsFollowup: boolean;
} {
	const followupMarker = "\nround_needs_followup";
	if (responseText.endsWith(followupMarker)) {
		return {
			cleanedText: responseText.slice(0, -followupMarker.length).trimEnd(),
			needsFollowup: true,
		};
	}
	return { cleanedText: responseText, needsFollowup: false };
}

/**
 * Read a round file and check its needsFollowup flag.
 * Returns the round data if needsFollowup is true, or null.
 *
 * NOTE: This function does NOT clear the flag anymore — the flag is preserved
 * in the saved JSON as permanent metadata. Runtime injection gating is handled
 * by an in-memory set (injectedFollowupRounds) in semblr.ts, not by mutating
 * the file. See https://github.com/vhallac/semblr/issues/XX
 */
export function readAndClearFollowupFlag(
	fullPath: string,
	fsImpl?: {
		existsSync: (p: string) => boolean;
		readFileSync: (p: string, encoding: "utf-8") => string;
		writeFileSync: (p: string, data: string) => void;
		renameSync: (oldP: string, newP: string) => void;
	},
): RoundData | null {
	const fs_ = fsImpl ?? fs;

	if (!fs_.existsSync(fullPath)) return null;

	let roundData: Record<string, unknown>;
	try {
		roundData = JSON.parse(fs_.readFileSync(fullPath, "utf-8"));
	} catch {
		return null;
	}

	if (!roundData.needsFollowup) return null;

	return roundData as unknown as RoundData;
}

export function getRelatedParentIdFromGroup<T extends { fileName: string }>(
	group: { rounds: readonly T[] },
	roundEntry: T,
): string | null {
	if (group.rounds.length <= 1) return null;
	const groupRoundIdx = group.rounds.indexOf(roundEntry);
	return groupRoundIdx > 0 ? group.rounds[groupRoundIdx - 1].fileName : null;
}

export interface MessageEndProcessingState {
	accumulatedText: string[];
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
	responseSegments: ResponseSegment[];
}

export function applyMessageEndToState(message: unknown, state: MessageEndProcessingState): void {
	if (!message) return;

	const msg = message as { role?: string; content?: unknown; toolCallId?: string };
	if (msg.role === "user") {
		// User sent something -- don't reset the accumulator, this is a new agent
		// cycle (agent_start will reset it). Keep safe.
		return;
	}

	if (msg.role === "assistant") {
		// Extract text from this assistant message
		const content = msg.content as Array<{ type: string; text?: string }> | undefined;
		if (Array.isArray(content)) {
			for (const block of content) {
				if ((block.type === "text" || block.type === "thinking") && block.text) {
					const text = block.type === "thinking" ? `[thinking] ${block.text} [/thinking]` : block.text;
					state.accumulatedText.push(text);
					state.responseSegments.push({ type: "text", text });
				} else if (block.type === "toolCall") {
					state.toolCallCount++;
					const blockRec = block as Record<string, unknown>;
					const name = blockRec.name as string | undefined;
					const id = blockRec.id as string | undefined;
					if (name && !state.toolCallNames.includes(name)) {
						state.toolCallNames.push(name);
					}
					if (id && name) {
						const detail: ToolCallDetail = {
							index: state.toolCalls.length,
							id,
							name,
							arguments: JSON.stringify(blockRec.arguments ?? {}),
							result_summary: "",
						};
						state.toolCalls.push(detail);
					}
					// Record this tool call's position in the response stream
					state.responseSegments.push({ type: "toolCall", toolCallIndex: state.toolCalls.length - 1 });
				}
			}
		}
		return;
	}

	if (msg.role === "toolResult") {
		// Pair tool results with their calls.
		// Prefer ID-based matching (precise, deterministic) and fall back to
		// backward sequential scan for legacy round data that lacks toolCallId.
		const toolCallId = msg.toolCallId;
		if (toolCallId) {
			const resultContent = msg.content as Array<{ type: string; text?: string }> | undefined;
			const resultText = resultContent ? extractText(resultContent) : "";

			// Try ID-based match first
			let target = state.toolCalls.find((tc) => tc.id === toolCallId && tc.result_summary === "");

			// Fallback: backward sequential scan (legacy round data without IDs)
			if (!target) {
				for (let i = state.toolCalls.length - 1; i >= 0; i--) {
					if (state.toolCalls[i].result_summary === "") {
						target = state.toolCalls[i];
						break;
					}
				}
			}

			if (target) {
				target.result_summary = resultText.slice(0, 300);
				target.result_full = resultText;
				target.result_truncated = false;
			}
		}
	}
}
