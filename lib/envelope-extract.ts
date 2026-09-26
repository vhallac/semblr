/**
 * Envelope extraction for user prompts (issue #83).
 *
 * A "prompt envelope" is a serialized LLM messages array (system/user/assistant
 * messages) that leaked into a session entry the importers treat as the user's
 * prompt. Two real-world shapes are handled:
 *
 * 1. Bare envelope: the whole prompt text is a JSON-stringified messages array
 *    (or single message object, or `{messages: [...]}`). The user's actual
 *    words live inside the last user message.
 * 2. Fenced envelope: the prompt is (prose around) a fenced code block whose
 *    body is a pretty-printed messages dump. The prose before/after the fence
 *    is the user's text; the fenced body collapses to a compact placeholder.
 *
 * The canonical contract: a round's `userPrompt` is the user's actual words,
 * never the envelope payload. `splitAndExtractPrompt` is idempotent — clean
 * text passes through untouched.
 */

const JSON_PARSE_LENGTH_CAP = 1_000_000;

export interface SplitExtractResult {
	/** The extracted user text (or a placeholder when nothing user-owned remains). */
	userText: string;
	/** Whether an envelope was detected and extracted/collapsed. */
	wasEnvelope: boolean;
}

export const ENVELOPE_PLACEHOLDER_PREFIX = "[MESSAGE_ENVELOPE: ~";

function makePlaceholder(body: string): string {
	return `${ENVELOPE_PLACEHOLDER_PREFIX}${body.length} chars]`;
}

function textFromContent(content: unknown): string {
	if (typeof content === "string") return content.trim();
	if (Array.isArray(content)) {
		return content
			.map((block) => {
				if (block && typeof block === "object" && (block as { type?: string }).type === "text") {
					return typeof (block as { text?: unknown }).text === "string" ? (block as { text: string }).text : "";
				}
				return "";
			})
			.filter(Boolean)
			.join("\n")
			.trim();
	}
	return "";
}

/** Structural check: is this parsed value a messages array (or single message)? */
function isMessageArray(value: unknown): value is Array<{ role?: unknown; content?: unknown }> {
	return (
		Array.isArray(value) &&
		value.length > 0 &&
		value.every((m) => m && typeof m === "object" && typeof (m as { role?: unknown }).role === "string")
	);
}

/** Extract the user's words from a parsed messages value; null if none found. */
function extractUserTextFromMessages(value: unknown): string | null {
	let messages: Array<{ role?: unknown; content?: unknown }>;
	if (isMessageArray(value)) {
		messages = value;
	} else if (value && typeof value === "object") {
		const rec = value as Record<string, unknown>;
		if (Array.isArray(rec.messages) && isMessageArray(rec.messages)) {
			messages = rec.messages;
		} else if (typeof rec.role === "string" && "content" in rec) {
			messages = [rec as { role?: unknown; content?: unknown }];
		} else {
			return null;
		}
	} else {
		return null;
	}

	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (!msg || msg.role !== "user") continue;
		const text = textFromContent(msg.content);
		if (text) return text;
	}
	return null;
}

/** Try to parse the text as an envelope; returns the parsed value or null. */
function tryParseEnvelope(text: string): unknown {
	if (text.length === 0 || text.length > JSON_PARSE_LENGTH_CAP) return null;
	const trimmed = text.trim();
	if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return null;
	try {
		const parsed: unknown = JSON.parse(trimmed);
		if (parsed === null || typeof parsed !== "object") return null;
		return parsed;
	} catch {
		return null;
	}
}

/** Detect a fenced block whose body is a messages envelope; returns match info or null. */
function findFencedEnvelope(text: string): { full: string; body: string; parsed: unknown } | null {
	const fenceRe = /^([ \t]*)(`{3,}|~{3})[^\n]*\n([\s\S]*?)^\1\2[^\n]*$/gm;
	for (const match of text.matchAll(fenceRe)) {
		const body = match[3] ?? "";
		const parsed = tryParseEnvelope(body);
		if (parsed === null) continue;
		if (!isMessageArray(parsed)) {
			const rec = parsed as Record<string, unknown>;
			const hasMessages = Array.isArray(rec.messages) && isMessageArray(rec.messages);
			const isSingleMessage = typeof rec.role === "string" && "content" in rec;
			if (!hasMessages && !isSingleMessage) continue;
		}
		return { full: match[0], body, parsed };
	}
	return null;
}

/**
 * Detect and split a prompt envelope out of `text` (issue #83).
 * Clean text passes through untouched (`wasEnvelope: false`).
 */
export function splitAndExtractPrompt(text: string): SplitExtractResult {
	// Bare envelope: the whole prompt is a serialized messages payload.
	const bare = tryParseEnvelope(text);
	if (bare !== null) {
		const userText = extractUserTextFromMessages(bare);
		if (userText !== null) return { userText, wasEnvelope: true };
		if (isMessageArray(bare)) return { userText: makePlaceholder(text), wasEnvelope: true };
		const rec = bare as Record<string, unknown>;
		if (
			(typeof rec.role === "string" && "content" in rec) ||
			(Array.isArray(rec.messages) && isMessageArray(rec.messages))
		) {
			return { userText: makePlaceholder(text), wasEnvelope: true };
		}
	}

	// Fenced envelope: collapse the fenced dump, keep the surrounding prose.
	const fenced = findFencedEnvelope(text);
	if (fenced) {
		const prose = text.replace(fenced.full, makePlaceholder(fenced.body));
		return { userText: prose, wasEnvelope: true };
	}

	return { userText: text, wasEnvelope: false };
}
