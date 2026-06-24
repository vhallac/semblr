/**
 * lib/session-round-extract.ts — Shared extraction/sanitization for imported rounds.
 *
 * Both pi session and Claude Code importers can encounter text blocks that contain
 * a JSON-stringified prompt envelope (system messages, context wrappers, the
 * "[ACTIONABLE PROMPT]" section delimiters, etc.) instead of just the actionable
 * user text. This module provides shared helpers to detect and strip those envelopes.
 */

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/**
 * A text block from a message content array.
 */
export interface TextBlock {
	type: "text";
	text: string;
}

/**
 * Known sections that may appear in a prompt envelope.
 * When extracting the actionable user prompt, we keep only the section
 * labeled "[ACTIONABLE PROMPT]" and filter out the rest.
 */
const KNOWN_SECTION_HEADERS = [
	"[ACTIONABLE PROMPT]",
	"[SESSION ARCHITECTURE]",
	"[CONTEXT BUILDING REFERENCES]",
	"[PREVIOUS ROUND FOLLOW-UP]",
	"[FINAL RESPONSE CONTRACT — REQUIRED]",
	"[ENVIRONMENT]",
];

// ─────────────────────────────────────────────
// Envelope shape detection
// ─────────────────────────────────────────────

/**
 * Check if a string looks like a JSON-stringified message/prompt envelope
 * rather than plain user text.
 *
 * Returns true if the string is valid JSON with a structure that matches
 * known envelope shapes:
 * - Array of message objects (e.g., [{ role: "system", ... }, { role: "user", ... }])
 * - Object with "role" property (a single message)
 * - Object with "messages" property (a conversation)
 * - Object with "content" property where content is a string or array of blocks
 */
export function looksLikePromptEnvelope(text: string): boolean {
	const trimmed = text.trim();
	if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return false;

	try {
		const parsed = JSON.parse(trimmed);

		// Array of message-like objects
		if (Array.isArray(parsed)) {
			return parsed.length > 0 && parsed.every((item) => isMessageLike(item));
		}

		if (typeof parsed !== "object" || parsed === null) return false;

		// Single message object: { role: "...", content: ... }
		if (typeof parsed.role === "string") return true;

		// Conversation wrapper: { messages: [...] }
		if (Array.isArray(parsed.messages)) {
			return parsed.messages.length > 0 && parsed.messages.every((item: unknown) => isMessageLike(item));
		}

		// Content wrapper: { content: string | array }
		if (typeof parsed.content === "string" || Array.isArray(parsed.content)) return true;

		return false;
	} catch {
		return false;
	}
}

function isMessageLike(value: unknown): boolean {
	if (typeof value !== "object" || value === null) return false;
	const obj = value as Record<string, unknown>;
	return typeof obj.role === "string" || typeof obj.content !== "undefined";
}

// ─────────────────────────────────────────────
// Section extraction
// ─────────────────────────────────────────────

/**
 * Detect if unparsed text (non-JSON) contains prompt envelope section markers
 * like "[ACTIONABLE PROMPT]".
 */
export function hasPromptSectionMarkers(text: string): boolean {
	return KNOWN_SECTION_HEADERS.some((header) => text.includes(header));
}

/**
 * Extract the content of a specific named section from text.
 * Returns null if the section is not found.
 *
 * Sections are delimited by lines like:
 *   [SECTION NAME]
 *   content
 *   [NEXT SECTION]
 */
export function extractSection(text: string, sectionHeader: string): string | null {
	const pattern = new RegExp(`^\\s*\\[?\\s*${escapeRegex(sectionHeader.replace(/[[\]]/g, ""))}\\s*\\]?\\s*$`, "m");
	const match = pattern.exec(text);
	if (!match) return null;

	const afterHeader = text.slice(match.index + match[0].length).trimStart();
	// Everything until the next section header or end of string
	const nextSection = KNOWN_SECTION_HEADERS.filter((h) => h !== sectionHeader)
		.map((h) => new RegExp(`^\\s*\\[?\\s*${escapeRegex(h.replace(/[[\]]/g, ""))}\\s*\\]?\\s*$`, "m"))
		.find((r) => r.exec(afterHeader));

	if (nextSection) {
		const nextMatch = nextSection.exec(afterHeader);
		if (nextMatch) return afterHeader.slice(0, nextMatch.index).trim();
	}

	return afterHeader.trim();
}

function escapeRegex(str: string): string {
	return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ─────────────────────────────────────────────
// Plain text extraction from message content
// ─────────────────────────────────────────────

/**
 * Join text blocks from a content array, filtering out non-text blocks.
 *
 * This is a shared low-level primitive used by both pi and Claude paths.
 * It does NOT do any envelope sanitization — that is applied later via
 * extractUserPromptText().
 */
export function joinTextBlocks(content: Array<{ type?: string; text?: unknown }>): string {
	if (!content || !Array.isArray(content)) return "";
	return content
		.filter((c): c is TextBlock => c.type === "text" && typeof c.text === "string")
		.map((c) => c.text)
		.join(" ")
		.trim();
}

// ─────────────────────────────────────────────
// User prompt sanitization
// ─────────────────────────────────────────────

/**
 * Extract a clean, actionable user prompt from raw text.
 *
 * Handles three cases:
 * 1. **Plain text** — returned as-is (trimmed).
 * 2. **JSON-stringified prompt envelope** — parsed and distilled:
 *    - Array of messages: take the last user message's text
 *    - Single message object with "role": extract content text
 *    - Conversation with "messages": take the last user message's text
 * 3. **Section-marked text** (with [ACTIONABLE PROMPT]) — extract that section.
 *
 * Returns the distilled user prompt text, or the original trimmed text if
 * no recognizable envelope is found.
 */
export function extractUserPromptText(rawText: string): string {
	const trimmed = rawText.trim();
	if (!trimmed) return trimmed;

	// Case 1: Plain text — not an envelope
	if (!looksLikePromptEnvelope(trimmed)) {
		// Check for section-marked text
		if (hasPromptSectionMarkers(trimmed)) {
			const actionable = extractSection(trimmed, "[ACTIONABLE PROMPT]");
			if (actionable) return actionable;
		}
		return trimmed;
	}

	// Case 2: JSON-stringified envelope
	try {
		const parsed = JSON.parse(trimmed);

		// Array of messages: find the last user message
		if (Array.isArray(parsed)) {
			const lastUserMessage = findLastUserMessage(parsed);
			if (lastUserMessage) return extractContentText(lastUserMessage.content);
			return trimmed;
		}

		if (typeof parsed !== "object" || parsed === null) return trimmed;

		// Object with "messages" key
		if (Array.isArray(parsed.messages)) {
			const lastUserMessage = findLastUserMessage(parsed.messages);
			if (lastUserMessage) return extractContentText(lastUserMessage.content);
			return trimmed;
		}

		// Single message object
		if (typeof parsed.role === "string") {
			return extractContentText(parsed.content);
		}

		// Content wrapper
		if (typeof parsed.content === "string") {
			return parsed.content.trim();
		}
		if (Array.isArray(parsed.content)) {
			// content can be an array of blocks (like [{ type: "text", text: "..." }])
			return extractContentText(parsed.content);
		}

		return trimmed;
	} catch {
		return trimmed;
	}
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

interface MessageLike {
	role?: string;
	content?: unknown;
}

function findLastUserMessage(messages: unknown[]): MessageLike | null {
	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i] as MessageLike | undefined;
		if (msg && msg.role === "user") return msg;
	}
	return null;
}

/**
 * Extract text content from a message's "content" field, which can be
 * a string or an array of content blocks.
 */
function extractContentText(content: unknown): string {
	if (typeof content === "string") return content.trim();
	if (Array.isArray(content)) {
		return joinTextBlocks(content as Array<{ type?: string; text?: unknown }>);
	}
	return "";
}
