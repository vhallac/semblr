import crypto from "node:crypto";
import * as path from "node:path";

interface ClaudeToolCallDetail {
	index: number;
	name: string;
	arguments: string;
	result_summary: string;
}

interface ClaudeResponseSegment {
	type: "text" | "toolCall";
	text?: string;
	toolCallIndex?: number;
}

export interface ClaudeRound {
	id: string;
	source: "claude-code";
	userPrompt: string;
	responseSequence: string;
	responseSegments: ClaudeResponseSegment[];
	userTimestamp: number;
	responseEndTimestamp: number;
	turnIndex: number;
	sessionLabel: string;
	claudeSessionId?: string;
	cwd?: string;
	gitBranch?: string;
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ClaudeToolCallDetail[];
}

export interface ParseClaudeCodeOptions {
	filePath: string;
	projectsDir: string;
	includeSidechains?: boolean;
	now?: () => number;
}

/**
 * Represents a single entry in a Claude Code JSONL export.
 * Uses Record<string, unknown> to avoid any while supporting arbitrary fields.
 */
type ClaudeEntry = Record<string, unknown>;

/** A block within a Claude content array (e.g., text, tool_use, tool_result). */
type ClaudeContentBlock = Record<string, unknown>;

export function textFromClaudeContent(content: unknown, opts: { includeToolResults?: boolean } = {}): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const rawBlock of content as ClaudeContentBlock[]) {
		if (rawBlock.type === "text" && typeof rawBlock.text === "string") parts.push(rawBlock.text);
		if (opts.includeToolResults && rawBlock.type === "tool_result") {
			const c = rawBlock.content;
			if (typeof c === "string") parts.push(c);
			else if (Array.isArray(c)) parts.push(textFromClaudeContent(c, { includeToolResults: true }));
		}
	}
	return parts.join("\n").trim();
}

export function isRealClaudeUserPrompt(entry: ClaudeEntry): boolean {
	if (entry.type !== "user") return false;
	const msg = entry.message as ClaudeEntry | undefined;
	const msgContent = msg?.content;
	if (typeof msgContent === "string") return msgContent.trim().length > 0;
	if (!Array.isArray(msgContent)) return false;
	return (msgContent as ClaudeContentBlock[]).some(
		(b) => b.type === "text" && typeof b.text === "string" && String(b.text).trim(),
	);
}

function claudeRoundId(round: Pick<ClaudeRound, "userPrompt" | "responseSequence">): string {
	return crypto
		.createHash("md5")
		.update(round.userPrompt + round.responseSequence)
		.digest("hex");
}

export function claudeRoundFileName(round: Pick<ClaudeRound, "userPrompt" | "responseSequence">): string {
	return `${claudeRoundId(round)}.json`;
}

export function parseClaudeCodeJsonl(raw: string, options: ParseClaudeCodeOptions): ClaudeRound[] {
	const now = options.now ?? Date.now;
	const entries: ClaudeEntry[] = raw
		.split("\n")
		.filter(Boolean)
		.map((line, i) => {
			try {
				return JSON.parse(line) as ClaudeEntry;
			} catch (e) {
				throw new Error(`${options.filePath}:${i + 1}: invalid JSON: ${(e as Error).message}`);
			}
		});

	const rounds: ClaudeRound[] = [];
	let currentUser: ClaudeEntry | null = null;
	let responseParts: string[] = [];
	let responseSegments: ClaudeResponseSegment[] = [];
	let toolCalls: ClaudeToolCallDetail[] = [];
	let toolCallNames: string[] = [];
	let pendingById = new Map<string, ClaudeToolCallDetail>();
	let roundIndex = 0;
	let responseEndTimestamp = 0;

	function flush() {
		if (!currentUser) return;
		const userPrompt = textFromClaudeContent((currentUser.message as ClaudeEntry | undefined)?.content);
		const responseSequence = responseParts.join("\n\n").trim();
		if (!userPrompt || responseSequence.length < 20) return;
		const cwd = currentUser.cwd as string | undefined;
		const sessionLabel = path.relative(options.projectsDir, options.filePath) || path.basename(options.filePath);
		const round: ClaudeRound = {
			id: claudeRoundId({ userPrompt, responseSequence }),
			source: "claude-code",
			userPrompt,
			responseSequence,
			responseSegments,
			userTimestamp: Date.parse(currentUser.timestamp as string) || 0,
			responseEndTimestamp: responseEndTimestamp || now(),
			turnIndex: roundIndex,
			sessionLabel,
			claudeSessionId: currentUser.sessionId as string | undefined,
			cwd,
			gitBranch: currentUser.gitBranch as string | undefined,
			toolCallCount: toolCalls.length,
			toolCallNames: [...new Set(toolCallNames)],
			toolCalls,
		};
		rounds.push(round);
		roundIndex++;
	}

	function resetFor(entry: ClaudeEntry) {
		currentUser = entry;
		responseParts = [];
		responseSegments = [];
		toolCalls = [];
		toolCallNames = [];
		pendingById = new Map();
		responseEndTimestamp = Date.parse(entry.timestamp as string) || 0;
	}

	for (const entry of entries) {
		if (!options.includeSidechains && entry.isSidechain) continue;

		if (isRealClaudeUserPrompt(entry)) {
			flush();
			resetFor(entry);
			continue;
		}

		if (!currentUser) continue;

		if (entry.type === "assistant" && Array.isArray((entry.message as ClaudeEntry | undefined)?.content)) {
			responseEndTimestamp = Date.parse(entry.timestamp as string) || responseEndTimestamp;
			for (const rawBlock of (entry.message as ClaudeEntry).content as ClaudeContentBlock[]) {
				if (rawBlock.type === "text" && typeof rawBlock.text === "string" && rawBlock.text.trim()) {
					responseParts.push(rawBlock.text);
					responseSegments.push({ type: "text", text: rawBlock.text });
				} else if (rawBlock.type === "tool_use") {
					const detail: ClaudeToolCallDetail = {
						index: toolCalls.length,
						name: typeof rawBlock.name === "string" ? rawBlock.name : "unknown",
						arguments: JSON.stringify(rawBlock.input ?? {}),
						result_summary: "",
					};
					toolCalls.push(detail);
					toolCallNames.push(detail.name);
					if (typeof rawBlock.id === "string") pendingById.set(rawBlock.id, detail);
					responseSegments.push({ type: "toolCall", toolCallIndex: detail.index });
				}
			}
		} else if (entry.type === "user" && Array.isArray((entry.message as ClaudeEntry | undefined)?.content)) {
			for (const rawBlock of (entry.message as ClaudeEntry).content as ClaudeContentBlock[]) {
				if (rawBlock.type !== "tool_result") continue;
				const id = typeof rawBlock.tool_use_id === "string" ? rawBlock.tool_use_id : undefined;
				const detail = id ? pendingById.get(id) : [...toolCalls].reverse().find((tc) => !tc.result_summary);
				if (detail)
					detail.result_summary = textFromClaudeContent([rawBlock], { includeToolResults: true }).slice(0, 300);
			}
		}
	}

	flush();
	return rounds;
}
