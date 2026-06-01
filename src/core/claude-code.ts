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

export function textFromClaudeContent(content: unknown, opts: { includeToolResults?: boolean } = {}): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const block of content as Array<Record<string, unknown>>) {
		if (block.type === "text" && typeof block.text === "string") parts.push(block.text);
		if (opts.includeToolResults && block.type === "tool_result") {
			const c = block.content;
			if (typeof c === "string") parts.push(c);
			else if (Array.isArray(c)) parts.push(textFromClaudeContent(c, { includeToolResults: true }));
		}
	}
	return parts.join("\n").trim();
}

export function isRealClaudeUserPrompt(entry: Record<string, any>): boolean {
	if (entry.type !== "user") return false;
	const content = entry.message?.content;
	if (typeof content === "string") return content.trim().length > 0;
	if (!Array.isArray(content)) return false;
	return content.some((b: any) => b?.type === "text" && typeof b.text === "string" && b.text.trim());
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
	const entries = raw
		.split("\n")
		.filter(Boolean)
		.map((line, i) => {
			try {
				return JSON.parse(line) as Record<string, any>;
			} catch (e) {
				throw new Error(`${options.filePath}:${i + 1}: invalid JSON: ${(e as Error).message}`);
			}
		});

	const rounds: ClaudeRound[] = [];
	let currentUser: Record<string, any> | null = null;
	let responseParts: string[] = [];
	let responseSegments: ClaudeResponseSegment[] = [];
	let toolCalls: ClaudeToolCallDetail[] = [];
	let toolCallNames: string[] = [];
	let pendingById = new Map<string, ClaudeToolCallDetail>();
	let roundIndex = 0;
	let responseEndTimestamp = 0;

	function flush() {
		if (!currentUser) return;
		const userPrompt = textFromClaudeContent(currentUser.message?.content);
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
			userTimestamp: Date.parse(currentUser.timestamp) || 0,
			responseEndTimestamp: responseEndTimestamp || now(),
			turnIndex: roundIndex,
			sessionLabel,
			claudeSessionId: currentUser.sessionId,
			cwd,
			gitBranch: currentUser.gitBranch,
			toolCallCount: toolCalls.length,
			toolCallNames: [...new Set(toolCallNames)],
			toolCalls,
		};
		rounds.push(round);
		roundIndex++;
	}

	function resetFor(entry: Record<string, any>) {
		currentUser = entry;
		responseParts = [];
		responseSegments = [];
		toolCalls = [];
		toolCallNames = [];
		pendingById = new Map();
		responseEndTimestamp = Date.parse(entry.timestamp) || 0;
	}

	for (const entry of entries) {
		if (!options.includeSidechains && entry.isSidechain) continue;

		if (isRealClaudeUserPrompt(entry)) {
			flush();
			resetFor(entry);
			continue;
		}

		if (!currentUser) continue;

		if (entry.type === "assistant" && Array.isArray(entry.message?.content)) {
			responseEndTimestamp = Date.parse(entry.timestamp) || responseEndTimestamp;
			for (const block of entry.message.content as Array<Record<string, any>>) {
				if (block.type === "text" && typeof block.text === "string" && block.text.trim()) {
					responseParts.push(block.text);
					responseSegments.push({ type: "text", text: block.text });
				} else if (block.type === "tool_use") {
					const detail: ClaudeToolCallDetail = {
						index: toolCalls.length,
						name: typeof block.name === "string" ? block.name : "unknown",
						arguments: JSON.stringify(block.input ?? {}),
						result_summary: "",
					};
					toolCalls.push(detail);
					toolCallNames.push(detail.name);
					if (typeof block.id === "string") pendingById.set(block.id, detail);
					responseSegments.push({ type: "toolCall", toolCallIndex: detail.index });
				}
			}
		} else if (entry.type === "user" && Array.isArray(entry.message?.content)) {
			for (const block of entry.message.content as Array<Record<string, any>>) {
				if (block.type !== "tool_result") continue;
				const id = typeof block.tool_use_id === "string" ? block.tool_use_id : undefined;
				const detail = id ? pendingById.get(id) : [...toolCalls].reverse().find((tc) => !tc.result_summary);
				if (detail)
					detail.result_summary = textFromClaudeContent([block], { includeToolResults: true }).slice(0, 300);
			}
		}
	}

	flush();
	return rounds;
}
