export interface ContextChainEntry {
	fileName: string;
	userPrompt: string;
	responseSequence?: string;
	toolSummary: string;
}

export interface ContextRoundGroup<T extends ContextChainEntry = ContextChainEntry> {
	centroid?: number[];
	rounds: T[];
}

interface ContextToolCallDetail {
	name: string;
	result_summary?: string;
	result_full?: string;
}

interface ContextRoundData {
	userPrompt: string;
	toolCallCount?: number;
	toolCalls?: ContextToolCallDetail[];
}

export interface RelevanceRound {
	fileName: string;
	bestScore: number;
	data: ContextRoundData;
}

export function formatRoundEntry(
	idx: number,
	fileName: string,
	score: string,
	toolSummary: string,
	userPrompt: string,
	sizeStr?: string,
): string[] {
	const promptLines = userPrompt.split("\n").map((line, i) => (i === 0 ? `  user: ${line}` : `  ${line}`));
	const sizePart = sizeStr ? ` | ${sizeStr}` : "";
	return [`${idx}. ${fileName} [${score} | ${toolSummary}${sizePart}]:`, ...promptLines, "  ---"];
}

export function formatGroupedRoundEntry(
	index: number,
	fileName: string,
	toolSummary: string,
	userPrompt: string,
	sizeStr?: string,
): string[] {
	const promptLines = userPrompt.split("\n").map((line, i) => (i === 0 ? `  user: ${line}` : `  ${line}`));
	const sizePart = sizeStr ? ` | ${sizeStr}` : "";
	return [`- [index: ${index}] ${fileName} [n/a | ${toolSummary}${sizePart}]:`, ...promptLines, "  ---"];
}

export function buildGroupedRecencyList<T extends ContextChainEntry>(
	groups: Array<ContextRoundGroup<T>>,
	causalChain: T[],
	getRoundSize: (fileName: string) => string | null = () => null,
): string | null {
	if (groups.length === 0) return null;
	const lines: string[] = [];
	const header = `--- RECENCY LIST (current session, by topic) ---
These rounds have n/a scores because they are presented by recency — they form
the immediate conversational context from this session.

IMPORTANT: This list shows ONLY the user's questions from past rounds.
You do NOT have the assistant responses or tool results unless you expand a
round. If you answer based on these prompts alone, you are hallucinating.

The groups below are recent messages that are likely to be related to the same
topic. Lower numbered indices in groups are more recent conversations.

Use this list when the current prompt ...:
- ... asks about past work, decisions, code, or findings from prior sessions
- ... is unusually short or lacks clear context/goals/outputs
- ... uses references with no clear antecedent in the causal chain ("that fix",
  "the plan", "where we left off")
- ... asks you to remember, verify, continue, or build upon prior work
- ... requires cross-session continuity (same project, recurring topic,
  long-running task)
- ... is ambiguous: lacks proper context or references, and seems to assume
  knowledge was established

When this happens:
1. Scan the list prompts for relevance. Higher score = stronger match.
2. If a round looks relevant, expand ONLY that round via get_round_details.
3. Stop as soon as the expanded round gives you enough context to answer.
4. If no round looks relevant but the query clearly needs past context,
   use search_interactions.

When NOT to expand:
- The query is fully self-contained (clear context, goals, and outputs present).
- The prompts in the context already provides sufficient information.

Rule: When in doubt, expand. A verification tool call is cheaper than a wrong
answer.`;
	lines.push(header);
	lines.push("");

	const sortedGroups = [...groups].sort((a, b) => {
		const aLast = a.rounds[a.rounds.length - 1];
		const bLast = b.rounds[b.rounds.length - 1];
		return causalChain.indexOf(bLast) - causalChain.indexOf(aLast);
	});

	const globalIndices = new Map<T, number>();
	let globalIdx = 0;
	for (const group of sortedGroups) {
		const reversed = [...group.rounds].reverse();
		for (const entry of reversed) {
			globalIdx++;
			globalIndices.set(entry, globalIdx);
		}
	}

	let groupNumber = 0;
	for (const group of sortedGroups) {
		groupNumber++;
		if (groupNumber > 1) {
			lines.push("");
			lines.push("---");
			lines.push("");
		}
		lines.push(`**Group ${groupNumber}**`);
		lines.push("");

		const reversed = [...group.rounds].reverse();
		for (const entry of reversed) {
			const idx = globalIndices.get(entry) ?? 0;
			const sizeStr = getRoundSize(entry.fileName) ?? undefined;
			lines.push(...formatGroupedRoundEntry(idx, entry.fileName, entry.toolSummary, entry.userPrompt, sizeStr));
		}
	}

	return lines.join("\n");
}

export function buildToolSummary(toolCalls: ContextToolCallDetail[], totalCount: number): string {
	// Group tool calls by name, summing sizes
	const grouped = new Map<string, { count: number; totalBytes: number }>();
	for (const tc of toolCalls) {
		const sourceText = tc.result_full ?? tc.result_summary ?? "";
		const bytes = sourceText.length > 0 ? Buffer.byteLength(sourceText, "utf-8") : 0;
		const entry = grouped.get(tc.name);
		if (entry) {
			entry.count++;
			entry.totalBytes += bytes;
		} else {
			grouped.set(tc.name, { count: 1, totalBytes: bytes });
		}
	}

	// Format as "name×count (size)" with stable ordering
	const parts = Array.from(grouped.entries()).map(([name, info]) =>
		info.totalBytes > 0 ? `${name}×${info.count} (${formatFileSize(info.totalBytes)})` : `${name}×${info.count}`,
	);
	return `${totalCount} tools (${parts.join(", ")})`;
}

export function buildRelevanceList(
	rounds: RelevanceRound[],
	getRoundSize: (fileName: string) => string | null = () => null,
): string | null {
	if (rounds.length === 0) return null;
	const lines: string[] = [];
	const header = `--- RELEVANCE LIST (all sessions, by similarity) ---
These rounds have numeric similarity scores (0.0–1.0). Higher = stronger
semantic match. They come from ALL past sessions, not just the current one.

The extension has pre-run a semantic search against your prompt. The results
are below. If something here rings a bell, expand it via get_round_details.
If nothing rings a bell, ignore this list — it's a pre-filter, not a map.`;
	lines.push(header);
	lines.push("");

	let idx = 0;
	for (const round of rounds) {
		idx++;
		const toolCount = round.data.toolCallCount ?? 0;
		const sizeStr = getRoundSize(round.fileName) ?? undefined;

		let toolSummary = `${toolCount} tools`;
		if (round.data.toolCalls && round.data.toolCalls.length > 0) {
			toolSummary = buildToolSummary(round.data.toolCalls, toolCount);
		}

		lines.push(
			...formatRoundEntry(
				idx,
				round.fileName,
				round.bestScore.toFixed(2),
				toolSummary,
				round.data.userPrompt,
				sizeStr,
			),
		);
	}
	return lines.join("\n");
}

export function buildContextPreamble(hasRecency: boolean, hasRelevance: boolean): string | null {
	if (!hasRecency && !hasRelevance) return null;
	return `[CONTEXT BUILDING REFERENCES]
The lists below show past conversation rounds. Each entry contains only the user prompt — responses and tool calls are collapsed.
Use get_round_details("hash.json") to expand a round's full conversation.
Use get_tool_details("hash.json", N) to inspect tool call N within a round.

Format: [index: N] hash.json [score | N tools | size]: followed by the full user prompt (indented).

These tools exist because you forget everything between rounds. See the SESSION ARCHITECTURE section for details.`;
}

/**
 * Build the SESSION ARCHITECTURE section — informs the LLM about the
 * fundamental constraint that information dies at round boundaries and
 * must be explicitly carried forward via survival mechanisms.
 *
 * Injected unconditionally (even on short-prompt fast path) after the
 * system message and before the context preamble, because this is
 * foundational session knowledge, not prompt-specific tool instruction.
 *
 * Stage 1 covers follow-up injection and checkpoint. Stage 2 (post-#68)
 * adds the working memory bullet.
 */
export function buildSessionArchitecture(): string {
	return `[SESSION ARCHITECTURE]
Each conversation round starts fresh. Your previous responses, tool results, and
reasoning from earlier rounds are NOT visible unless they are explicitly injected
through one of semblr's survival mechanisms:

- **Follow-up injection:** \`round_needs_followup\` on your last line pulls the
  full previous round into the next round's context.
- **Checkpoint:** \`semblr_checkpoint\` persists a structured progress summary
  across context-size boundaries.`;
}

export function formatFileSize(bytes: number): string {
	if (bytes < 1024) return `${Math.round(bytes / 10.24) / 100}KB`;
	if (bytes < 10240) return `${Math.round(bytes / 1024)}KB`;
	if (bytes < 1048576) return `${Math.round(bytes / 1024)}KB`;
	return `${Math.round(bytes / 10485.76) / 100}MB`;
}

/**
 * Build the follow-up injection section for the context.
 * This is injected when the previous round had `needsFollowup: true` and
 * contains the full previous round content so the LLM can see what question
 * was asked.
 */
export function buildFollowUpSectionContent(fileName: string, userPrompt: string, responseSequence: string): string {
	return `--- PREVIOUS ROUND FOLLOW-UP ---
The previous round (${fileName}) was flagged for follow-up. Its full content is included below so you can see what question was asked:

USER PROMPT:
${userPrompt}

ASSISTANT RESPONSE:
${responseSequence}`;
}

/**
 * Build the checkpoint injection section for context.
 * This is injected when a previous round has a `summary` checkpoint
 * (generated by a context-size warning). It presents the structured
 * progress summary so the agent can resume work.
 */
export function buildCheckpointSectionContent(
	fileName: string,
	summary: {
		currentTask: string;
		progressMade: string[];
		currentState: string[];
		nextSteps: string[];
		keyFindings: string[];
	},
): string {
	const lines: string[] = [];
	lines.push(`--- PREVIOUS ROUND CHECKPOINT ---`);
	lines.push(`The previous round (${fileName}) was checkpointed due to context size limits.`);
	lines.push(
		`Below is the progress summary from that round. Use this to understand what was in progress and resume work.`,
	);
	lines.push("");
	lines.push(`## Current Task`);
	lines.push(summary.currentTask);
	if (summary.progressMade.length > 0) {
		lines.push("");
		lines.push("## Progress Made");
		for (const item of summary.progressMade) lines.push(`- ${item}`);
	}
	if (summary.currentState.length > 0) {
		lines.push("");
		lines.push("## Current State");
		for (const item of summary.currentState) lines.push(`- ${item}`);
	}
	if (summary.nextSteps.length > 0) {
		lines.push("");
		lines.push("## Next Steps");
		for (const item of summary.nextSteps) lines.push(`- ${item}`);
	}
	if (summary.keyFindings.length > 0) {
		lines.push("");
		lines.push("## Key Findings / Decisions");
		for (const item of summary.keyFindings) lines.push(`- ${item}`);
	}
	return lines.join("\n");
}

/** Follow-up marker instruction — teaches the model about round_needs_followup. */
export function buildFollowupSection(): string {
	return `If this response requires a user follow-up — such as asking a question, requesting
confirmation, or pausing for user input — add the following line, *exactly* as
shown, as the very last line of your output:

round_needs_followup`;
}

/**
 * Final response contract — always injected immediately before the actionable prompt.
 * Stronger than buildFollowupSection(): applies only to the ACTIONABLE PROMPT,
 * not to quoted examples, historical rounds, or environment metadata.
 */
export function buildFinalResponseContract(): string {
	return `[FINAL RESPONSE CONTRACT — REQUIRED]
Before sending your final answer, check whether your response to the ACTIONABLE PROMPT below asks the user a question, requests confirmation, presents options for the user to choose from, or otherwise pauses for user input.

If yes, the final line of your response MUST be exactly:

round_needs_followup

MUST NOT put any text after that line.
MUST NOT wrap it in a code block.
This contract applies only to your final response to the ACTIONABLE PROMPT, not to quoted examples, historical rounds, context references, or environment metadata.`;
}

export function splitCommandArgs(args: string): string[] {
	const out: string[] = [];
	let current = "";
	let quote: '"' | "'" | null = null;
	let escaping = false;

	for (const ch of args) {
		if (escaping) {
			current += ch;
			escaping = false;
			continue;
		}
		if (ch === "\\") {
			escaping = true;
			continue;
		}
		if (quote) {
			if (ch === quote) quote = null;
			else current += ch;
			continue;
		}
		if (ch === '"' || ch === "'") {
			quote = ch;
			continue;
		}
		if (/\s/.test(ch)) {
			if (current) {
				out.push(current);
				current = "";
			}
			continue;
		}
		current += ch;
	}
	if (escaping) current += "\\";
	if (current) out.push(current);
	return out;
}
