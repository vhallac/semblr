import { estimateTokens } from "./tokens.ts";
import type { MiniMemStore } from "./working-memory.ts";

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

export interface PromptTruncationOptions {
	/** Characters kept from the start of the prompt. Non-positive disables truncation. */
	headChars: number;
	/** Characters kept from the end of the prompt. Non-positive disables truncation. */
	tailChars: number;
}

/** Default injection clamp: keep ~400 chars of head+tail per list entry (issue #106). */
export const DEFAULT_PROMPT_TRUNCATION: PromptTruncationOptions = {
	headChars: 260,
	tailChars: 140,
};

/**
 * Clamp a user prompt for list injection: keep head and tail (first/last
 * lines carry intent), elide the middle with a marker. Prompts at or below
 * the head+tail budget pass through unchanged. When a window can snap to a
 * line boundary inside it, complete lines are preferred over partial ones.
 */
export function truncateUserPrompt(
	prompt: string,
	options: PromptTruncationOptions = DEFAULT_PROMPT_TRUNCATION,
): string {
	const { headChars, tailChars } = options;
	if (headChars <= 0 || tailChars <= 0) return prompt;
	if (prompt.length <= headChars + tailChars) return prompt;

	let head = prompt.slice(0, headChars);
	let tail = prompt.slice(prompt.length - tailChars);

	const headBreak = head.lastIndexOf("\n");
	if (headBreak > 0) head = head.slice(0, headBreak);
	const tailBreak = tail.indexOf("\n");
	if (tailBreak !== -1 && tailBreak < tail.length - 1) tail = tail.slice(tailBreak + 1);

	const elided = prompt.length - head.length - tail.length;
	return `${head}\n… [${elided} chars elided] …\n${tail}`;
}

export function formatRoundEntry(
	idx: number,
	fileName: string,
	score: string,
	toolSummary: string,
	userPrompt: string,
	sizeStr?: string,
	truncation: PromptTruncationOptions = DEFAULT_PROMPT_TRUNCATION,
): string[] {
	const promptLines = truncateUserPrompt(userPrompt, truncation)
		.split("\n")
		.map((line, i) => (i === 0 ? `  user: ${line}` : `  ${line}`));
	const sizePart = sizeStr ? ` | ${sizeStr}` : "";
	return [`${idx}. ${fileName} [${score} | ${toolSummary}${sizePart}]:`, ...promptLines, "  ---"];
}

export function formatGroupedRoundEntry(
	index: number,
	fileName: string,
	toolSummary: string,
	userPrompt: string,
	sizeStr?: string,
	truncation: PromptTruncationOptions = DEFAULT_PROMPT_TRUNCATION,
): string[] {
	const promptLines = truncateUserPrompt(userPrompt, truncation)
		.split("\n")
		.map((line, i) => (i === 0 ? `  user: ${line}` : `  ${line}`));
	const sizePart = sizeStr ? ` | ${sizeStr}` : "";
	return [`- [index: ${index}] ${fileName} [n/a | ${toolSummary}${sizePart}]:`, ...promptLines, "  ---"];
}

/** Default recency list entry cap (issue #106): hard bound across all groups. */
export const DEFAULT_MAX_RECENCY_ENTRIES = 20;

/** Static header for the recency list section (issue #106: counted against the injection budget). */
export const RECENCY_LIST_HEADER = `--- RECENCY LIST (current session, by topic) ---
These rounds have n/a scores because they are presented by recency — they form
the immediate conversational context from this session.

IMPORTANT: This list shows ONLY the user's questions from past rounds.
You do NOT have the assistant responses or tool results unless you expand a
round. If you answer based on these prompts alone, you are hallucinating.

The groups below are recent messages that are likely to be related to the same
topic. Lower numbered indices in groups are more recent conversations.

Use this list when the current prompt ...:
- ... asks about past work, decisions, code, or findings from earlier in this
  session
- ... is unusually short or lacks clear context/goals/outputs
- ... uses references with no clear antecedent in the causal chain ("that fix",
  "the plan", "where we left off")
- ... asks you to remember, verify, continue, or build upon prior work
- ... is ambiguous: lacks proper context or references, and seems to assume
  knowledge was established

When this happens:
1. Scan the group topics and prompts for relevance to the current prompt.
   Prefer the most recent entry (lowest index) in the most related group.
2. If a round looks relevant, expand ONLY that round via get_round_details.
3. Stop as soon as the expanded round gives you enough context to answer.
4. If no round looks relevant but the query clearly needs past context,
   use search_interactions.

When NOT to expand:
- The query is fully self-contained (clear context, goals, and outputs present).
- The prompts in the context already provides sufficient information.

Rule: When in doubt, expand. A verification tool call is cheaper than a wrong
answer.`;

/**
 * Build the recency list in two phases (issue #107 F2):
 *
 * Phase 1 — selection walks the causal chain newest-first and retains rounds
 * under the entry cap and token budget by GLOBAL chronology, never by topic
 * group: with topics interleaved in the chain (A1,B1,A2,B2), a 2-entry cap
 * keeps B2 and A2 — a group-major walk would keep B2 and B1, discarding A2
 * although it is newer than B1. Rounds that never reached a topic group
 * (agent_end embedding failed, so assignToGroup never ran) stay visible: they
 * render as singleton groups keyed by file name.
 *
 * Phase 2 — renders the retained subset grouped by topic. Groups emit in the
 * order their newest retained round was selected (i.e. global recency order),
 * rounds within a group render newest-first, and a group header renders — and
 * is charged to the budget — exactly when the group keeps at least one
 * retained round. A group's first-selected round is the round that leads it in
 * the render, so the header charge is known during selection and the
 * accounting matches the emitted output exactly (issue #106: what is charged
 * is what is injected).
 *
 * The most recent round (kept === 0) is always kept: the recency list is the
 * always-on causal context and must never be emptied by a tight budget — the
 * budget bounds growth, not existence.
 */
export function buildGroupedRecencyList<T extends ContextChainEntry>(
	groups: Array<ContextRoundGroup<T>>,
	causalChain: T[],
	getRoundSize: (fileName: string) => string | null = () => null,
	truncation: PromptTruncationOptions = DEFAULT_PROMPT_TRUNCATION,
	options: {
		/** Hard cap on entries across all groups (default 20, issue #106). `0` disables the list. */
		maxEntries?: number;
		/** Total token budget for the whole list (header + group headers + entries). Non-positive disables the bound. */
		budgetTokens?: number;
		estimateTokensFn?: (text: string) => number;
	} = {},
): string | null {
	const maxEntries = options.maxEntries ?? DEFAULT_MAX_RECENCY_ENTRIES;
	if (causalChain.length === 0 || maxEntries <= 0) return null;
	const budgetTokens = options.budgetTokens ?? 0;
	const bounded = budgetTokens > 0;
	const estimateTokensFn = options.estimateTokensFn ?? estimateTokens;

	// Map each causal-chain entry to its topic group by object identity (the
	// same entry object is pushed into a group by assignToGroup at agent_end).
	// Chain entries found in no group have no topic — they render as singleton
	// groups keyed by file name.
	const entryToGroup = new Map<T, number>();
	for (let gi = 0; gi < groups.length; gi++) {
		for (const round of groups[gi].rounds) {
			if (!entryToGroup.has(round)) entryToGroup.set(round, gi);
		}
	}

	// Groups in render order: a group is created the first time one of its
	// rounds is retained, led by that round (its newest retained member), and
	// holds its retained entry lines. Groups whose rounds are all dropped by
	// the cap or budget are never created — no stray headers.
	const orderedGroups: Array<{ headerLines: string[]; items: string[][] }> = [];
	const groupByKey = new Map<string, { headerLines: string[]; items: string[][] }>();

	let usedTokens = bounded ? estimateTokensFn(RECENCY_LIST_HEADER) : 0;
	let kept = 0;
	for (let i = causalChain.length - 1; i >= 0 && kept < maxEntries; i--) {
		const entry = causalChain[i];
		const sizeStr = getRoundSize(entry.fileName) ?? undefined;
		const entryLines = formatGroupedRoundEntry(
			kept + 1,
			entry.fileName,
			entry.toolSummary,
			entry.userPrompt,
			sizeStr,
			truncation,
		);
		const topicGroupIdx = entryToGroup.get(entry);
		const groupKey = topicGroupIdx === undefined ? `u:${entry.fileName}` : `t:${topicGroupIdx}`;

		// A group first encountered in this walk renders as the next group; its
		// header block (separator + heading) is charged together with its leading
		// entry, so cost and output cannot drift (issue #106). The block is only
		// committed when the leading entry is retained, so a group emptied by the
		// cap or budget never leaves a stray header.
		const headerLines: string[] = [];
		if (!groupByKey.has(groupKey)) {
			if (orderedGroups.length > 0) headerLines.push("", "---", "");
			headerLines.push(`**Group ${orderedGroups.length + 1}**`, "");
		}

		let entryTokens = 0;
		if (bounded) {
			entryTokens = estimateTokensFn(entryLines.join("\n"));
			if (headerLines.length > 0) entryTokens += estimateTokensFn(headerLines.join("\n"));
			// kept === 0 is always kept (see doc comment); later rounds must fit,
			// and the walk stops at the first entry that does not — what remains
			// is strictly the most recent context.
			if (kept > 0 && usedTokens + entryTokens > budgetTokens) break;
		}

		usedTokens += entryTokens;
		let group = groupByKey.get(groupKey);
		if (!group) {
			group = { headerLines, items: [] };
			orderedGroups.push(group);
			groupByKey.set(groupKey, group);
		}
		group.items.push(entryLines);
		kept++;
	}

	const lines: string[] = [];
	lines.push(RECENCY_LIST_HEADER);
	lines.push("");
	for (const group of orderedGroups) {
		lines.push(...group.headerLines);
		for (const entryLines of group.items) lines.push(...entryLines);
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

/** Static header for the relevance list section (issue #106: counted against the injection budget). */
export const RELEVANCE_LIST_HEADER = `--- RELEVANCE LIST (all sessions, by hybrid relevance) ---
These rounds have numeric similarity scores (0.0–1.0). Higher = stronger
match. They combine semantic vector similarity with exact keyword matching.
They come from ALL past sessions, not just the current one.

The extension has pre-run a hybrid search against your prompt. The results
are below. If something here rings a bell, expand it via get_round_details.
If nothing rings a bell, ignore this list — it's a pre-filter, not a map.

Use this list when the prompt asks about past work, decisions, or findings
from prior sessions, or requires cross-session continuity (same project,
recurring topic, long-running task). If nothing here matches but the query
clearly needs past context, use search_interactions.`;

/**
 * Render one relevance-list entry. Shared by buildRelevanceList (rendering)
 * and selectContextRounds (injection-cost accounting) so both stay in
 * lockstep: what is charged to the budget is exactly what is injected.
 */
export function buildRelevanceEntry(
	idx: number,
	round: RelevanceRound,
	truncation: PromptTruncationOptions = DEFAULT_PROMPT_TRUNCATION,
	sizeStr?: string,
): string[] {
	const toolCount = round.data.toolCallCount ?? 0;
	let toolSummary = `${toolCount} tools`;
	if (round.data.toolCalls && round.data.toolCalls.length > 0) {
		toolSummary = buildToolSummary(round.data.toolCalls, toolCount);
	}
	return formatRoundEntry(
		idx,
		round.fileName,
		round.bestScore.toFixed(2),
		toolSummary,
		round.data.userPrompt,
		sizeStr,
		truncation,
	);
}

export function buildRelevanceList(
	rounds: RelevanceRound[],
	getRoundSize: (fileName: string) => string | null = () => null,
	truncation: PromptTruncationOptions = DEFAULT_PROMPT_TRUNCATION,
): string | null {
	if (rounds.length === 0) return null;
	const lines: string[] = [];
	lines.push(RELEVANCE_LIST_HEADER);
	lines.push("");

	let idx = 0;
	for (const round of rounds) {
		idx++;
		const sizeStr = getRoundSize(round.fileName) ?? undefined;
		lines.push(...buildRelevanceEntry(idx, round, truncation, sizeStr));
	}
	return lines.join("\n");
}

export function buildContextPreamble(hasRecency: boolean, hasRelevance: boolean): string | null {
	if (!hasRecency && !hasRelevance) return null;
	return `[CONTEXT BUILDING REFERENCES]
The lists below show past conversation rounds. Each entry contains only the user prompt — responses and tool calls are collapsed.
Use get_round_details("hash.json") to expand a round's full conversation.
Use get_tool_details("hash.json", N) to inspect tool call N within a round.

Format: [index: N] hash.json [score | N tools | size]: followed by the user
prompt (indented). Very long prompts are elided mid-section — use get_round_details
for the full text.

These tools fill in what the context summaries leave out — use them to expand hidden parts of past rounds and build up the full picture. See the SESSION ARCHITECTURE section for details.`;
}

/**
 * Build the WORKING MEMORY section. Returns null when the store is empty.
 * Injected between [SESSION ARCHITECTURE] and [CONTEXT BUILDING REFERENCES].
 *
 * The list shows id + summary only — the LLM uses mini_mem__get to expand.
 */
export function buildWorkingMemorySection(store: MiniMemStore): string | null {
	if (store.slots.length === 0) return null;
	const lines: string[] = [
		"[WORKING MEMORY]",
		"The following list is the id and summary of working memory. Use mini_mem__xxx tools to access and manipulate it.",
		"",
	];
	for (const slot of store.slots) {
		lines.push(`- [id: ${slot.id}] ${slot.summary}`);
	}
	return lines.join("\n");
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
Each conversation round starts fresh by default. Continuity across rounds is
NOT automatic — it exists only through semblr's explicit survival mechanisms:

- **Follow-up injection:** \`round_needs_followup\` on your last line pulls the
  full previous round into the next round's context.
- **Checkpoint:** \`semblr_checkpoint\` persists a structured progress summary
  across context-size boundaries.
- **Working Memory:** \`mini_mem__add\` / \`mini_mem__get\` / \`mini_mem__update\` /
  \`mini_mem__delete\` / \`mini_mem__get_and_delete\` provides named slots
  for short-term notes that survive round boundaries within a session.

When a previous round has been injected into your context (via follow-up or
checkpoint), you DO have access to it — trust what you see, not the default.
Likewise, the context lists below are summaries: the full details are hidden.
Use get_round_details, get_tool_details, and search_interactions to expand
those hidden parts and build up the full picture yourself.`;
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
