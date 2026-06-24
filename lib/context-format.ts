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
If nothing rings a bell, ignore this list — it's a pre-filter, not a map.

Use this list when the prompt asks about past work, decisions, or findings
from prior sessions, or requires cross-session continuity (same project,
recurring topic, long-running task). If nothing here matches but the query
clearly needs past context, use search_interactions.`;
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

/**
 * Build routing instructions for the semblr_report_phase tool.
 * Injected into context when multi-model routing is enabled.
 * Teaches the LLM when to report each generation phase.
 */
export function buildRoutingInstructions(): string {
	return `[MULTI-MODEL ROUTING]
This session has multi-model routing enabled. Different models handle different
generation phases. To route your next turn to the right model, call the
\`semblr_report_phase\` tool BEFORE your final response.

Phases and when to report them:
- **exploring**: pulling in external data by reading, searching, exploring.
  No model switch (stays on current model).
- **planning**: formulating a plan of response, structured thinking.
  Routes to a fast distillation model.
- **executing**: implementing a plan, writing code, making edits.
  Routes to a high-capability coding model.
- **stuck**: underspecified task, insufficient data, need creative debugging.
  Routes to a deep-reasoning model.
- **verifying**: execution done, validating output and created files.
  Routes to a thorough verification model.
- **reporting**: done with work, about to deliver final output or summary.
  Routes to a fast, lightweight model for formatting/summarization.

Call \`semblr_report_phase\` with the phase that describes what you will do
NEXT (not what you just did). Call it once per round — the last call before
your final response wins. Model switches happen at turn end (after your final
response), and the original model is restored at round end. This means the
phase-specific model takes effect at the next round boundary — not mid-response.`;
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
