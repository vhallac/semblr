import { countWordsInMessageContent, shouldDropEmbedding } from "./context-messages.ts";
import {
	buildAgentEndEmbeddingTexts,
	buildPromptEmbeddingInput,
	DEFAULT_PROMPT_NOISE_CLEANUP,
	type PromptNoiseOptions,
} from "./round-capture.ts";
import { normalize } from "./vector.ts";

/**
 * Shared embedding-policy core for both round-embedding call sites (F4/F6,
 * PR !131): the live agent_end path and the recovery path
 * (`embedRecoveredRounds`). Owning the policy here — short-prompt drop,
 * prompt cleanup and hash stamping, response clipping, index row labels, the
 * promptEmbedding scale convention, and the optional checkpoint `:summary`
 * row — makes live and recovered embeddings parity by construction rather
 * than by mirroring.
 *
 * Call-site extras stay outside this core: the live prompt-vec stash (passed
 * in as `embedPrompt`), live grouping and follow-up handling, and the
 * recovery-only `hasIndexRow` idempotence guard.
 */
export interface EmbedRoundDeps {
	/** Embed arbitrary text. Used for response, combined, and summary inputs. */
	embed: (text: string) => Promise<number[]>;
	/**
	 * Embed the prompt input. Defaults to `embed`. The live call site uses
	 * this hook to reuse its hash-gated stashed prompt vector instead of
	 * paying for a second embedding of the same input.
	 */
	embedPrompt?: (text: string, hash: string) => Promise<number[]>;
	/** Append one labeled row to the vector index; `hash` stamps :prompt rows. */
	appendIndexRow: (label: string, vec: number[], hash?: string) => void;
	/** Optional label guard (recovery): skip appends for rows already present. */
	hasIndexRow?: (label: string) => boolean;
	/** Store the round's promptEmbedding vector on the round file. */
	writeRoundEmbedding: (fileName: string, vec: number[]) => void;
}

export interface EmbedRoundRequest {
	fileName: string;
	userPrompt: string;
	responseText: string;
	/** Optional pre-rendered checkpoint summary text; embeds as a `:summary` row. */
	checkpointSummaryText?: string | null;
	maxResponseBytes?: number;
	promptNoiseOptions?: PromptNoiseOptions;
	promptMaxTokens?: number;
}

export interface EmbedRoundResult {
	/** True when the short-prompt policy dropped the prompt embedding. */
	promptDropped: boolean;
	/** The vector stored as the round's promptEmbedding, in the live convention. */
	promptEmbedding: number[];
}

/**
 * The index-row suffixes `embedRound` reproduces for a round with this prompt
 * and summary. A reindex replaces exactly these rows; any other suffix on disk
 * (a legacy `:round`/bare row, or a `:summary` whose source is gone) is an
 * orphan that must be preserved, not traded away.
 *
 * `:prompt` is reproduced only when the prompt survives the short-prompt drop
 * (F1, PR !141 review): a short prompt yields a `:response` row and no
 * `:prompt` row, so a stale `:prompt` row on such a round is an orphan —
 * treating it as reproducible would re-flag the round on every run forever,
 * because no reindex can ever replace it. `:response` is always reproduced;
 * `:summary` only when a summary text is present.
 */
export function reproducedIndexSuffixes(userPrompt: string, hasSummary: boolean): Set<string> {
	const suffixes = new Set<string>([":response"]);
	if (!shouldDropEmbedding(countWordsInMessageContent(userPrompt))) suffixes.add(":prompt");
	if (hasSummary) suffixes.add(":summary");
	return suffixes;
}

/**
 * Suffixes whose staleness can force a model-change reindex. `:summary` is
 * excluded even though a reindex reproduces it: it is auxiliary and
 * content-dependent, so a `:summary` row whose source is gone would re-flag
 * the round forever (the #140 D4 non-convergence). A reindex triggered by a
 * forcing row still refreshes a reproduced `:summary` row; it just never
 * triggers one. `:round`/bare rows are never rewritten by a reindex, so they
 * never force one either.
 */
export const FORCING_INDEX_SUFFIXES: ReadonlySet<string> = new Set([":prompt", ":response"]);

/**
 * The suffixes whose staleness *may* force a reindex for a round with this
 * prompt and summary: `reproducedIndexSuffixes` restricted to
 * `FORCING_INDEX_SUFFIXES`. A short prompt's `:prompt` row is non-reproducible
 * (F1), so a stale one is an orphan rather than a reindex trigger.
 */
export function forcingReproducibleSuffixes(userPrompt: string, hasSummary: boolean): Set<string> {
	return new Set([...reproducedIndexSuffixes(userPrompt, hasSummary)].filter((s) => FORCING_INDEX_SUFFIXES.has(s)));
}

/** How a round's index rows classify under one model. `covered` is the shared
 * "this round is indexed" answer: it has at least one current-model forcing row
 * and no stale forcing row. It is the exact negation of the `just index` sweep's
 * enqueue condition, so the startup pending count and the sweep agree by
 * construction (issue #140 D1 / F2, PR !141 review). */
export interface RoundCoverage {
	hasReproducibleRow: boolean;
	hasStaleReproducibleRow: boolean;
	covered: boolean;
}

/**
 * Classify `entries` (all index rows for one round) against `currentModel`,
 * considering only `forcingSuffixes`. Legacy rows without a model are current
 * (issue #62), matching every other index reader.
 */
export function classifyRoundCoverage(
	entries: readonly { filePath: string; model?: string }[],
	currentModel: string,
	forcingSuffixes: ReadonlySet<string>,
	rowSuffix: (filePath: string) => string,
): RoundCoverage {
	let hasReproducibleRow = false;
	let hasStaleReproducibleRow = false;
	for (const entry of entries) {
		if (!forcingSuffixes.has(rowSuffix(entry.filePath))) continue;
		hasReproducibleRow = true;
		if (entry.model !== undefined && entry.model !== currentModel) hasStaleReproducibleRow = true;
	}
	return {
		hasReproducibleRow,
		hasStaleReproducibleRow,
		covered: hasReproducibleRow && !hasStaleReproducibleRow,
	};
}

/**
 * Embed one round under the single shared policy.
 *
 * Short-prompt drop (shouldDropEmbedding over the raw prompt's word count):
 * no :prompt row, no prompt embedding; the response is embedded alone and
 * its normalized vector becomes the round's promptEmbedding. Otherwise the
 * prompt goes through buildPromptEmbeddingInput (cleanup + clip + hash
 * stamp), the response is clipped to the configured budget, and the raw
 * combined vector becomes the round's promptEmbedding — exactly the live
 * agent_end convention.
 */
export async function embedRound(request: EmbedRoundRequest, deps: EmbedRoundDeps): Promise<EmbedRoundResult> {
	const {
		fileName,
		userPrompt,
		responseText,
		checkpointSummaryText,
		maxResponseBytes,
		promptNoiseOptions = DEFAULT_PROMPT_NOISE_CLEANUP,
		promptMaxTokens,
	} = request;
	const skipPrompt = shouldDropEmbedding(countWordsInMessageContent(userPrompt));

	if (skipPrompt) {
		const { clippedResponse } = buildAgentEndEmbeddingTexts("", responseText, maxResponseBytes);
		const responseVec = normalize(await deps.embed(clippedResponse));
		const responseLabel = `${fileName}:response`;
		if (!deps.hasIndexRow?.(responseLabel)) {
			deps.appendIndexRow(responseLabel, responseVec);
		}
		if (checkpointSummaryText) {
			const summaryVec = normalize(await deps.embed(checkpointSummaryText));
			const summaryLabel = `${fileName}:summary`;
			if (!deps.hasIndexRow?.(summaryLabel)) {
				deps.appendIndexRow(summaryLabel, summaryVec);
			}
		}
		deps.writeRoundEmbedding(fileName, responseVec);
		return { promptDropped: true, promptEmbedding: responseVec };
	}

	const { text: promptInput, hash: promptInputHash } = buildPromptEmbeddingInput(
		userPrompt,
		promptNoiseOptions,
		promptMaxTokens,
	);
	const { clippedResponse, combinedText } = buildAgentEndEmbeddingTexts(promptInput, responseText, maxResponseBytes);

	const promptVec = normalize(
		await (deps.embedPrompt ? deps.embedPrompt(promptInput, promptInputHash) : deps.embed(promptInput)),
	);
	const [responseVec, combinedVec] = await Promise.all([deps.embed(clippedResponse), deps.embed(combinedText)]);

	const promptLabel = `${fileName}:prompt`;
	if (!deps.hasIndexRow?.(promptLabel)) {
		deps.appendIndexRow(promptLabel, promptVec, promptInputHash);
	}
	const responseLabel = `${fileName}:response`;
	if (!deps.hasIndexRow?.(responseLabel)) {
		deps.appendIndexRow(responseLabel, normalize(responseVec));
	}
	if (checkpointSummaryText) {
		const summaryVec = normalize(await deps.embed(checkpointSummaryText));
		const summaryLabel = `${fileName}:summary`;
		if (!deps.hasIndexRow?.(summaryLabel)) {
			deps.appendIndexRow(summaryLabel, summaryVec);
		}
	}
	// Live convention: the round's promptEmbedding is the raw combined vector,
	// not the normalized index rows.
	deps.writeRoundEmbedding(fileName, combinedVec);
	return { promptDropped: false, promptEmbedding: combinedVec };
}
