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
