/**
 * lib/embed.ts — Shared embedding utilities for both the extension (src/) and CLI scripts (scripts/).
 *
 * Re-exports embedding client and vector operations.
 */

export {
	type ApiKeyContext,
	type ApiKeyLookupDeps,
	createEmbeddingModelRegistry,
	EMBEDDING_MODEL,
	type EmbeddingClientConfig,
	type EmbeddingModelLike,
	type EmbeddingModelRegistry,
	type EmbedTextDeps,
	embedText,
	getApiKey,
	resolveEmbeddingApiUrl,
} from "./embedding-client.ts";
export { cosineSimilarity, normalize } from "./vector.ts";
