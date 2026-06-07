/**
 * lib/embed.ts — Shared embedding utilities for both the extension (src/) and CLI scripts (scripts/).
 *
 * Re-exports embedding client and vector operations.
 */

export {
	type ApiKeyContext,
	type ApiKeyLookupDeps,
	EMBEDDING_MODEL,
	type EmbedTextDeps,
	embedText,
	getApiKey,
} from "./embedding-client.ts";
export { cosineSimilarity, normalize } from "./vector.ts";
