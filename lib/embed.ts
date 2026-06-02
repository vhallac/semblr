/**
 * lib/embed.ts — Shared embedding utilities for both the extension (src/) and CLI scripts (scripts/).
 *
 * Re-exports embedding client and vector operations from the core implementations.
 */

export {
	type ApiKeyContext,
	type ApiKeyLookupDeps,
	type EmbedTextDeps,
	embedText,
	getApiKey,
} from "../src/core/embedding-client.ts";
export { cosineSimilarity, normalize } from "../src/core/vector.ts";
