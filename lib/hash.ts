/**
 * lib/hash.ts — Shared hash utilities for both the extension (src/) and CLI scripts (scripts/).
 *
 * Re-exports round content hash computation from the core implementation.
 */

export { computeContentHash, createRoundFilePath, type HashToolCallDetail } from "../src/core/hash.ts";
