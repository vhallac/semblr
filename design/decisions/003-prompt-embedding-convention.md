# 003 — Prompt embedding convention

**Status:** accepted

## Context
The `:prompt` index row stores a vector for semantic retrieval. The live path computes a combined vector `concat(promptEmbedding, responseEmbedding)` as `promptEmbedding`. Short prompts or prompts with empty/noise-only cleaned text produce a degenerate prompt vector; storing that alone would degrade retrieval quality.

## Decision
The live `promptEmbedding` convention is the **combined vector** (concatenation of prompt and response embeddings). For short prompts (below the word threshold defined in issue #38's `shouldDropEmbedding` / `RELEVANCE_LIST_MIN_WORDS`), the **response vector** serves as the fallback: the response vector is stored as `promptEmbedding`. This response-vector fallback is the accepted degraded marker — consumers gate on truthiness and treat it as a real vector.

The `:response` index row stores the response vector independently.

## Consequences
- Retrieval quality is maintained for short prompts via the response-vector fallback.
- Three embedding API calls per round in the normal path (prompt + response + combined), not two.
- Healed rounds that lack both rows cannot reconstruct the combined vector without an API call; the `:response` row vector is the degraded marker used in that path.
- Grouping is unsupported for healed rounds that lack the combined vector.