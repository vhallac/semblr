# 001 — Prompt embedding input derivation

**Status:** accepted

## Context
Semblr embeds prompt text at multiple sites: the agent hook (semantic query for context selection), agent_end (`:prompt` index row), digest-all (sweep re-embedding), digest-session (session backfill), and migration (detection + apply). Before this decision, these sites used different derivations — the hook used a preamble-prefixed augmented prompt, agent_end used raw prompt, and digest scripts clipped raw prompt bytes — causing the query vector to diverge from the stored vector, which blocked stash reuse and degraded semantic retrieval.

The noise-cleaning pipeline already existed (`cleanPromptNoise` — fence collapse, JSON extraction, repetition folding) but was not uniformly applied, and clipping was absent from the new prompt path.

## Decision
All prompt-side embed sites share one derivation: `buildPromptEmbeddingInput(raw, config, maxTokens)`. The derivation order is **cleanup first** (placeholders shrink high-entropy spans before the budget), **clip second** (slice to `maxTokens` UTF-16 chars), **hash over the exact final input**. Every site passes its resolved `config.embeddingMaxTokens`.

The `promptVecHash` field (sha256 of the derivation's output) is the discriminator for stash reuse: `round.promptVecHash === promptInputHash` gates reuse of the hook's cached embedding vector. When hashes diverge (e.g., the hook's 200-word clip on long string prompts, or leading-space divergence when array content starts with a non-text block), agent_end re-embeds fresh so the `:prompt` row's hash stamp stays honest.

Migration stamps hash the exact final embedding input, so pre-decision over-budget rows are detected as hash-mismatch and re-embedded + restamped.

## Consequences
- One derivation, one budget: all five prompt-side embed sites share the contract; drift is structurally impossible.
- Stash reuse is exact: the hash gate catches every derivation divergence, not just the old `cleanedPrompt === userPrompt` no-op check.
- Correctness always, reuse skipped only where inputs genuinely differ.
- Non-positive `maxTokens` disables the clip (noise-option convention).
- BM25 keeps the raw, un-cleaned prompt text — preamble included — because preamble terms match no stored document, contributing ~0.