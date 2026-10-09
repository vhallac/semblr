# 016 — Raw prompt recovery via exact prefix strip

**Status:** accepted

## Context
The agent hook receives an augmented prompt: pi prepends an `[ENVIRONMENT]` preamble block + `[ACTIONABLE PROMPT]` marker. The hook needs the raw user prompt for embedding (to produce the same vector that agent_end will store) and for BM25 indexing. A heuristic check ("starts with `[ENVIRONMENT]`") would falsely strip host-injected env-style blocks with a different preamble, altering the prompt.

## Decision
**Exact prefix strip**, not heuristic. The hook recovers the raw prompt by stripping the exact `envPreamble + ACTIONABLE_PROMPT_MARKER` prefix via `stripEnvPreamble` in `lib/context-messages.ts`. Augmentation sites share `ACTIONABLE_PROMPT_MARKER` with the strip function — single source of truth, drift impossible.

The exact strip means only the specific preamble pi injects is removed; any user prompt that genuinely starts with `[ENVIRONMENT]` (e.g., discussing semblr's own format) is preserved intact.

## Consequences
- False-positive stripping of user content is structurally impossible.
- The marker string is shared between augmentation and stripping — changing it requires changing both, but that's a single-source edit.
- The raw prompt recovered at the hook is byte-identical to what agent_end sees, enabling stash reuse and semantic query parity.
- BM25 keeps the raw prompt including preamble — preamble terms match no stored document, contributing ~0 to retrieval.