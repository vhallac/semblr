# 004 — import-claude-code.ts exemption from shared derivation

**Status:** accepted

## Context
`import-claude-code.ts` imports rounds from Claude Code session files. It uses a legacy raw `slice(0, embeddingMaxTokens)` for prompt embedding — already bounded, unstamped, and without noise cleanup. Bringing it under the shared `buildPromptEmbeddingInput` convention would change newly-imported rounds' embeddings, which is out of scope for the import path's contract (it should faithfully reproduce what Claude Code generated, not alter it).

## Decision
`import-claude-code.ts` keeps its legacy raw-slice prompt embed. Migration reconciles its rows automatically at sweep time:
- Short no-op-cleanup rows → stamp-only (zero API calls).
- Long rows → legacy-clean re-embed over possibly identical bytes, gaining a correct stamp.

No migration is needed at import time; the sweep handles it on the next `just index` run.

## Consequences
- Imported rounds keep their original embedding domain; import is fast and side-effect-free.
- Migration cost is deferred to the sweep, which is idempotent.
- The exemption is the only site in semblr that does not route through `buildPromptEmbeddingInput`.