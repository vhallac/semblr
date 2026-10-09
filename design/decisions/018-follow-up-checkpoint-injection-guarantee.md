# 018 — Follow-up and checkpoint injection on all context paths

**Status:** accepted

## Context
Semblr injects `round_needs_followup` and `semblr_checkpoint` sections into the context to maintain continuity across session boundaries. The original code only injected these on the normal (embedding-available) path. Degraded paths — no API key, empty index, no relevance results, or embedding error — silently dropped these continuity sections, breaking cross-session context exactly when the embedding API was unavailable.

Additionally, the last round was unconditionally appended after relevance selection, potentially duplicating the first recency entry (the previous round is always the first recency entry).

## Decision
**All context construction paths inject follow-up/checkpoint sections.** The early-return structure is restructured so recency and follow-up/checkpoint are built independently of relevance outcome:

1. Recency list is constructed first and always (even when relevance is null).
2. Follow-up/checkpoint sections are injected on every path — normal, no-API-key, empty-index, no-relevance, and error.
3. Dedup sets prevent double-injection when both follow-up and checkpoint reference the same round.
4. The unconditional last-round append is removed from relevance — the previous round is always the first recency entry.

## Consequences
- Cross-session continuity is never silently dropped, even when the embedding API is unavailable.
- Recency always includes the most recent rounds, regardless of embedding state.
- No duplicate rounds in context (dedup sets + removal of unconditional last-round append).
- Degraded context paths are now functionally equivalent to the normal path for continuity features.