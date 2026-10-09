# 013 — Context budget charging: charged == injected

**Status:** accepted

## Context
The context budget (`selectContextRounds`) charges entries to stay within the token budget. An invariant claim in the code ("what is charged to the budget is exactly what is injected") was violated: `selectContextRounds` charged entries without size tags, while `buildRelevanceList` rendered entries with size tags (` | 12.34KB`). The injected text was ~3 tokens/entry larger than what was charged. The difference was bounded (20-entry cap × ~3 tokens = ~60 tokens), but the invariant was false.

## Decision
**Charge the size tag** (restore the invariant) rather than weaken the doc wording. Threading a `getRoundSizeFn` into `selectContextRounds` mirrors the existing `renderSearchInteractionsToolResult` pattern.

Charging slightly more per entry (size tag ~3 tokens) means the budget bites marginally earlier — caps remain hard, the invariant holds.

## Consequences
- The `charged == injected` invariant is true by construction.
- The budget is marginally tighter (~60 tokens for a 20-entry list), but the cap already binds before the budget in typical configurations.
- The `getRoundSize` function is threaded through the selection path, following the same pattern as the rendering path.