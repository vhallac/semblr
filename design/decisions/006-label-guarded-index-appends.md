# 006 — Label-guarded index appends for idempotency

**Status:** accepted

## Context
Index rows carry a `fileName:label` composite key (e.g., `abc123.json:prompt`). When a round is re-indexed — due to a model change, digest sweep, or recovery re-run — appending rows naively would produce duplicate `fileName:label` entries. The alternative ("write `promptEmbedding` marker first, then append") trades duplicate rows for silently lost rows when a crash lands between the marker write and the row appends.

## Decision
**Label-guarded index appends:** before appending a row, check whether a row with the same `fileName:label` already exists in the index. If it does, skip the append. This makes re-runs idempotent regardless of append order.

Index writes use `appendLineWithLock` (tmp+rename inside a lockfile protocol), and the guard operates on the in-memory view after the lock is acquired but before the append.

## Consequences
- Reindexing is idempotent: running it twice produces the same index state.
- Crash safety: a crash between row appends leaves partial rows, but a re-run skips duplicates.
- No dependency on marker ordering (the `promptEmbedding` field in the round file).
- The guard compares by `fileName:label` suffix, not whole line — matching the replace-before-append semantics of `replaceIndexEntriesForRoundFile`.