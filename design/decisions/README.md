# Design decisions

One indexed line per decision. Newest at the top.

## Context-provider contract

- 021 — Two-phase provider render with a frozen SectionsView — accepted
- 020 — Provider contract public surface: package-entry named exports — accepted
- 019 — Provider contract: registration validation and fail-by-omission in the base contract — accepted

## Context formatting

- 018 — Follow-up and checkpoint injection on all context paths — accepted
- 017 — Fenced-envelope fallback for prompt extraction — accepted
- 016 — Raw prompt recovery via exact prefix strip — accepted

## Performance and correctness invariants

- 015 — Atomic file writes: tmp+rename — accepted
- 014 — Hoist-once artifact loading — accepted
- 013 — Context budget charging: charged == injected — accepted
- 012 — Depth cap in findMatchingBracket — accepted

## Round recovery and startup

- 011 — Startup embedding deferral — accepted
- 010 — Marker semantics: index rows are retrieval truth — accepted
- 009 — Session-JSONL backfill — accepted
- 008 — Write-first + defensive-catch round persistence — accepted

## Index consistency and concurrency

- 007 — Orphan index entry preservation — accepted
- 006 — Label-guarded index appends for idempotency — accepted
- 005 — Short-lock merge model for reindexing — accepted

## Embedding pipeline

- 004 — import-claude-code.ts exemption from shared derivation — accepted
- 003 — Prompt embedding convention — accepted
- 002 — Round file naming hash contract — accepted
- 001 — Prompt embedding input derivation — accepted