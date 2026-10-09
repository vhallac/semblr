# 014 — Hoist-once artifact loading

**Status:** accepted

## Context
`processRound` (the digest-all worker) reloaded whole-store artifacts for every round in the sweep: `findStaleContentMatches` (all round files), `loadBm25Index`/`writeBm25Index` (315 MB), `loadToolIndexedRoundFiles` (54 MB), `loadIndexedRoundFiles` + `loadRoundFilesWithDifferentModel` (496 MB CSV parsed twice). For an N-round sweep, this was ~5N full-store loads — O(n²). A full backfill of thousands of rounds was impractically slow (~20s/round).

## Decision
**Single load + in-memory dedup sets.** All store-wide artifacts are loaded exactly once at sweep start:

- `index.csv` is parsed once; three in-memory structures (dedup set, model-mismatch set, response-vector map) are derived from that single parse.
- BM25 index is loaded once and written once after `Promise.all(workers)`.
- `staleMatchesByTarget` is built from a single `buildStaleContentMatchMap` pass before the worker loop.
- Per-round reloads are eliminated entirely.

The sets stay current through in-loop updates: on stale-file migration, on tool-index append, and on row write (append adds the file; replace clears the mismatch flag).

The O(n²) reload pattern is outlawed for any future code touching the sweep path.

## Consequences
- Sweep performance is O(n) for store-wide artifact access.
- The `fsImpl` seam (015) is needed to make reads observable for testing, since ESM namespace exports cannot be `vi.spyOn`-ed.
- In-memory set currency is maintained by in-loop updates; no re-parse needed.
- Cross-size equality tests confirm O(1) artifact loads (N=3 and N=6 produce identical read/write counts).