# 010 — Marker semantics: index rows are retrieval truth

**Status:** accepted

## Context
The `promptEmbedding` field in round JSON files was historically treated as a "done" signal — both the startup pending count and embed-deferral logic keyed on its absence. However, this field is derived state written after index rows are appended. A crash between row append and marker write leaves a round with current-model index rows but no marker; a marker written without rows (impossible under normal ordering, but possible if a path writes the marker without writing rows) would look "done" while lacking retrieval data.

The sweep (`just index`) used `index.csv` rows as its predicate, while the startup backfill used the `promptEmbedding` marker. These predicates disagreed, causing rounds to be double-counted or silently skipped.

## Decision
**`index.csv` rows are retrieval truth.** The `promptEmbedding` marker in the round file is **derived state, never a done-signal**.

- A round covered by current-model `index.csv` rows is **not counted pending** even when the marker is missing.
- A marker present with rows missing is a real gap and is re-embedded.
- Only a missing marker (with rows present) is healed from the `:response` row vector — zero API calls.

Both startup backfill (`planStartupEmbedding`) and the sweep use the same row-based predicate (`buildCurrentModelRowPredicate`: covered if the round has current-model rows AND no mismatched-model rows). Legacy model-less rows count as current.

## Consequences
- Startup pending count and sweep agree by construction — the two predicates cannot drift.
- Crashes between row append and marker write are harmless: the rows are the truth, and the marker is re-derived on the next run.
- Rounds with mixed rows (some current, some mismatched) are treated as not covered — matching the sweep's reindex-on-any-mismatch behavior.