# 007 — Orphan index entry preservation

**Status:** accepted

## Context
`replaceIndexEntriesForRoundFile` drops every existing row for a round file and writes a replacement list. If the replacement list omits a label suffix that exists in the current index (e.g., a `:summary` row that the reindex path does not reproduce), that information is silently lost. A crash between row append and marker write, or a session-scan path that doesn't derive `:summary`, would cause permanent data loss on the next reindex.

## Decision
**Never trade away information on replace.** Before dropping existing rows, identify any whose label suffix (`:prompt`, `:response`, `:round`, `:summary`, or bare) is not reproduced by the new entries. Those orphan rows are reported (`⚠️ Preserving non-reproduced index row: <label>`) and re-appended after the replacement.

The comparison is by **suffix**, not whole line — so legitimate replaces (old model's `:prompt` row dropped in favor of the new model's `:prompt` row) still work correctly.

The guard lives at the call site in `scripts/digest-all.ts`, not inside `lib/index-io.ts`, so `replaceIndexEntriesForRoundFile` keeps its exact "replace all rows for roundFile" contract for any other caller.

## Consequences
- A reindex never silently drops an index row it did not reproduce.
- Orphan `:summary` rows from session-scan rounds survive model-change reindexes.
- The guard is additive — it preserves, never blocks.
- The suffix comparison ensures model-change replacements are still effective (old row dropped, new row written).