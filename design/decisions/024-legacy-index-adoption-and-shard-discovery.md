# 024 — Legacy index adoption by rename and shard discovery exclusions

**Status:** proposed

## Context
Every existing store holds one headerless `index.csv`. The shard model (decision 023) has to absorb those rows without re-embedding them and without changing the file format. Issue #149 asked how a legacy index is adopted (Q1 — option b: adopt by rename) and whether a shard carries a header of its own (Q2 — option 3: no header).

## Decision
**Discovery.** An entry of the rounds directory is an index shard if and only if its name starts with `index.` and it is not an excluded artifact. Excluded: `index.bm25.json` (the BM25 keyword index), `index-tools.fulltext.csv` (the tool-call fulltext index), migration backups (`index.*.bak*` in every naming variant the scripts write), lockfiles (`index.*.lock`), and atomic-write temps (`index.*.tmp.*`). The legacy `index.csv` counts as a shard until it is adopted.

**Adoption.** On the first write by a host that has no shard of its own, a legacy `index.csv` in the shard root is renamed to `index.<hostname>` — an atomic same-directory rename (decision 015). The renamed rows become that host's own shard rows: no row is rewritten and no round is re-embedded.

**Guard against clobbering.** A host that already owns a shard leaves a legacy `index.csv` untouched: POSIX rename overwrites, so renaming would clobber the host's own shard. The legacy file stays part of the read shard set until a host without a shard claims it.

**Lost race.** If another host claimed the legacy file between the existence check and the rename, the rename failure is swallowed and the host proceeds with its own shard.

## Consequences
- No format migration: adoption is a rename, and the shard format *is* the legacy format. A reader upgrade needs no conversion step.
- No reader gap: the legacy rows stay visible until the rename, and the rename itself never loses a row (unlike copy-then-delete).
- The claim is not coordinated across hosts. The first writing host wins; a host that loses starts with an empty shard of its own and re-embeds its rounds. Accepted for now (issue #149 non-goal, first-time adoption race).
- The exclusion rule is a naming contract, not a type marker — every artifact that shares the `index.` prefix must be excluded explicitly, and a new sibling artifact that is not listed is read as a shard. That failure mode was real: the migration backup naming `index.csv.bak-<ts>` was discovered as a shard until the rule was widened to `index.*.bak*`.
