# 025 — Foreign shards are read-only; shared round files are kept and reported

**Status:** proposed

## Context
The compaction sweep and the index-rewriting migration scripts perform a read-modify-write over the index and, for some scripts, rename or delete round files. Applying that to a shard another host appends to would reintroduce the hazard decision 023 exists to remove. Round *files* are shared — a synced directory gives every host the same files — but the rows that reference them live in each host's own shard, so a file can be referenced from several shards at once. Issue #149 constraints D-005, D-006, D-007.

## Decision
- **A rewrite touches the host's own shard only.** While a legacy `index.csv` is still claimable, its rows are part of the rewrite — the write claims the legacy file by rename first (decision 024) so those rows land in the own shard instead of being stranded. Once the host owns a shard, a leftover legacy file stays a separate read artifact.
- **A round file that a foreign shard still references is kept in place**: not renamed (content-hash migration), not deleted (tool-id repair, duplicate prune). The row cannot be rewritten from this host, so renaming or deleting the file would leave the other host's rows dangling. The host that owns the rows performs the change when it runs the same script.
- **Every rewrite script reports the foreign shards** it found, with their row counts, in a one-line notice — the operator learns which rows still await the other host's run.

## Consequences
- Foreign shards stay byte-identical across a rewrite by this host: the core multi-host safety property of the shard model.
- Migration is per-host. Running it on one host does not migrate another host's rows or round-file references; the operator must run it everywhere, and the report is the only signal that this is outstanding.
- Rows in a foreign shard keep their original model and timestamp until that host runs the rewrite, so the merged read mixes models and ages (decision 023).
- A round file is removable only when no shard in the set references it — the orphan rule of decision 007, evaluated over the whole shard set instead of one file.
