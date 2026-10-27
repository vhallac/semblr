# 023 — Per-host index shards: own-shard writes, merged reads

**Status:** proposed

## Context
The vector index was a single file (`index.csv`) in the rounds directory. Every host that shares a rounds directory — a synced folder, a shared filesystem — appends to that one file, and the compaction sweep replaces it wholesale. The append lockfile serializes writers only within one host, and a wholesale rewrite is a read-modify-write of the whole file: a host that compacts while another host appends can drop the other host's rows. The same hazard applies to the index-rewriting migration scripts. Issue #149.

## Decision
**The vector index is a set of per-host shards: one file per host, `index.<hostname>`, in the rounds directory** (the shard root is the directory of the resolved index path, so an explicitly configured index path keeps its shards beside it).

- **Writes go to the host's own shard only.** Live appends, the `just index` sweep/compaction, and the index-rewriting migration scripts all publish to `index.<hostname>`.
- **Reads see the merged shard set.** The startup load, the TUI counts, retrieval, the startup pending predicate, the sweep's "already indexed" checks, and the read-only scripts all iterate every discovered shard.
- **The hostname lives in the filename; a shard carries no header.** Shard files stay byte-compatible with the legacy index format, so there is no format version and no reader migration.
- **An empty shard set is an empty index, not an error**, and a corrupt row is skipped with a warning naming the shard it came from.
- **A shard write is atomic** (tmp+rename, decision 015) and keeps the legacy row format.

Accepted non-goals (issue #149): **no cross-shard dedup and no global compaction.** Shards are independent — a round embedded on two hosts exists as rows in both shards and is counted twice by the merged read.

## Consequences
- The inter-host lost-update hazard disappears: a host rewrites only its own shard, and a foreign shard is never touched.
- The cost is coordination. A rewrite (model migration, content-hash migration, prune) must run on every host; a foreign shard's rows are only updated by the host that owns it, and the scripts report them instead of changing them (decision 025).
- Merged counts are a sum over shards and can exceed the number of distinct rounds. The content-hash round-id contract (decision 002) keeps round *files* unique but no longer makes index *rows* globally unique.
- A host that changes its embedding model leaves its own old-model rows in place until its next sweep rewrites them, and a foreign shard's old-model rows are rewritten only by that host. The merged read therefore mixes models and ages — scoring has to be model-aware and dimension-aware, which it is not everywhere yet (see [issue #153](https://github.com/vhallac/semblr/issues/153) and README "Known Limitations").
- Shard discovery must exclude every sibling artifact of the index, or a backup or a sidecar file is read as a shard (decision 024).
- Related: 024 (legacy adoption and shard discovery), 025 (foreign shards are read-only).
