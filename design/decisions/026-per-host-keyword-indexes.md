# 026 — Per-host keyword indexes: BM25 and tool-index shards

**Status:** proposed

## Context
Decisions 023–025 made the vector index a set of per-host shards with own-shard writes and merged reads. Two derived keyword artifacts remained single shared files: the BM25 index (`index.bm25.json`) and the tool index (`index-tools.fulltext.csv`). Multiple hosts sharing a rounds directory overwrote each other's keyword artifacts, and a per-host BM25 prune (the duplicate-resolution path) would clobber a foreign host's rows. Issue #152.

## Decision
**The keyword artifacts become per-host files, mirroring the vector shard model:** `index.<hostId>.bm25.json` and `index-tools.<hostId>.fulltext.csv` beside the vector shards.

- **Derived, not sharded.** Both indexes are fully rebuildable from round files, so there is no cross-shard merge logic; each host maintains only its own derived artifact. A sharded BM25 needing cross-shard term statistics has no consumer.
- **Legacy adoption by rename-once**, mirroring decision 024: on the first write by a host, a pre-existing legacy `index.bm25.json` / `index-tools.fulltext.csv` is adopted by rename into the host's own name, with a clobber guard and a swallowed lost race. A legacy index built from all rounds stays valid as a superset; per-host flushes then upsert the host's own rounds.
- **File formats unchanged.** The BM25 JSON and tool-CSV schemas stay byte-compatible; the host id lives only in the filename (decision 023's no-header choice).
- **Discovery exclusions are pattern-based.** `isIndexShardFilename` excludes the per-host keyword artifact names (and their backups) instead of the two exact legacy names, keeping decision 024's naming contract.
- **Atomic writes.** `writeBm25Index` and `writeToolIndexRows` move from in-place `writeFileSync` to tmp+rename, aligning the code with decision 015.
- **eval-retrieval routes to the own artifact**, so a multi-host corpus eval reads what the corpus host built; single-host behavior is unchanged.

## Consequences
- Per-host BM25 prunes no longer clobber foreign hosts' keyword artifacts; each host only rewrites its own files.
- Consistency with the vector index: same discovery, same adoption pattern, same host-id resolution (`resolveHostId`).
- Cost: foreign rounds not re-embedded locally appear in the vector index (merged reads) but are covered in the local BM25 only after the startup rebuild from locally visible round files — per use case step 3 of issue #152.
- Related: 023 (per-host index shards), 024 (legacy adoption and shard discovery), 025 (foreign shards are read-only), 015 (atomic file writes).
