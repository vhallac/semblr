# Open issues

A running log of issues that are known and unfixed but are not (yet) owned by a tracked GitHub issue — plus the tracked ones that remain open after the multi-host index work (#149).

Entries are appended, never rewritten: a closed entry keeps its text and gains a closing line. Every entry names its anchor (the code, the decision, or the tracker issue it comes from), why it is still open, and what a fix would look like.

Started at the archive of the #149 plan (per-host index shards). New entries are appended as the work uncovers them.

## Tracked elsewhere

| Issue | Subject | State |
|---|---|---|
| [#146](https://github.com/vhallac/semblr/issues/146) | A rewrite that races a concurrent append can drop the appended row (read-modify-write without the append lock) | Open. #149 made the *own-shard* rewrites atomic (tmp+rename) and confined them to one shard; the lock-ordering hazard between a rewrite and a concurrent append, and the failure-injection tests, are still open. |
| [#136](https://github.com/vhallac/semblr/issues/136) | A startup embedding burst that crashes midway is never retried | Open. Unrelated to the shard model. |

## OI-1 — A merged read mixes embedding models, and the injection path scores every row with one query vector

- **Anchor:** `src/semblr.ts` — the context-injection path calls `collectSearchRoundScores` with a single query vector, while the `search_interactions` tool path calls `collectMultiModelSearchRoundScores` (one query vector per model, per row model). Row format: `<vector>,<file>:<suffix>,<model>`. `lib/vector.ts` — `cosineSimilarity`.
- **What:** the automatic context injection loads the merged shard set and scores every row against one query vector; it never looks at the row's model column.
- **Why it matters:** a shard written by a host that uses a different embedding model is scored in an incompatible vector space. `cosineSimilarity` iterates over the *query* length, so a foreign vector that is shorter yields `NaN` scores (a silent poison for the ranking) and a longer one is silently truncated to the query's dimensionality — plausible-looking but meaningless scores.
- **Fix sketch:** filter the merged rows by the current model in the injection path (or score per model as the tool path does), and skip a row whose vector dimension does not match the query.
- **Severity:** high — silently wrong context on a multi-host store whose hosts use different models.

## OI-2 — The BM25 index and the tool-call fulltext index are still single shared artifacts, written in place

- **Anchor:** `lib/bm25-index.ts` — `bm25IndexPathForRoundsDir`, `writeBm25Index`; `lib/search-tools.ts` — `toolIndexPathForRoundsDir`, `writeToolIndexRows`. Callers: `src/semblr.ts`, `scripts/digest-all.ts`, `scripts/rebuild-bm25.ts`, `scripts/prune-recovery-duplicates.ts`.
- **What:** unlike the vector index (decision 023), these two sibling indexes are not sharded. `index.bm25.json` is rebuilt wholesale from the round files; `index-tools.fulltext.csv` is appended per round and rewritten in place by the prune script.
- **Why it matters:** on a shared rounds directory they carry exactly the write-conflict hazard #149 removed for the vector index — a wholesale rewrite can drop another host's rows — and both write in place with `writeFileSync`, contradicting decision 015 (atomic writes; the BM25 write is named there explicitly). A concurrent reader can observe a half-written file.
- **Fix sketch:** either shard them like the vector index (own-shard writes, merged reads), or declare them per-host local artifacts that are rebuilt from the round files on demand — both are derivable from the round files, so a shared copy may be unnecessary.
- **Severity:** medium-high.
- **Tracked:** [#152](https://github.com/vhallac/semblr/issues/152) — per-host, rebuildable keyword and tool-call artifacts (decided: per-host derived artifacts, no cross-host merge).

## OI-3 — The stale-file migration in `digest-session` writes the own shard in place, without the append lock

- **Anchor:** `scripts/digest-session.ts` — `migrateIndexEntriesFile(index.ownShardPath, staleFile, roundFile)`; `lib/index-io.ts` — `migrateIndexEntries` → `writeIndexLines` (in-place `writeFileSync`).
- **What:** the content-hash migration renames the rows of a stale round file inside the own shard through a read-modify-write that ends in an in-place write — no tmp+rename (decision 015) and no lockfile (decision 005's short-lock merge model).
- **Why it matters:** a crash mid-write leaves a truncated shard, and a concurrent append by the extension can be lost by the read-modify-write. The rewrite scripts migrated in #149 all publish through the atomic path (`writeOwnLines`); this one did not.
- **Fix sketch:** route the row migration through the same atomic own-shard write as the other rewrite scripts.
- **Severity:** medium.

## OI-4 — The merged read does not deduplicate across shards

- **Anchor:** `lib/index-io.ts` — `loadIndexFromShards` concatenates the rows of every shard; README "Known Problems — No cross-shard dedup"; decision 023 (accepted non-goal).
- **What:** a round embedded by two hosts exists as a row in both shards and is counted twice by the merged read.
- **Why it matters:** the merged count overstates the number of distinct rounds, and any future global statistic (coverage, cost, orphan checks) inherits the double count.
- **Fix sketch:** a global compaction pass (one host owns the dedup) or a round-id-keyed merge at read time — the latter costs a map over the whole index on every read.
- **Severity:** medium — documented, but every count is affected.
- **Status:** fixed on the PR !151 branch — the merged read (`loadIndexFromShards`, `loadShardVectorIndex`) now deduplicates by round file, row suffix and embedding model (`dedupeIndexEntries` in `lib/index-io.ts`); the shards on disk are untouched. Rows of different models are deliberately *not* duplicates: the search groups the index by model and the coverage predicate needs the stale-model rows (issue #140 D1).

## OI-5 — A stale row is never replaced once any shard holds a current-model row for the same label suffix

- **Anchor:** `lib/embed-round.ts` — `classifyRoundCoverage` (a stale suffix only forces a re-embed when no current-model row exists *anywhere* in the merged set); `lib/session-backfill.ts` — `buildCurrentModelRowPredicate` (the startup pending predicate shares the rule).
- **What:** a shard holding an old-model row for `file:prompt` is treated as covered as soon as *another* shard holds a current-model row for the same suffix. The stale row stays in its shard.
- **Why it matters:** the stale embedding is never replaced by anyone — the host that owns it sees the round as covered, and the host that owns the current row cannot rewrite a foreign shard (decision 025). Stale rows therefore accumulate in shards and keep feeding the mixed-model scoring of OI-1.
- **Fix sketch:** make the staleness decision per shard (ignore foreign shards), or add a sweep step that rewrites the own shard's stale rows regardless of what other shards hold.
- **Severity:** medium.
- **Status:** fixed on the PR !151 branch — the `just index` sweep now decides staleness per shard on the write side: a stale forcing row in the host's own shard triggers a rewrite of that shard's rows even when a foreign shard holds a current-model row for the same suffix (`ownShardHasStaleForcingRow` in `lib/embed-round.ts`, applied in `scripts/digest-all.ts`). The start-up pending predicate keeps its cross-shard meaning (a round covered by current-model rows anywhere is not pending), so startup does not re-embed in a loop. A stale row in a foreign shard is untouched — it is replaced when the host that owns it runs the sweep.

## OI-6 — Host identity is `os.hostname()`, so a renamed host orphans its shard

- **Anchor:** `lib/index-shards.ts` — `ownShardPath(roundsDir, hostname = os.hostname())`.
- **What:** the shard name is the host's name at write time. A renamed machine, a new container id, or a moved home directory starts a new shard and leaves the old one behind as a permanent foreign shard.
- **Why it matters:** the old shard keeps contributing rows to every merged read (including stale-model rows, OI-1/OI-5), and nothing prunes it.
- **Fix sketch:** a stable host identity (a configured id, or a marker file inside the rounds dir) instead of the raw hostname, plus a way to retire a shard deliberately.
- **Severity:** low-medium.

## OI-7 — `replaceIndexEntriesForRoundFile` is an exported, non-atomic rewrite helper with no production caller left

- **Anchor:** `lib/index-io.ts` — `replaceIndexEntriesForRoundFile` (read-modify-write ending in `writeIndexLines`, i.e. in-place) and `writeIndexLines`.
- **What:** the sweep migrated to the shard-aware atomic path; this single-file rewrite helper stayed behind. No production code calls it any more (only `lib/index-io.test.ts` exercises it), but it is still exported.
- **Why it matters:** it is a trap for the next caller: a correct-looking helper that reintroduces a non-atomic whole-index write (decision 015) and that takes no shard argument, so it cannot respect the own-shard rule of decision 023.
- **Fix sketch:** delete it, or reroute it to the atomic own-shard write.
- **Severity:** low.

## OI-8 — Outstanding per-host work is reported, never tracked

- **Anchor:** `lib/script-config.ts` — `reportForeignShards` / `foreignShardsReferencing`; decision 025.
- **What:** a rewrite script prints which foreign shards still hold rows (and which round files they reference), but nothing records that foreign half of the migration as outstanding.
- **Why it matters:** the operator has to notice a line of log output to know that a migration is only half done on a multi-host store; nothing surfaces the backlog later.
- **Fix sketch:** record the outstanding foreign shards in a small state file in the rounds directory (or expose them through a status command), and have the scripts clear it when the foreign host runs the same migration.
- **Severity:** low.
