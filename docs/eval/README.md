# Eval outputs

`baseline-weak.json` is the committed maintainer retrieval baseline produced by `scripts/eval-retrieval.ts`.
It was computed against a private maintainer corpus, so it is useful for tracking this repository's regression history but is not directly reproducible by users without those round files.

Weak labels come from `parentId` links and `get_round_details` expansions observed in session logs.
Treat the metrics as a noisy baseline, not as ground-truth relevance.

## Local baseline workflow

Create a frozen snapshot of your own rounds and pi sessions, then build a local baseline from that snapshot:

```sh
just snapshot
just build-baseline-weak ~/.pi/agent/semblr/snapshots/<snapshot-dir>
```

Both `just eval` and `just build-baseline-weak` pass `--sessions <corpus>/sessions`, matching the snapshot layout exactly. `scripts/eval-retrieval.ts` follows the same convention when `--sessions` is omitted.

The snapshot layout is:

```text
<snapshot>/rounds/index.csv
<snapshot>/chain-read-stats.json
<snapshot>/sessions/
```

`just build-baseline-weak` writes `docs/eval/baseline-weak.local.json` by default, which is ignored by git.
Pass an explicit output path if you want to store the report elsewhere.

## Golden labels workflow

Golden labels are local maintainer artifacts. They include private prompt text and corpus-specific round hashes, so keep them out of git.
The selection pool, worksheet, and ingested labels are local-only artifacts ignored by git:

- `docs/eval/golden-pool.local.json`
- `docs/eval/golden-worksheet.local.md`
- `docs/eval/golden-labels.local.json`

Build the local pool and worksheet from a snapshot:

```sh
just build-golden-pool ~/.pi/agent/semblr/snapshots/<snapshot-dir>
```

Then edit `docs/eval/golden-worksheet.local.md`, marking true labels by changing `- [ ] candidate.json` to `- [x] candidate.json`.
Each candidate now includes a short excerpt under the hash, and you may mark one selected candidate as primary by adding `(primary)` on its checkbox line.
Finally ingest the local worksheet into local labels:

```sh
just ingest-golden-labels
```

`ingest-golden-labels` reads `golden-pool.local.json`, reads `golden-worksheet.local.md`, and writes `golden-labels.local.json` with both `labels` and optional `primary` per query.

## Committed maintainer baseline

Current committed baseline was computed against snapshot `corpus-2026-06-12`
(sha256 `fbb8088b7234641143ffc58b45f9506150776efdc9fc605d4f2a7af00dba19d7`,
2947 rounds; repo tag `corpus-2026-06-12`). Maintainers with that private snapshot can reproduce it with:

```sh
just eval ~/.pi/agent/semblr/snapshots/corpus-2026-06-12 docs/eval/baseline-weak.json
```
