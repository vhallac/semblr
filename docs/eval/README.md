# Eval outputs

`baseline-weak.json` is the retrieval baseline produced by `scripts/eval-retrieval.ts`.
Compute it against an unpacked snapshot corpus that contains `rounds/index.csv` and `chain-read-stats.json`.
The labels are weak: they come from `parentId` links and `get_round_details` expansions observed in session logs.
Treat the metrics as a noisy baseline, not as ground-truth relevance.

Current baseline was computed against snapshot `corpus-2026-06-12`
(sha256 `fbb8088b7234641143ffc58b45f9506150776efdc9fc605d4f2a7af00dba19d7`,
2947 rounds; repo tag `corpus-2026-06-12`). Reproduce with:

```sh
just eval ~/.pi/agent/semblr/snapshots/corpus-2026-06-12
```
