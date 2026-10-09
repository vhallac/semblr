# 012 — Depth cap in findMatchingBracket

**Status:** accepted

## Context
`cleanPromptNoise` applies structural collapses in order: fences → JSON → repeat → long tokens. `findMatchingBracket` is triggered for every `[` character in a `[`-run (each opener passes the `looksLikeJsonStart` prefilter) and scans to end-of-text when no closer exists, producing O(n²) behavior. Measured: 10,000 `[` characters → 104 ms; 50,000 → 2,419 ms. This is pure latency in the `agent_end` hot path.

Reordering repeat-before-JSON would violate the documented "repetition last" design and could break JSON detection for dumps containing repeat runs.

## Decision
**Depth cap inside `findMatchingBracket`**: bail → -1 once depth exceeds a small constant. Precedent: `JSON_PARSE_LENGTH_CAP` = 1M for parse size. Worst case becomes O(n·cap) ≈ linear for a fixed cap.

Sane JSON (any realistic nesting depth) is unaffected. Performance is pinned by a generous-threshold smoke test (timing-assert flakiness avoided by a wide margin).

## Consequences
- O(n²) → O(n) for pathological `[`-run inputs in the hot path.
- No reordering of the noise-cleaning pipeline — "repetition last" is preserved.
- JSON with extreme-but-valid nesting depth may fail to parse (theoretical; cap is generous).
- The fix is a one-line depth guard, not a pipeline restructure.