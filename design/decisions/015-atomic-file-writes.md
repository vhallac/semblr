# 015 — Atomic file writes: tmp+rename

**Status:** accepted

## Context
Multiple write paths in semblr used in-place `fs.writeFileSync` — including index rewrites and round file writes. An in-place write can produce a partial file visible to concurrent readers, and a crash mid-write leaves a corrupted file. The `appendLineWithLock` function already used tmp+rename inside a lockfile protocol.

## Decision
**All durable writes use tmp+rename, never in-place overwrite.** The pattern is: write to a temporary file, then `fs.rename(tmp, target)` — an atomic operation on the same filesystem. This applies to:

- Index rewrites (short-lock merge, 005).
- Round file writes (`writeRoundEmbedding` closure in digest-all).
- BM25 index writes.
- Any future path that writes a file that concurrent readers may open.

In-place `writeFileSync` is banned for files larger than trivial config writes.

## Consequences
- Concurrent readers never see a partial file.
- A crash mid-write leaves either the old file intact or the new file complete — never a corrupted intermediate.
- The `fsImpl` seam (dependency-injected filesystem interface) is needed for testing: ESM `vi.spyOn` cannot intercept `node:fs` namespace exports.
- Tests assert that in-place writes of `.json` paths never occur; tmp+rename pairing is verified by spying on the injected filesystem.