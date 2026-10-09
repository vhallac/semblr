# 005 — Short-lock merge model for reindexing

**Status:** accepted

## Context
Reindexing sweeps (migration, digest-all) rewrite the entire `index.csv` after potentially minutes of re-embedding work with thousands of API calls. The original approach held no lock during the re-embed phase and then did a bare `fs.writeFileSync` for the final write — a concurrent session appending rows during the re-embed window would have its rows clobbered by the sweep's write, while the sweep's result could also be clobbered by a session's atomic rename. A whole-RMW lock would have blocked live sessions for the entire re-embed duration, which is unacceptable.

## Decision
**Short-lock merge** instead of whole-RMW lock:
1. The re-embed loop (minutes, thousands of API calls) runs **outside the lock**.
2. Only the final merge phase acquires `index.csv.lock` (same protocol as `appendLineWithLock`: wx-create, retry/backoff, stale takeover).
3. Inside the lock: re-read the index, merge appended lines by exact-string set-difference, write tmp + atomic rename, release.
4. **Fail-fast** on lock exhaustion — no bare-append fallback for a wholesale rewrite.

The residual window narrows to the few-second merge phase. A session appending during that phase waits on the same lock and lands after the rename. Live sessions never block during the re-embed phase.

## Consequences
- Live sessions are never blocked during long re-embedding runs.
- The merge phase is short (seconds) and uses the same locking protocol as normal appends.
- Lock exhaustion is a hard error (suggest re-run), not a silent data-loss path.
- The merge-by-set-difference approach means appends during the re-embed phase are correctly incorporated.