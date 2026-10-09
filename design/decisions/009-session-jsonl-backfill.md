# 009 — Session-JSONL backfill

**Status:** accepted

## Context
When a process death kills semblr before `agent_end` fires, the round file is never written. However, pi's session JSONL file contains the complete message history — including the messages that would have produced the lost round. This is the only viable recovery path for process-death round loss.

## Decision
**Session-JSONL backfill on next startup:**

1. Key off `session_start` event's `previousSessionFile` (pi provides it for "resume"/"fork"/"new" sessions).
2. Parse the previous session file via `parsePiSessionJsonl` + `reconstructPiSessionRounds` to extract all rounds.
3. Dedup gate: `existsSync` on the target round path (round filenames match semblr's content-hash scheme through the shared hash contract, 002).
4. **Whole-file scan**, not just tail — cheap, idempotent, and also recovers rounds lost to any earlier cause.
5. **BM25-only**: recovered rounds get BM25 upsert but NO vector embedding at backfill time. Embedding at startup would block session start and cost API calls.
6. Full vector restore = run `just digest-session` on the file (or `just index` sweep).

The hash contract (002) ensures reconstruction produces the same filename as the live write path, so `existsSync` is the correct dedup gate.

## Consequences
- Process-death round loss is recoverable from the session JSONL.
- Startup is fast: no embedding API calls, BM25 + tool index only.
- Vectors are deferred; rounds without embeddings are listed as pending and the user is prompted to run `just index`.
- Backfill scans the whole file, not tail — recovers any round lost to any cause, not just the most recent.