# 008 — Write-first + defensive-catch round persistence

**Status:** accepted

## Context
The `agent_end` handler writes the round file, then appends tool index rows and computes embeddings. Originally, `fs.writeFileSync` for the round file was placed after grouping logic and before embeddings — any exception before that write (including in grouping) lost the round silently. Process death (kill / Ctrl+C shutdown) produced no round file at all, with no recovery path from semblr's own handlers since `agent_end` never fired.

Empirical testing confirmed:
- ESC abort during tool call or inference: `agent_end` still fires → round saved (partial text preserved).
- Process death mid-run: no `agent_end` → full loss.
- Exception in semblr's `agent_end` handler before `writeFileSync`: swallowed by pi's per-handler catch → silent loss.

## Decision
**Two-pronged fix:**

1. **Write-first ordering**: the round file is written to disk as early as possible in `agent_end` — before tool index appends, before embedding computation. A failed write at position X must never be skipped over by later rounds (gap-free write ordering).

2. **Defensive catch**: the entire `agent_end` handler body beyond the write is wrapped in a try/catch that prevents silent loss. Catch always returns (null / error log), so TypeScript narrowing across try/catch is handled by declaring the saved variable as `| null` and checking after the block.

A `postWrite` callback receives `(saved, roundData)` as arguments — avoiding TDZ traps from closures referencing results declared after the persist call.

The dedup branch (existing round file already present) stays in the handler because grouping reads the existing file and needs session state; persist only detects dedup.

## Consequences
- Silent round loss from handler exceptions is structurally prevented.
- Process death still loses the round (no `agent_end` fires), but session-JSONL backfill (009) provides the recovery path.
- Write-first means the round file is durable even if embedding or tool-index work fails afterward.
- The dedup path is preserved: if a round file already exists, grouping reads it and behaves correctly.