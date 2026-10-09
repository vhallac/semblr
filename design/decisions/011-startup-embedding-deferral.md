# 011 — Startup embedding deferral

**Status:** accepted

## Context
When semblr starts and discovers recovered rounds needing embedding, the original design embedded them synchronously at startup. This blocked the session from starting until all embeddings were computed — potentially minutes of API calls. The embedding burst at startup was a poor user experience and could cause timeouts.

## Decision
**Only embedding is deferred on startup.** BM25 indexing and tool index (`indexRecoveredRounds`) MUST stay synchronous — these are fast and required for immediate retrieval functionality.

Deferred rounds are counted and reported with a message prompting the user to run `just index` (or `just digest-session` for session-specific backfill). The defer threshold is configurable; the message wording is extracted into a testable builder function.

Vectors are deferred, not dropped: the next `just index` sweep processes them via the standard `processRound` / `embedRound` pipeline.

## Consequences
- Startup is fast regardless of how many rounds need embedding.
- BM25 + tool index are immediately available for the current session.
- The user is explicitly told how many rounds are pending and how to embed them.
- The defer threshold boundary is tested and message wording is pinned.