# 002 — Round file naming hash contract

**Status:** accepted

## Context
Semblr round files are named by a content hash to enable deduplication. Multiple code paths compute this hash: live write (`buildAgentEndRoundFile`), session-JSONL reconstruction (`reconstructPiSessionRounds`), and backfill early exit (`findMissingRounds`). These paths independently hashed the round content, leading to hash asymmetry — reconstruction hashed raw `responseSequence` (including a trailing `round_needs_followup` marker), while the live write path hashed marker-stripped text. The result: 155/155 marker-carrying rounds hash-diverged, producing duplicate files in the rounds store.

## Decision
One shared derivation is the hash contract: **raw text → strip marker → apply `needsFollowup` flag + capture `cleanedText` → `createRoundFilePath(cleanedText)`**. Live write, reconstruction, early exit, and digest scripts all route through this single derivation. The marker strip + `needsFollowup` flag extraction happens at one canonical site; no caller hashes raw marker-included text.

## Consequences
- Filename parity across all write paths: reconstruction always produces the same filename as the live write path.
- Early exit in backfill is correct — `existsSync` on the target path matches.
- Duplicate-in-waiting rounds are structurally prevented.
- Digest script fix-up paths (`scripts/fix-round-tool-ids.ts`) also route through the shared derivation.