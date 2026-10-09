# 017 — Fenced-envelope fallback for prompt extraction

**Status:** accepted

## Context
User prompts sometimes arrive wrapped in a fenced envelope (triple-backtick code block containing the actual message). `findFencedEnvelope` extracts the parsed content from inside the fence. When the fenced content collapses to whitespace-only prose, the old behavior returned an empty or near-empty prompt, losing the user's actual message.

## Decision
**Fenced-only fallback:** when fenced collapse leaves whitespace-only prose (checked by `text.replace(fenced.full, "").trim() === ""`), extract the user text from the parsed message content:

```typescript
extractUserTextFromMessages(fenced.parsed) ?? makePlaceholder(fenced.body)
```

`fenced.parsed` is already available from `findFencedEnvelope` — no re-parse needed. The placeholder ensures the prompt is never empty even when extraction produces nothing.

## Consequences
- Envelope-wrapped prompts are never lost to whitespace-only collapse.
- No re-parse cost: `findFencedEnvelope` already returns the parsed structure.
- The fallback is a safety net, not the primary path — normal fenced content still extracts directly.
- The trim check correctly distinguishes "placeholder-only" from "genuine user text outside the fence".