# Semblr

**Semantic context assembly for AI agents — experimental.**

> **⚠️  Experimental.** We are actively testing the core claims below. They are hypotheses, not settled facts. Everything here is subject to change as we gather data.

Semblr stores every conversation round permanently, embeds it, and retrieves the most relevant rounds on each user prompt — by meaning, not by recency. We want to find out whether semantic retrieval beats lossy summarization in practice.

Runs as a [pi coding agent](https://pi.dev) extension at `src`.

Version-controlled with [Jujutsu (jj)](https://martinvonz.github.io/jj/latest/), not git.

## Installation

Semblr runs as a [pi coding agent](https://pi.dev) extension.

### Dependencies

- **[pi coding agent](https://pi.dev/install)** — the prerequisite runtime. Install it first, then clone and use Semblr.

### Setup

```bash
# Clone the repository
jj git clone <repo-url> semblr
cd semblr

# Install project dependencies (for indexing scripts)
npm install
```

### Loading the extension

The extension lives at `src/semblr.ts` (not `.pi/extensions/`), so pi does **not** auto-load it. Two ways to activate:

**Temporarily (per session):**
```bash
pi -e ./src/semblr.ts
```

**Permanently (installed as a local pi package):**
```bash
just install        # runs pi install ./src/semblr.ts
```
This registers the extension in your pi settings and loads it on every startup. To remove it later:
```bash
just uninstall      # runs pi remove ./src/semblr.ts
```

## Premise

Current AI agent sessions degrade as they accumulate context. Pi's compaction mechanism summarises past rounds to free memory, but the summaries lose detail. Semblr replaces this with a different approach:

1. **Save every round permanently.** Each user prompt + full assistant response sequence (tool calls, thinking, final answer) is saved as an individual JSON file.
2. **Embed prompt and response.** Both are sent to an embedding API (`text-embedding-3-small`) and stored as vectors in an append-only CSV index.
3. **Retrieve by relevance.** On every user prompt, the prompt is embedded and compared against all stored vectors via cosine similarity. The closest rounds are injected into context — up to a dynamic token budget.
4. **Drill-down via tools.** By default, rounds are shown as a compact numbered index. The LLM uses `get_round_details()` to expand a round and `get_tool_details()` to inspect individual tool calls within it.

The result: context that is **always roughly the same size, always the most relevant, and never lossy** — even across sessions.

## Injected Context Structure

In collapsed mode, Semblr injects historical rounds as three distinct sections in the user prompt (after the system message, before the current conversation). Each section is independently gated — it only appears when its data is non-empty.

### Context building references

This section appears when **any list** is present. It explains the format of what follows and how to use the expansion tools. The lists themselves are self-describing — they carry their own headers and instructions — so the preamble only provides the shared format and tool reference.

*Exact prompt:*

```
[CONTEXT BUILDING REFERENCES]
The lists below show past conversation rounds. Each entry contains only the user prompt — responses and tool calls are collapsed.
Use get_round_details("hash.json") to expand a round's full conversation.
Use get_tool_details("hash.json", N) to inspect tool call N within a round.

Format: N. hash.json [score | N tools]: followed by the full user prompt (indented).
Number 1 in the list is the most recent round.
```

### Recency List

This section appears when the **current session has prior rounds**. It contains the in-memory causal chain — rounds you just discussed. This is the context-building list: use it when the model needs to reconstruct what was said moments ago.

*Exact prompt:*

```
--- RECENCY LIST (current session, newest first) ---
These rounds have n/a scores because they are presented by recency — they form
the immediate conversational context from this session.

IMPORTANT: This list shows ONLY the user's questions from past rounds.
You do NOT have the assistant responses or tool results unless you expand a
round. If you answer based on these prompts alone, you are hallucinating.

Use this list when the current prompt ...:
- ... asks about past work, decisions, code, or findings from prior sessions
- ... is unusually short or lacks clear context/goals/outputs
- ... uses references with no clear antecedent in the causal chain ("that fix",
  "the plan", "where we left off")
- ... asks you to remember, verify, continue, or build upon prior work
- ... requires cross-session continuity (same project, recurring topic,
  long-running task)
- ... is ambiguous: lacks proper context or references, and seems to assume
  knowledge was established

When this happens:
1. Scan the list prompts for relevance. Higher score = stronger match.
2. If a round looks relevant, expand ONLY that round via get_round_details.
3. Stop as soon as the expanded round gives you enough context to answer.
4. If no round looks relevant but the query clearly needs past context,
   use search_interactions.

When NOT to expand:
- The query is fully self-contained (clear context, goals, and outputs present).
- The prompts in the context already provides sufficient information.

Rule: When in doubt, expand. A verification tool call is cheaper than a wrong
answer.

1. abc123.json [n/a | 0 tools]:
  user: Find a budget hotel in Hong Kong under $100/night near Causeway Bay.
---
2. def456.json [n/a | 3 tools]:
  user: Tell me more about the Harbour View Hotel, is it within budget?
---
```

### Relevance List

This section appears when **semantic search returned matches** above the similarity threshold. It contains rounds from all past sessions, ordered by descending similarity. This is immediate recall, not context building — the extension pre-runs `search_interactions` so you don't have to.

*Exact prompt:*

```
--- RELEVANCE LIST (all sessions, by similarity) ---
These rounds have numeric similarity scores (0.0–1.0). Higher = stronger
semantic match. They come from ALL past sessions, not just the current one.

The extension has pre-run a semantic search against your prompt. The results
are below. If something here rings a bell, expand it via get_round_details.
If nothing rings a bell, ignore this list — it's a pre-filter, not a map.

1. ghi789.json [0.37 | 2 tools]:
  user: Look up boutique hotels in the 7th arrondissement of Paris for
  under €150/night.
---
```

### Section presence matrix

| Scenario | Context building refs | Recency List | Relevance List |
|---|---|---|---|
| First message, no semantic matches | omitted | omitted | omitted |
| First message, has semantic matches | shown | omitted | shown |
| Ongoing session, has matches | shown | shown | shown |
| Ongoing session, no matches | shown | shown | omitted |

### Why three sections?

| Section | Source | When shown | Purpose |
|---|---|---|---|
| Context building refs | Static text | If any list exists | Set expectations, explain tools |
| Recency List | Causal chain (in-memory) | If session has prior rounds | Resolve references within the session |
| Relevance List | Semantic vector search | If matches found | Surface cross-session context — pre-made search |

The architecture is intentionally sectioned: new types of context (pinned rounds, compaction summaries, excluded rounds) can be added as additional sections without changing the existing logic. Each section is independently gated by its own condition.

## Cost

Semblr has two sources of API cost:

| Operation | Cost per invocation |
|---|---|
| **Saving a round** | 2 embedding API calls (prompt + response) |
| **Context assembly** | 1 embedding API call (the current prompt) |
| **Index search** (via `search_interactions` tool) | 1 embedding API call per search |

All embeddings go to OpenRouter → `text-embedding-3-small`. At current pricing (~$0.13/1M tokens for input, ~0.26/1M for output for text-embedding-3-small, but OpenRouter may add a small markup), the cost per embedding is on the order of fractions of a cent.

The ongoing cost is ~1 embedding per user prompt.

### Caching tradeoff

Every user prompt is a **separate LLM call** with freshly assembled context. Even though the prompt looks similar to the last round's, the LLM provider sees each as an independent request — so there's **no KV/prompt cache carryover from round to round**. Each round pays full attention-computation cost on the newly injected rounds.

**Within a round**, however, tool calls (like `get_round_details()` or `search_interactions()`) happen in the same LLM invocation — a single streaming conversation with back-and-forth tool turns. The provider's cache persists across those turns, so the cost of repeatedly expanding retrieved rounds from collapsed stubs is mostly in latency, not in re-processing the full context from scratch.

The asymmetry: you pay for full re-processing between rounds, but within a round the tool-based drill-down is comparatively cheap.

## Index & Session Management

Semblr stores conversation data in two areas, both outside the project tree so they survive repository moves:

### Round Storage (`SEMBLR_ROUNDS_DIR`)

| File | Purpose |
|---|---|
| `<id>.json` | A single round: user prompt, full assistant response, tool call metadata |
| `index.csv` | Append-only vector index — one line per embedding: `base64(vector),<filepath>:prompt\|response` |

Round IDs are content-addressed (MD5 of `userPrompt + responseSequence`), so re-indexing is idempotent — same content produces the same file.

### Digest Scripts

Two scripts parse historical pi session files (JSONL format) into semblr rounds:

| Script | What it does |
|---|---|
| `scripts/digest-all.ts` | Iterates all pi session files, deduplicates against already-indexed rounds, embeds new ones in parallel (concurrency: 5) |
| `scripts/digest-session.ts` | Parses a single session file, embeds each round, appends to the vector index |

Both parse the pi session JSONL into `Round` objects containing:
- `userPrompt` — the user's text
- `responseSequence` — the assistant's full text response
- `toolCalls` — structured list of tool invocations (name, arguments, result summary)
- `turnIndex` — position within the session
- `sessionLabel` — source session directory name

### Deduplication

When `digest-all.ts` runs:
1. Loads all existing entries from `index.csv`
2. Computes the expected file path for each parsed round via content hash
3. Skips any round whose file path already appears in the index
4. Only new rounds are sent to the embedding API

This makes it safe to run repeatedly — only unindexed session data gets embedded.

### Session file format

Pi stores session data as JSONL files in its session directory. Each line is a JSON event with a `type` field. Semblr filters for `type: "message"` entries and pairs `user` messages with subsequent `assistant` and `toolResult` messages to reconstruct full rounds.

### Utility commands

```bash
# Bulk-index all historical sessions
just index

# Index a single session file
just digest-session path/to/session.jsonl

# Search the index from the command line
just query "what did we discuss about caching"
```

### Index format

The index is an append-only CSV with no schema header:

```
<base64url(JSON vector)>,<round_id>.json:prompt
<base64url(JSON vector)>,<round_id>.json:response
```

Each round produces two rows: one for the user prompt embedding, one for the assistant response embedding. The vector dimensions match the embedding model (1536 for `text-embedding-3-small`). Cosine similarity is used for retrieval.

## Known Problems

### Most-recent-round context loss (addressed by Recency List)
The Recency List (see [Injected Context Structure](#injected-context-structure)) shows the most recent rounds from the current session with instructions for the model to expand them when resolving references like "those changes" or "it". The list covers all prior rounds in the session, so the special-case last-round injection is no longer needed.

### Embedding API dependency
Semblr requires a working OpenRouter API key (or an alternative embedding endpoint) to function. If the API is unreachable, context assembly falls back to a no-op (no historical context injected). The extension degrades gracefully but silently.

### No local embedding fallback
Currently only one embedding model (`text-embedding-3-small` via OpenRouter) is wired. There is no local embedding option (e.g., `sentence-transformers` → ONNX → TypeScript). Adding one would eliminate the API dependency and cost for index queries.

## Quick Start

Semblr runs when the extension is loaded:

```bash
# Verify it's working
pi -e ./src/semblr.ts
```

Check that the status bar shows `🧠 semblr loaded — N rounds indexed`.

### Bulk-indexing historical sessions

```bash
# Index all historical pi sessions into semblr
just index
```

This parses every JSONL session file in `~/.pi/agent/sessions/`, deduplicates against already-indexed rounds, and embeds new ones in parallel (concurrency: 5).

### Query the index

```bash
# Search the index from the command line
just query "what did we discuss about caching"
```

## Project Structure

```
├── src                         # The extension directory
│   └── semblr.ts                 # Main extension file (~1240 lines)
├── scripts/
│   ├── digest-all.ts             # Bulk-embed all historical sessions
│   ├── digest-session.ts         # Embed a single session file
│   └── test-register-tool.ts     # Tool registration test harness
├── docs/                         # (empty — extended docs not yet written)
├── VISION.md                     # Full project vision, architecture, roadmap
├── AGENTS.md                     # Project context for AI agents
├── README.md                     # This file
├── justfile                      # just command recipes
└── package.json                  # Project metadata
```

Rounds are stored in a global directory (outside the project tree) so they survive repository moves and clones.

## VISION.md

For the full vision, design principles, architecture docs, and roadmap (with implemented items checked off), see [VISION.md](VISION.md).

## Tools Reference

Semblr registers three tools that the LLM can call to explore historical rounds. They're the interface between the collapsed index and the full conversation history.

### `search_interactions`

Searches all past conversations by semantic similarity.

| Parameter | Type | Description |
|---|---|---|
| `query` | string | What to find — natural language, not keywords |
| `minSimilarity` | number (0–1) | Minimum similarity threshold. Default 0.25. Lower for broader matches |
| `turns` | string[] | Optional — scope search to specific round files (drill into compaction summaries) |

```
# The LLM calls this to find past discussions
search_interactions(query: "why did we pick PostgreSQL over MySQL")
```

Returns a list of matching round IDs with scores and previews.

### `get_round_details`

Retrieves the full content of a specific round — user prompt, full response, and all tool call metadata (with collapsed arguments).

| Parameter | Type | Description |
|---|---|---|
| `round` | string | The round filename, e.g. `"a1b2c3d4.json"` |

This is how the LLM expands a numbered index entry into the full conversation turn, including text it hasn't seen yet.

### `get_tool_details`

Expands a single collapsed tool call from a historical round — full arguments and complete result.

| Parameter | Type | Description |
|---|---|---|
| `round` | string | The round filename |
| `index` | number | 0-based position in the round's toolCalls array |

This is what "drill down" means: from a stub like `Turn 2: read — [REDACTED: ...]`, the LLM calls `get_tool_details("a1b2c3d4.json", 2)` and sees exactly what file was read and what it contained.

## License

MIT
