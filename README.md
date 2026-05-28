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

The extension lives at `src` (not `.pi/extensions/`), so pi does **not** auto-load it. Two ways to activate:

**Temporarily (per session):**
```bash
pi -e ./src
# or via the re-export entry point:
pi -e ./index.ts
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

## Context Injection Architecture

When Semblr retrieves relevant rounds, it injects them into the LLM's context. By default it uses **collapsed mode** — a compact numbered index rather than full text. This keeps context size predictable and stops historical rounds from drowning out the current task.

### Why collapse?

Every round contains the full assistant response — tool calls, thinking blocks, final answer. That's a lot of tokens, especially when the LLM might only need the gist to resolve a reference. Collapsing trades convenience for token efficiency: the LLM sees a short entry for each round and can **drill down on demand** using the provided tools.

### Collapsed mode (default)

Historical rounds appear as a numbered index in the system prompt:

```
1. a1b2c3d4.json [0.81 | 3 tools]: Why did we decide on PostgreSQL over MySQL?
2. e5f6g7h8.json [0.64 | 0 tools]: Let's review the schema for the orders table
```

Each line shows the round's file ID, similarity score, tool count, and the full user prompt. The LLM never sees historical response text unless it expands.

Every tool call inside a historical round is **redacted** into a stub:

> `Turn 2: read — [REDACTED: arguments and result collapsed. Use get_tool_details("a1b2c3d4.json", 2) to expand.]`

To see what actually happened, the model calls `get_round_details()` or `get_tool_details()` — like opening a file or reading a log.

### Full mode

Pass `semblr_mode: full` in your prompt. Retrieved rounds are injected as complete conversation text — user prompt, full response, everything. More tokens, zero tool calls needed. Useful when you're digging into a specific past thread and want the whole picture at once.

### Recency buffer (experimental)

We hypothesise that a **recency buffer** — keeping the last N rounds in full, outside the indexed retrieval logic — is the right solution for bridging the referential gap (resolving backreferences like "those changes" or "as I said"). To test this cleanly, we currently set `DROP_LAST_ROUND = true`, which suppresses the special-case last-round injection entirely. This isolates the recency buffer effect so we can measure its impact without confounding variables.

Future work: determine the optimal buffer size (N). Current hypothesis is 3–5 rounds.

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

### Most-recent-round context loss (collapsed mode, being tested)
In collapsed mode (the default), every retrieved round — including the immediately preceding conversation turn — appears as a compact numbered entry. The LLM cannot directly resolve "those changes" or "it" from the previous round without calling `get_round_details()`. We are experimenting with a **recency buffer** (keeping the last 3–5 rounds in full) as the solution. To isolate its effect, we currently set `DROP_LAST_ROUND = true`, which removes the special-case last-round injection.

**Ongoing experiment:** Does a recency buffer completely eliminate referential ambiguity? Or do we still need the special-case last-round injection? We'll gather data and decide.

### Embedding API dependency
Semblr requires a working OpenRouter API key (or an alternative embedding endpoint) to function. If the API is unreachable, context assembly falls back to a no-op (no historical context injected). The extension degrades gracefully but silently.

### No local embedding fallback
Currently only one embedding model (`text-embedding-3-small` via OpenRouter) is wired. There is no local embedding option (e.g., `sentence-transformers` → ONNX → TypeScript). Adding one would eliminate the API dependency and cost for index queries.

## Quick Start

Semblr runs when the extension is loaded:

```bash
# Verify it's working
pi -e ./src
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
├── src                         # The extension (~1240 lines)
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
