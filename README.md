# Semblr

**Semantic context assembly for AI agents — experimental.**

> **⚠️  Experimental.** We are actively testing the core claims below. They are hypotheses, not settled facts. Everything here is subject to change as we gather data.

Semblr stores every conversation round permanently, embeds it, and retrieves the most relevant rounds on each user prompt — by meaning, not by recency. We want to find out whether semantic retrieval beats lossy summarization in practice.

Runs as a [pi coding agent](https://pi.dev) extension at `src`.


## Installation

Semblr runs as a [pi coding agent](https://pi.dev) extension.

### Dependencies

#### Runtime

- **[pi coding agent](https://pi.dev/install)** — the prerequisite runtime. Must be installed before using Semblr.
- **[Node.js](https://nodejs.org/)** >= 22 — required by the extension runtime and scripts.
- **Embedding provider API key** — by default Semblr uses pi's `openrouter` provider with `openai/text-embedding-3-small`; configure the provider/API key through pi, or set a Semblr embedding endpoint override.

#### Development

- **[Node.js](https://nodejs.org/)** >= 22 with npm
- **TypeScript** (`typescript`) — type checking the extension and scripts
- **Biome** (`@biomejs/biome`) — linting and formatting (version 1.9.x; installed via npm or Nix)
- **`@earendil-works/pi-coding-agent`** — pi SDK types for type checking
- **`@types/node`** — Node.js type declarations

Install dev dependencies:

```bash
npm install
```

### Setup

```bash
# Clone the repository
git clone <repo-url> semblr
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

### Commands

When the extension is loaded, pi exposes:

```text
/semblr:import-claude [--dry-run] [--include-sidechains] [--limit N]
/semblr:recent-read-stats
```

- `/semblr:import-claude` — imports Claude Code JSONL history from `~/.claude/projects` into the shared Semblr round repository and vector index.
- `/semblr:recent-read-stats` — displays detailed grouping and chain-read statistics for the current session.

## Premise

Current AI agent sessions degrade as they accumulate context. Pi's compaction mechanism summarises past rounds to free memory, but the summaries lose detail. Semblr replaces this with a different approach:

1. **Save every round permanently.** Each user prompt + full assistant response sequence (tool calls, thinking, final answer) is saved as an individual JSON file.
2. **Embed prompt, response, and combined text.** Three texts are sent to the configured embedding API: the user prompt, the clipped response (by default truncated to ~24KB, context-injection artifacts stripped), and the concatenation of both (`prompt + "\n\n" + clippedResponse`). The prompt and response vectors are stored in an append-only CSV index. The combined vector is stored in the round file for semantic grouping.
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

Format: [index: N] hash.json [score | N tools | size]: followed by the full user prompt (indented).
```

### Recency List

This section appears when the **current session has prior rounds**. It contains the in-memory causal chain — rounds you just discussed. This is the context-building list: use it when the model needs to reconstruct what was said moments ago.

*Exact prompt:*

```
--- RECENCY LIST (current session, by topic) ---
These rounds have n/a scores because they are presented by recency — they form
the immediate conversational context from this session.

IMPORTANT: This list shows ONLY the user's questions from past rounds.
You do NOT have the assistant responses or tool results unless you expand a
round. If you answer based on these prompts alone, you are hallucinating.

The groups below are recent messages that are likely to be related to the same
topic. Lower numbered indices in groups are more recent conversations.

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

**Group 1**

- [index: 1] abc123.json [n/a | 0 tools]:
  user: Find a budget hotel in Hong Kong under $100/night near Causeway Bay.
  ---

---

**Group 2**

- [index: 2] def456.json [n/a | 3 tools]:
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

- [index: 1] ghi789.json [0.37 | 2 tools]:
  user: Look up boutique hotels in the 7th arrondissement of Paris for
  under €150/night.
  ---
```

### Section presence matrix

| Scenario | Context building refs | Recency List | Relevance List | Final response contract | Actionable prompt marker |
|---|---|---|---|---|---|
| First message, no semantic matches | omitted | omitted | omitted | Always | Always |
| First message, has semantic matches | shown | omitted | shown | Always | Always |
| Ongoing session, has matches | shown | shown | shown | Always | Always |
| Ongoing session, no matches | shown | shown | omitted | Always | Always |
| Short prompt (< 20 words) | shown (if any list) | shown (if prior rounds) | omitted | Always | Always |
| `DROP_RELEVANCE_LIST=true` | shown (if any list) | shown (if prior rounds) | omitted | Always | Always |

### Why these sections?

| Section | Source | When shown | Purpose |
|---|---|---|---|
| Context building refs | Static text | If any list exists | Set expectations, explain tools |
| Recency List | Causal chain (in-memory) | If session has prior rounds | Resolve references within the session |
| Relevance List | Semantic vector search | If matches found | Surface cross-session context — pre-made search |
| Previous Round Follow-up | Last round file | If last round was flagged | Show full previous round when the LLM asked a question |
| Final response contract | Static text | Always | Tell the LLM when and how to emit `round_needs_followup` |
| Actionable prompt marker | Current user message | Always | Mark the real prompt the contract applies to |

The architecture is intentionally sectioned: new types of context (pinned rounds, compaction summaries, excluded rounds) can be added as additional sections without changing the existing logic. Each section is independently gated by its own condition.

### Follow-up flagging

When the LLM appends `round_needs_followup` as the very last line of its output (per the final response contract), the extension:

1. **At agent_end**: Strips the marker from the saved response text, sets `needsFollowup: true` in the round JSON.
2. **At context (next round)**: Checks the previous round for `needsFollowup: true`. If found, injects a `[PREVIOUS ROUND FOLLOW-UP]` section with the full round content, then clears the flag.
3. **Auto-grouping**: When a round has `needsFollowup: true`, the next round is automatically assigned to the same topic group, bypassing semantic similarity matching. The rationale: if the LLM asked a question and the user responded, those two rounds are definitively the same topic.

This gives the LLM explicit recall of what question was asked, without bloating context on every turn.

## Cost

Semblr has two sources of API cost:

| Operation | Cost per invocation |
|---|---|
| **Saving a round** | 2–3 embedding API calls (prompt, clipped response, combined) |
| **Context assembly** | 1 embedding API call (the current prompt) |
| **Index search** (via `search_interactions` tool) | 1 embedding API call per search |

By default, embeddings go to OpenRouter → `openai/text-embedding-3-small`. If you configure another embedding provider/model or `embeddingApiUrl`, pricing and vector dimensions follow that endpoint/model instead.

The ongoing cost is ~2–3 embeddings per user prompt (the prompt embedding from context assembly is reused for saving, so the net cost is one response embedding plus one combined embedding).

### Caching

#### Cross-round vs cross-turn tradeoff

Every user prompt is a **separate LLM call** with freshly assembled context. Even though the prompt looks similar to the last round's, the LLM provider sees each as an independent request — so there's **no KV/prompt cache carryover from round to round**. Each round pays full attention-computation cost on the newly injected rounds.

**Within a round**, however, tool calls (like `get_round_details()` or `search_interactions()`) happen in the same LLM invocation — a single streaming conversation with back-and-forth tool turns. The provider's cache persists across those turns, so the cost of repeatedly expanding retrieved rounds from collapsed stubs is mostly in latency, not in re-processing the full context from scratch.

The asymmetry: you pay for full re-processing between rounds, but within a round the tool-based drill-down is comparatively cheap.

#### Context stability fix

Previously, every tool turn within a cycle would re-construct the `[ENVIRONMENT]` preamble and context-building blocks. This meant timestamps changed between turns, **busting the LLM prompt cache** on every tool call — eliminating the cross-turn caching benefit entirely.

This was fixed by caching the assembled preamble and context blocks once per invocation cycle. The timestamps are frozen at the start of the round, so all tool turns within it see identical prompt text. The cache is snapshot-copied with spread to prevent cross-turn contamination from accidental mutation.

## Index & Session Management

Semblr stores conversation data in two areas, both outside the project tree so they survive repository moves:

### Round Storage (`semblr.roundsDir` / `SEMBLR_ROUNDS_DIR`)

| File | Purpose |
|---|---|
| `<id>.json` | A single round: user prompt, full assistant response, tool call metadata |
| `index.csv` | Append-only vector index — one line per embedding: `base64(vector),<filepath>:prompt|response,<model>` |

Round IDs are content-addressed (MD5 of `userPrompt + responseSequence`), so re-indexing is idempotent — same content produces the same file.

### Configuration

Semblr reads a `semblr` section from pi settings. Values are resolved per key in this order:

1. Environment variable
2. Project `.pi/settings.json`
3. Global `$PI_CODING_AGENT_DIR/settings.json` (default `~/.pi/agent/settings.json`)
4. Hardcoded default

Example:

```json
{
  "semblr": {
    "embeddingProvider": "openrouter",
    "embeddingModel": "openai/text-embedding-3-small",
    "embeddingMaxTokens": 8000,
    "roundsDir": "semblr/rounds",
    "groupThreshold": 0.77,
    "minSimilarity": 0.3,
    "embedTimeoutMs": 15000,
    "embedMaxRetries": 3,
    "embedBackoffMs": 1000
  }
}
```

`embeddingProvider` is a reference to pi's provider registry (`models.json` + auth storage). Semblr asks pi for the provider base URL and API key, then sends OpenAI-compatible embedding requests. If `embeddingApiUrl` is set, it is used as a full embeddings endpoint override instead of deriving `<provider baseUrl>/v1/embeddings`.

Relative `roundsDir` values in project settings resolve under the project cwd. Relative `roundsDir` values in global settings resolve under `$PI_CODING_AGENT_DIR` (default `~/.pi/agent`).

| Setting | Env var | Default | Purpose |
|---|---|---|---|
| `embeddingProvider` | `SEMBLR_EMBEDDING_PROVIDER` | `openrouter` | pi provider name used for embedding API key/base URL lookup |
| `embeddingModel` | `SEMBLR_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Model identifier sent in embedding request bodies and written to index rows |
| `embeddingMaxTokens` | `SEMBLR_EMBEDDING_MAX_TOKENS` | `8000` | Prompt/response clipping budget for embedding inputs |
| `embeddingApiUrl` | `SEMBLR_EMBEDDING_API_URL` | derived | Full embeddings endpoint override |
| `roundsDir` | `SEMBLR_ROUNDS_DIR` | `$PI_CODING_AGENT_DIR/semblr/rounds` | Round repository and `index.csv` directory |
| `groupThreshold` | `SEMBLR_GROUP_THRESHOLD` | `0.77` | Minimum cosine similarity for grouping rounds into topics. Higher = more groups |
| `minSimilarity` | `SEMBLR_MIN_SIMILARITY` | `0.3` | Minimum semantic similarity for relevance retrieval |
| `embedTimeoutMs` | `SEMBLR_EMBED_TIMEOUT` | `15000` | Embedding request timeout |
| `embedMaxRetries` | `SEMBLR_EMBED_RETRIES` | `3` | Embedding request retry count |
| `embedBackoffMs` | `SEMBLR_EMBED_BACKOFF` | `1000` | Base retry backoff in milliseconds |

Additional runtime-only switches:

| Variable | Default | Purpose |
|---|---|---|
| `RELEVANCE_LIST_MIN_WORDS` | `20` | Raw word count threshold. Prompts shorter than this skip the relevance list entirely |
| `DROP_RELEVANCE_LIST` | not set | If `1` or `true`, the relevance list section is always suppressed regardless of matches |

### Digest Scripts

Three scripts parse historical conversation data into semblr rounds:

| Script | What it does |
|---|---|
| `scripts/digest-all.ts` | Iterates all pi session files, deduplicates against already-indexed rounds, embeds new ones in parallel (concurrency: 5) |
| `scripts/digest-session.ts` | Parses a single session file, embeds each round, appends to the vector index |
| `scripts/import-claude-code.ts` | Imports Claude Code JSONL history from `~/.claude/projects` into the shared index |

`digest-all.ts` and `digest-session.ts` parse the pi session JSONL into `Round` objects containing:
- `userPrompt` — the user's text
- `responseSequence` — the assistant's full text response
- `toolCalls` — structured list of tool invocations (name, arguments, result summary)
- `turnIndex` — position within the session
- `sessionLabel` — source session directory name

### Deduplication

When `digest-all.ts` runs:
1. Loads all existing entries from `index.csv`
2. Computes the expected file path for each parsed round via content hash
3. Skips rounds already indexed with the current embedding model
4. Re-embeds rounds whose index rows have an explicit different embedding model
5. Only new or model-stale rounds are sent to the embedding API

This makes it safe to run repeatedly — only unindexed or model-stale session data gets embedded. Legacy two-column rows without a model column are treated as current-model rows; run `just migrate` to stamp them with the configured model.

### Session file format

Pi stores session data as JSONL files in its session directory. Each line is a JSON event with a `type` field. Semblr filters for `type: "message"` entries and pairs `user` messages with subsequent `assistant` and `toolResult` messages to reconstruct full rounds.

### Utility commands

```bash
# Bulk-index all historical sessions
just index

# Index a single session file
just digest-session path/to/session.jsonl

# Import Claude Code JSONL history into the shared index
just import-claude
#   --dry-run, --include-sidechains, --limit N

# Run all pending round migrations (idempotent)
just migrate

# Search the index from the command line
just query "what did we discuss about caching"
```

### Index format

The index is a CSV with no schema header:

```
<base64url(JSON vector)>,<round_id>.json:prompt,<embedding-model>
<base64url(JSON vector)>,<round_id>.json:response,<embedding-model>
```

Legacy two-column rows without the model column are still readable and are assumed to use the current configured embedding model. Each round produces two rows: one for the prompt embedding and one for the clipped-response embedding. The combined prompt+response embedding is stored in the round JSON file for grouping, not in the index CSV. The vector dimensions match the configured embedding model (1536 for the default `openai/text-embedding-3-small`). `just index` rewrites rows for rounds that were explicitly embedded with a different model so the index converges to the configured model. Cosine similarity is used for retrieval.

## Known Problems

### Most-recent-round context loss (addressed by Recency List)
The Recency List (see [Injected Context Structure](#injected-context-structure)) shows the most recent rounds from the current session with instructions for the model to expand them when resolving references like "those changes" or "it". The list covers all prior rounds in the session, so the special-case last-round injection is no longer needed.

### Embedding API dependency
Semblr requires a working embedding provider/API key to function. By default it uses pi's `openrouter` provider with `openai/text-embedding-3-small`, but you can configure another pi provider/model or set `embeddingApiUrl` as a full endpoint override. If the API is unreachable, context assembly falls back to a no-op (no historical context injected). The extension degrades gracefully but silently.

### No local embedding fallback
Semblr currently sends OpenAI-compatible embedding requests to a configured HTTP endpoint. There is no local embedding option (e.g., `sentence-transformers` → ONNX → TypeScript). Adding one would eliminate the API dependency and cost for index queries.

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
│   └── semblr.ts                 # Main extension orchestration / pi lifecycle glue
├── lib/                           # Domain helpers used by the extension, scripts, and tests
│   ├── context-messages.ts        # User-message preparation and context prefix assembly
│   ├── embedding-client.ts        # Provider-aware embedding client
│   ├── index-storage.ts           # Runtime vector-index loading and locked appends
│   ├── message-content.ts         # Shared text extraction from message content blocks
│   ├── round-capture.ts           # agent_end/message_end round capture helpers
│   ├── round-data.ts              # Shared round/tool-call data contracts
│   ├── round-tool-results.ts      # get_round_details/get_tool_details result rendering
│   ├── script-config.ts           # Shared config/auth setup for scripts
│   ├── search-interactions.ts     # search_interactions scoring, selection, and rendering
│   └── semblr-config.ts           # Semblr settings/env resolution
├── scripts/
│   ├── digest-all.ts               # Bulk-embed all historical sessions
│   ├── digest-session.ts           # Embed a single session file
│   ├── import-claude-code.ts       # Import Claude Code JSONL history
│   ├── migrate-content-hash.ts     # Content-hash-based round migration
│   ├── migrate-rounds.ts           # Old round migration (pre-content-hash)
│   └── test-register-tool.ts       # Tool registration test harness
├── docs/                         # (empty — extended docs not yet written)
├── VISION.md                     # Full project vision, architecture, roadmap
├── AGENTS.md                     # Project context for AI agents
├── README.md                     # This file
├── justfile                      # just command recipes
└── package.json                  # Project metadata
```

Rounds are stored in a global directory (outside the project tree) so they survive repository moves and clones.

## Developer's Guide

### Linting and type checking

```bash
# Type check (TypeScript strict mode)
npm run typecheck

# Lint + format check
npm run lint

# Auto-fix lint and format issues
npm run lint:fix

# Run both type check and lint
npm run check
```

Configuration:
- `tsconfig.json` — TypeScript with strict mode, ES2022 target, Node16 modules
- `biome.json` — Biome linter and formatter (recommended rules, tabs, 120 char width)

> **NixOS note:** The npm-installed Biome binary is dynamically linked and won't run. Use the nix-provided one via `nix-shell -p biome --command "biome check"` or add it to `shell.nix`.

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
