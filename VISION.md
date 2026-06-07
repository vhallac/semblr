# Semblr — Project Vision

## Elevator Pitch

Current AI agent sessions are reborn every time you start one. Every chat, every context, every insight — gone. Within a session, context decays through lossy summarization the moment you exceed the window. Semblr replaces this with **semantic context assembly**: rounds are stored permanently as individual files, embedded, and retrieved by relevance — not recency. The context is always roughly the same size, always the most relevant to what you're working on.

## Core Problems Solved

### 1. Session Amnesia
Every new session is tabula rasa. Prior work, decisions, dead ends — all lost.

**Solution:** A persistent repository of rounds (each user prompt + model response sequence). Embedded and retrievable by semantic similarity.

### 2. Context Decay
Within a session, as context grows past the window, summarization compresses it. What survives is a rough sketch, not the details.

**Solution:** Dynamic context assembly from the round repository. Each round gets a fixed budget. Always the most relevant rounds, never lossy summarization. Additionally, compaction summaries are indexed as rounds with references to the original turns, turning compaction from data-loss into index compression.



## Architecture

### Round Repository
- Each round is saved as an individual file under a `rounds/<hash>.json` directory
- A "round" = one user message → full model response sequence (thinking, tool calls, tool results, final answer, etc.)
- Files are append-only, never modified after creation
- Deduplication by MD5 content hash

### Embedding Index
- Each round gets two embeddings: prompt vector + response vector
- Stored as an append-only CSV: `base64url(vector_json),<filepath>:prompt|:response`
- Embedding provider/model: configurable through Semblr settings; default is pi's `openrouter` provider with `openai/text-embedding-3-small`
- Bulk indexing scripts in `scripts/` for historical sessions, using the same config resolution as the extension

### Context Assembly (on each prompt)
1. Embed the incoming user prompt
2. Compute cosine similarity against all stored vectors
3. Sort by distance (descending — closest first)
4. Pull in rounds above a dynamic similarity threshold until the token budget is reached
5. Construct context: system prompt + enriched environment + retrieved rounds + last round (pinned) + current messages

**Two injection modes:**
- **Collapsed (default):** Injects a compact numbered index of retrieved rounds. The LLM calls `get_round_details()` / `get_tool_details()` to expand individual rounds. Massively reduces token overhead while preserving full granularity.
- **Full:** Injects complete historical round text. Rich enough for the model to work without tools, but expensive.

### Context Budget
- A percentage of the model's max context size (default 50%)
- Dynamic interpolation: at configured `minSimilarity` (default `0.30`) the budget is 2,000 tokens; at 1.0 it's 50% of the context window
- Room reserved for system prompt, current prompt, and model response

### Three-Section Context Injection
The context block is assembled from three independently gated sections:

1. **Context Building References** (preamble) — explains the entry format, the expansion tools, and names the two lists below.
2. **Recency List** — rounds from the current session's in-memory causal chain. Context building: the model expands entries to resolve references like "that fix", "the plan", "where we left off". Trigger cases and expand-on-demand rules are embedded in the section heading.
3. **Relevance List** — semantically similar rounds from all past sessions. Immediate recall: the extension pre-runs `search_interactions`. Scan for a bell, or ignore.

Each section is gated independently: the preamble only shows if any list exists, the recency list only if the current session has prior rounds, the relevance list only if semantic search returned matches.

**Design decisions:**
- **Simple recency:** No causality detection. Rounds are added in temporal order as they occur. Pending a future mechanism to trace parallel/divergent chains.
- **No deduplication with semantic index:** Rounds that also appear in the semantic retrieval index are included in both sections. The duplication is intentional — the score contrast (`n/a` vs a number) is a signal the LLM can leverage.
- **No truncation:** All consecutive rounds in the session are included. No limit on buffer size for now.
- **Flush on session start:** The buffer is cleared at `session_start`. No attempt is made to re-establish chains across session boundaries. This is consistent with the simple-recency approach — chain persistence would require causality metadata.
- **Collapsed-only:** Full mode was removed. All injection uses the compact numbered-list format. Use `get_round_details()` to expand.

### Debug/Quality Logging
- Each context construction is logged to the TUI status bar: which files selected, token usage, indexed count
- Enables real-time review and improvement of retrieval quality

## Known Problems

### 1. Most-recent-round context loss (addressed)
The Recency List (see [Three-Section Context Injection](#three-section-context-injection)) contains all prior rounds from the current session with trigger-case instructions and expand-on-demand rules. The model resolves "it", "that", "those changes" by expanding the relevant recency entry.

**Known limitation — causality vs recency:** Not all consecutive rounds form a causal chain. The current implementation uses simple recency. True causality discovery (distinguishing parallel vs sequential chains, detecting topic boundaries) is deferred to future research.

**Future direction:** The buffer is non-persistent — flushed on `session_start` with no attempt to re-establish chains on session resume.

### 2. Embedding cost
Every round saved costs 2 embedding API calls (prompt + response). Every context assembly costs 1 embedding call. Embedding costs are the main operational expense.

### 3. Cache misses
Because context is assembled dynamically, every prompt is a fresh embedding API call (cannot be cached). This is inherent to the approach — you can't predict the cache key when every prompt generates a novel query.

## Technology

- **Platform:** [pi coding agent](https://pi.dev) — extensions
- **Agent loop, tools, TUI, model abstraction:** Inherited from pi
- **Round repository:** Flat files on disk under a `rounds/` directory
- **Embeddings:** Configurable OpenAI-compatible embedding provider/model; default `openrouter` + `openai/text-embedding-3-small`
- **Vector index:** Flat CSV + cosine similarity in TypeScript
- **Context assembly:** Extension hooks (`agent_start`, `agent_end`, `message_end`, `context`, `session_before_compact`, `session_compact`, `session_start`)
- **Native tools:** `search_interactions`, `get_round_details`, `get_tool_details` (registered in `session_start`)

## Design Principles

1. **Semantic over sequential.** Relevance beats recency.
2. **Fixed-size context.** Predictable, cache-friendly windows.
3. **Append-only repository.** Never modify a saved round. Build better retrieval instead.
4. **Observable retrieval.** Log every context construction for quality iteration.
5. **Pluggable embeddings.** Multiple vector strategies over time.
6. **Framework-lean.** Own the context logic. Borrow the agent loop.

## Roadmap

### ✅ Phase 1 — MVP (Proof that extension hooks work)
- [x] Pi extension that wipes context clean on every round ("total amnesia")
- [x] Verify: model only sees current prompt, no prior conversation
- [x] Verify: no compaction fires
- [x] Verify: tools still work
- [x] Verify: TUI still works

*Note: The amnesia extension was superseded by the real semblr extension. The concept was validated and then replaced.*

### ✅ Phase 2 — Round Repository
- [x] Save each completed round to a file on disk
- [x] Structure: `rounds/<id>.json` with prompt, response sequence, timestamps
- [x] Embed prompt and response separately
- [x] Maintain vector index file

### ⏲ Phase 3 — Retrieval
- [x] On `context` hook, embed incoming prompt
- [x] Query index by distance
- [x] Assemble context from closest rounds up to token budget
- [x] Inject into the agent as replaced messages
- [x] Recency buffer — implemented as the [Recency List](#three-section-context-injection). Flushed on session start. No truncation. No causality detection yet.

### Phase 4 — Quality & Iteration
- [x] Log context construction decisions (status bar)
- [x] Experiment with different embedding models (provider/model now configurable; benchmarking remains future work)
- [ ] Experiment with prompt vs response vector weighting
- [ ] Measure retrieval quality (precision/recall against manual ideal)
- [ ] Explore collapse modes: variable collapse threshold by round score
- [ ] Expose retrieval quality metrics in TUI

### Phase 5 — Advanced
- [ ] Multiple embedding strategies per round
- [ ] Hybrid retrieval (semantic + keyword/BM25)
- [ ] User-directed context curation (exclude, pin, boost)
- [ ] Cross-project round repository sharing
- [ ] Embedding model benchmarking & automatic selection
