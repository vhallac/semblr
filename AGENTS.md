# Semblr

Next-generation context management for AI agents. See [VISION.md](VISION.md) for the full vision, architecture, and roadmap with completion status.

## The Extension

The extension lives at `src/semblr.ts` (moved from `.pi/extensions/semblr.ts` to avoid pi auto-loading). It:

- Saves every completed conversation round to a persistent round repository (`rounds/<hash>.json`)
- Embeds combined prompt+response vectors to `index.csv` (via OpenRouter, `text-embedding-3-small`, clipped to ~8K tokens)
- On every user prompt, retrieves the most semantically similar rounds and injects them into context
- Registers three native tools: `search_interactions`, `get_round_details`, `get_tool_details`
- Integrates with pi's compaction system (captures summaries as rounds with referenced turn pointers)
- Runs in two modes: `collapsed` (default, compact numbered index) or `full` (complete round text)



## Project Structure

- `VISION.md` — project vision, architecture, roadmap (with ✅ completion markers)
- `README.md` — project overview, premise, cost, known problems
- `AGENTS.md` — this file (project context for AI agents)
- `src/semblr.ts` — the extension (single file)
- `scripts/digest-all.ts` — bulk-embed all historical pi sessions
- `scripts/digest-session.ts` — embed a single session JSONL file
- `scripts/import-claude-code.ts` — import Claude Code JSONL history
- `scripts/migrate-content-hash.ts` — content-hash-based round migration
- `scripts/migrate-rounds.ts` — old round migration (pre-content-hash)
- `justfile` — command recipes (index, digest-session, query, import-claude, migrate)
- `rounds/` — round repository (stored outside the project tree, survives moves and clones)

## Keeping README up to date

When adding a new feature or making a significant change, the README must be updated to reflect it. Before marking a task complete, verify the following are in sync:

- **Commands** — any new `/semblr:*` commands must appear in the Commands section
- **Environment variables** — any new env var must appear in the Environment Variables table
- **Project structure** — any new script or directory must appear in the tree
- **Section presence matrix** — any new gating condition (e.g., short-prompt suppression) must be added
- **Injected Context sections** — any change to the format of recency list, relevance list, or preamble must be reflected in the exact-prompt examples

This section exists next to the A in AGENTS.md so agents read it before making changes.

## Attributions

- When committing code that is entirely written by you, add

🤖 LLM authored

- When committing code that is partially written by you (50% or less), and the rest is written by a human, add

🤖 LLM assisted

- When writing code forge pull requests, comments, issues, or contributing to discussions, add

🤖 Content created by LLM

as the last line of the text.

When only creating commit messages to code fully written by a human; do not add an LLM attibution.
