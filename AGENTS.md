# Semblr

Next-generation context management for AI agents. See [VISION.md](VISION.md) for the full vision, architecture, and roadmap with completion status.

## The Extension

The extension lives at `src/semblr.ts` (~1240 lines, moved from `.pi/extensions/semblr.ts` to avoid pi auto-loading). It:

- Saves every completed conversation round to a persistent round repository (`rounds/<hash>.json`)
- Embeds prompt and response vectors to `index.csv` (via OpenRouter, `text-embedding-3-small`)
- On every user prompt, retrieves the most semantically similar rounds and injects them into context
- Registers three native tools: `search_interactions`, `get_round_details`, `get_tool_details`
- Integrates with pi's compaction system (captures summaries as rounds with referenced turn pointers)
- Runs in two modes: `collapsed` (default, compact numbered index) or `full` (complete round text)



## Project Structure

- `VISION.md` — project vision, architecture, roadmap (with ✅ completion markers)
- `README.md` — project overview, premise, cost, known problems
- `AGENTS.md` — this file (project context for AI agents)
- `src/semblr.ts` — the extension (single file, ~1240 lines)
- `scripts/digest-all.ts` — bulk-embed all historical pi sessions
- `scripts/digest-session.ts` — embed a single session JSONL file
- `justfile` — command recipes (index, digest-session, query)
- `rounds/` — round repository (stored outside the project tree, survives moves and clones)

## Attributions

- When committing code that is entirely written by you, add

🤖 LLM authored

- When committing code that is partially written by you (50% or less), and the rest is written by a human, add

🤖 LLM assisted

- When writing code forge pull requests, comments, issues, or contributing to discussions, add

🤖 Content created by LLM

as the last line of the text.

When only creating commit messages to code fully written by a human; do not add an LLM attibution.
