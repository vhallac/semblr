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

## Development Methodology

For implementation tasks, use code-and-test-together development.

1. **Understand**
   - Restate the required behavior.
   - Identify behavioral anchors: docs, fixtures, CLI output, logs, existing behavior, user examples.
   - Surface assumptions before changing code.

2. **Plan code and tests together**
   - Identify production changes.
   - Identify tests that protect required and affected existing behavior.
   - Refactor only as needed to expose test seams; preserve behavior.

3. **Implement first pass**
   - Write production code and matching tests together.
   - Tests MUST check externally anchored behavior, not mirror implementation.
   - Prefer focused regression tests.

4. **Validate externally**
   - Run `npm run verify` (typecheck + lint + knip + test:coverage with 80% threshold).
   - Passing tests alone are insufficient if the requirement was not checked against an anchor.

5. **Diagnose failures before fixing**
   - Classify each failure by source:
     - **code:** wrong behavior, edge case, integration
     - **test:** wrong expectation, setup, fixture, harness
     - **mechanical:** syntax, type, lint, formatting
     - **requirement:** ambiguous, contradicted by existing behavior
   - Fix the diagnosed source.
   - Mechanical fixes may be direct.
   - NEVER weaken tests merely to pass; changed expectations need an anchor.

6. **Harden**
   - Add branch, edge-case, and regression tests after the main behavior works.
   - For risky logic, use targeted mutation tests.

7. **Update README.md if needed**
   - Update it for changed commands, env vars, structure, context format, gating, or user-visible behavior.
   - Verify sync before marking complete.

## Attributions

- When committing code that is entirely written by you, add

🤖 LLM authored

- When committing code that is partially written by you (50% or less), and the rest is written by a human, add

🤖 LLM assisted

- When writing code forge pull requests, comments, issues, or contributing to discussions, add

🤖 Content created by LLM

as the last line of the text.

When only creating commit messages to code fully written by a human; do not add an LLM attibution.
