# ── semblr ─────────────────────────────────────────────────────────────────
# See VISION.md for architecture and AGENTS.md for project context.

# Configuration is read from Semblr settings/env. For one-off rounds-dir overrides:
#   just --set SEMBLR_ROUNDS_DIR /custom/path index
# or set SEMBLR_ROUNDS_DIR in the environment.
SEMBLR_ROUNDS_DIR := env_var_or_default("SEMBLR_ROUNDS_DIR", "")

# Run the local verification suite
verify:
    npm run verify

# Run mutation testing against src/core/
# Slow — runs ~8 minutes, tests thousands of mutants
mutate:
    npx stryker run

# Index all new turns from pi session files
# Skips turns already indexed (by MD5 content hash)
index:
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$(pass show ai/openrouter 2>/dev/null || true)}" \
        SEMBLR_ROUNDS_DIR="{{SEMBLR_ROUNDS_DIR}}" \
        npx tsx scripts/digest-all.ts

# Digest a specific session file
# Usage: just digest-session <path-to-jsonl>
digest-session path:
    SEMBLR_ROUNDS_DIR="{{SEMBLR_ROUNDS_DIR}}" \
        npx tsx scripts/digest-session.ts {{path}}

# Import Claude Code history from ~/.claude/projects into the same Semblr index
# Usage: just import-claude
#        just import-claude --dry-run
#        just import-claude --include-sidechains --limit 100
import-claude *args:
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$(pass show ai/openrouter 2>/dev/null || true)}" \
        SEMBLR_ROUNDS_DIR="{{SEMBLR_ROUNDS_DIR}}" \
        npx tsx scripts/import-claude-code.ts {{args}}

# Install the extension as a local pi package (loads on every startup)
install:
    pi install ./src/semblr.ts

# Remove the extension from pi settings
uninstall:
    pi remove ./src/semblr.ts

# Run all pending round migrations (idempotent — safe to re-run)
# Round migrations first, then index migration (model column)
migrate *args:
    SEMBLR_ROUNDS_DIR="{{SEMBLR_ROUNDS_DIR}}" \
        npx tsx scripts/migrate-rounds.ts
    SEMBLR_ROUNDS_DIR="{{SEMBLR_ROUNDS_DIR}}" \
        npx tsx scripts/migrate-content-hash.ts {{args}}
    SEMBLR_ROUNDS_DIR="{{SEMBLR_ROUNDS_DIR}}" \
        npx tsx scripts/migrate-model-column.ts {{args}}

# Erase embeddings for short prompts from index.csv and round JSON files
# (Idempotent — safe to re-run)
erase-short-embeddings *args:
    SEMBLR_ROUNDS_DIR="{{SEMBLR_ROUNDS_DIR}}" \
        npx tsx scripts/erase-short-embeddings.ts {{args}}

# Query the index with a natural language query (via pi + semblr extension)
# Usage: just query "<your question>"
query query *args:
    echo "search interactions for '{{query}}'" | pi --print --no-builtin-tools -e ./src/semblr.ts
