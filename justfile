# ── semblr ─────────────────────────────────────────────────────────────────
# See VISION.md for architecture and AGENTS.md for project context.

# Override via: just --set SEMBLR_ROUNDS_DIR /custom/path index
# or set SEMBLR_ROUNDS_DIR in the environment.
SEMBLR_ROUNDS_DIR := env_var_or_default("SEMBLR_ROUNDS_DIR", "")

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

# Install the extension as a local pi package (loads on every startup)
install:
    pi install ./src/semblr.ts

# Remove the extension from pi settings
uninstall:
    pi remove ./src/semblr.ts

# Query the index with a natural language query (via pi + semblr extension)
# Usage: just query "<your question>"
query query *args:
    echo "search interactions for '{{query}}'" | pi --print --no-builtin-tools -e ./src
