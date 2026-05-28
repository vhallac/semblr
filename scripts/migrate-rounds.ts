/**
 * migrate-rounds.ts — Add missing `responseSegments` to existing round files.
 *
 * Round files created before step2 do not have the `responseSegments` field,
 * which allows `get_round_details()` to show tool call positions interleaved
 * with the assistant's text response.
 *
 * This script reconstructs `responseSegments` from the existing `responseSequence`
 * (all text at once) and `toolCalls[]` (all tool calls appended at end). While
 * exact positions are lost for old rounds, the interleaved markers are still
 * more useful than the previous flat "ALL REDACTED" blob.
 *
 * The hash (file name) does NOT change — we add the field in-place.
 *
 * Usage:
 *   npx tsx scripts/migrate-rounds.ts
 *
 * Override rounds dir (default ~/.pi/agent/semblr/rounds):
 *   SEMBLR_ROUNDS_DIR=/custom/path npx tsx scripts/migrate-rounds.ts
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const ROUNDS_DIR = process.env.SEMBLR_ROUNDS_DIR ||
  path.resolve(os.homedir(), ".pi", "agent", "semblr", "rounds");

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface ResponseSegment {
  type: "text" | "toolCall";
  text?: string;
  toolCallIndex?: number;
}

interface ToolCallDetail {
  index: number;
  name: string;
  arguments: string;
  result_summary: string;
}

interface Round {
  userPrompt: string;
  responseSequence: string;
  responseSegments?: ResponseSegment[];
  toolCalls?: ToolCallDetail[];
  [key: string]: unknown;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

function main() {
  if (!fs.existsSync(ROUNDS_DIR)) {
    console.error(`❌ Rounds directory not found: ${ROUNDS_DIR}`);
    process.exit(1);
  }

  const files = fs.readdirSync(ROUNDS_DIR)
    .filter((f) => f.endsWith(".json"));

  let migrated = 0;
  let skipped = 0;

  for (const file of files) {
    const filePath = path.resolve(ROUNDS_DIR, file);

    let data: Round;
    try {
      const raw = fs.readFileSync(filePath, "utf-8");
      data = JSON.parse(raw);
    } catch {
      console.warn(`⚠ Skipping unparseable: ${file}`);
      continue;
    }

    // Already has responseSegments
    if (data.responseSegments && Array.isArray(data.responseSegments) && data.responseSegments.length > 0) {
      skipped++;
      continue;
    }

    // Reconstruct from existing fields
    const segments: ResponseSegment[] = [];

    // Add all text as one block (exact positions lost, but better than nothing)
    if (data.responseSequence && data.responseSequence.trim()) {
      segments.push({ type: "text", text: data.responseSequence });
    }

    // Add tool call markers at end
    if (data.toolCalls && Array.isArray(data.toolCalls)) {
      for (let i = 0; i < data.toolCalls.length; i++) {
        segments.push({ type: "toolCall", toolCallIndex: i });
      }
    }

    // Only write if we have something useful (skip empty rounds)
    if (segments.length === 0) {
      console.warn(`⚠ Skipping empty: ${file}`);
      continue;
    }

    data.responseSegments = segments;

    // Write back with 2-space indent
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    migrated++;
  }

  console.log(`\n✅ Done. ${migrated} round files migrated, ${skipped} already had responseSegments.`);
}

main();
