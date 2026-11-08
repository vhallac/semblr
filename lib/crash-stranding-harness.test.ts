import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { forcingReproducibleSuffixes } from "./embed-round.ts";
import { loadShardVectorIndex } from "./index-io.ts";
import { appendIndexRowToShards } from "./index-storage.ts";
import {
	backfillMissingRounds,
	buildCurrentModelRowPredicate,
	embedRecoveredRounds,
	planStartupPendingScan,
	startupEmbedStatusMessage,
} from "./session-backfill.ts";

// Issue #136 acceptance condition: the two-run crash-stranding harness.
//
// Run 1 recovers rounds from a previous session's JSONL and "crashes" after
// the round writes but before embedding completes (no vector-index rows, no
// promptEmbedding marker). Run 2 starts with that session already complete —
// the backfill early exit (`skippedComplete`) — and must still surface the
// stranded rounds: the pending scan reports them (pinned defer message when
// above the inline cap, inline pass under it) and after the pass every
// recovered round gains current-model vector-index rows.
//
// The harness drives the same lib functions the extension's session_start
// handler calls, in the same order, with the real fs — the externally
// anchored behavior is the state on disk and the message wording, not the
// extension internals.

const CURRENT_MODEL = "text-embedding-3-small";

function writeSessionFile(dir: string, lines: object[], name = "session.jsonl"): string {
	const file = path.join(dir, name);
	fs.writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	return file;
}

function userMsg(text: string, id: string, timestamp = 1000) {
	return { type: "message", id, message: { role: "user", content: [{ type: "text", text }], timestamp } };
}

function assistantMsg(text: string) {
	return { type: "message", message: { role: "assistant", content: [{ type: "text", text }] } };
}

// Mirrors the startup call site: derive the forcing suffix set per round from
// the round file (a short prompt's `:prompt` row is non-reproducible).
function forcingSuffixesFor(roundsDir: string) {
	const cache = new Map<string, ReadonlySet<string>>();
	return (fileName: string): ReadonlySet<string> => {
		const cached = cache.get(fileName);
		if (cached) return cached;
		const round = JSON.parse(fs.readFileSync(path.join(roundsDir, fileName), "utf-8")) as {
			userPrompt?: string;
			summary?: unknown;
		};
		const suffixes = forcingReproducibleSuffixes(String(round.userPrompt ?? ""), Boolean(round.summary));
		cache.set(fileName, suffixes);
		return suffixes;
	};
}

function hasCurrentModelRows(roundsDir: string, fileName: string, forcing: (f: string) => ReadonlySet<string>) {
	const predicate = buildCurrentModelRowPredicate(loadShardVectorIndex(roundsDir), CURRENT_MODEL, forcing);
	return predicate(fileName);
}

describe("two-run crash-stranding harness (issue #136 acceptance)", () => {
	let tmp: string;
	beforeEach(() => {
		tmp = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-crash-stranding-"));
	});
	afterEach(() => {
		fs.rmSync(tmp, { recursive: true, force: true });
	});

	it("run 2 surfaces rounds stranded by a run-1 crash after round writes before embedding (inline under cap)", async () => {
		// ── Run 1: recover from the previous session, then crash ──
		const sessionFile = writeSessionFile(tmp, [
			userMsg("please explain the border provinces dispute in full", "u1"),
			assistantMsg("a long answer"),
		]);
		const roundsDir = path.join(tmp, "rounds");
		const run1 = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		expect(run1.recoveredFiles).toHaveLength(1);
		// Round files exist on disk, but the crash cut the run before embedding:
		// no vector-index rows, no promptEmbedding marker.
		const stranded = run1.recoveredFiles[0];
		expect(fs.existsSync(path.join(roundsDir, stranded))).toBe(true);
		expect(loadShardVectorIndex(roundsDir)).toEqual([]);
		expect(JSON.parse(fs.readFileSync(path.join(roundsDir, stranded), "utf-8")).promptEmbedding).toBeUndefined();

		// ── Run 2: session already complete → early exit, pending scan ──
		const run2 = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		expect(run2.skippedComplete).toBe(true);
		expect(run2.recoveredFiles).toEqual([]);
		// The scan (readdirSync + index diff over an empty shard set) counts the
		// stranded round pending and — under the inline cap — queues it.
		const forcing = forcingSuffixesFor(roundsDir);
		const roundFileNames = fs.readdirSync(roundsDir).filter((fn) => fn.endsWith(".json") && !fn.startsWith("index"));
		const embedPlan = planStartupPendingScan(
			roundFileNames,
			() => false,
			(name) => JSON.parse(fs.readFileSync(path.join(roundsDir, name), "utf-8")),
		);
		expect(embedPlan).toEqual({ mode: "inline", pendingCount: 1, pendingFiles: [stranded] });

		// The inline pass embeds the pending files: rows land in the host's own
		// shard and the marker is written — the live-parity write surface.
		const result = await embedRecoveredRounds(embedPlan.pendingFiles, roundsDir, {
			embed: (text) => Promise.resolve([text.length, 1]),
			appendIndexRow: (label, vec) => appendIndexRowToShards(roundsDir, label, vec, {}, CURRENT_MODEL),
			writeRoundEmbedding: (name, vec) => {
				const p = path.join(roundsDir, name);
				const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
				existing.promptEmbedding = vec;
				fs.writeFileSync(p, JSON.stringify(existing, null, 2));
			},
		});
		expect(result).toEqual({ embedded: [stranded], errors: [] });
		expect(result).toEqual({ embedded: [stranded], errors: [] });

		// After the pass every recovered round has current-model rows (rows are
		// truth — issue #140 D1), and a third run would count zero pending.
		expect(hasCurrentModelRows(roundsDir, stranded, forcing)).toBe(true);
		const after = planStartupPendingScan(
			roundFileNames,
			(fileName) => hasCurrentModelRows(roundsDir, fileName, forcing),
			(name) => JSON.parse(fs.readFileSync(path.join(roundsDir, name), "utf-8")),
		);
		expect(after).toEqual({ mode: "inline", pendingCount: 0, pendingFiles: [] });
	});

	it("defers when stranded rounds exceed the inline cap, reporting the pinned message; rows gained after the sweep-style pass", async () => {
		// Run 1: 11 stranded rounds (above STARTUP_EMBED_INLINE_MAX = 10).
		const lines: object[] = [];
		for (let i = 0; i < 11; i++) {
			lines.push(
				userMsg(`please discuss border province dispute number ${i} at length`, `u${i}`),
				assistantMsg(`answer number ${i}`),
			);
		}
		const sessionFile = writeSessionFile(tmp, lines);
		const roundsDir = path.join(tmp, "rounds");
		const run1 = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		expect(run1.recoveredFiles).toHaveLength(11);

		// Run 2: skippedComplete early exit; the scan defers the burst and the
		// pinned defer message (decision 011) reports the count.
		const run2 = backfillMissingRounds(sessionFile, roundsDir, undefined, { liveWindowMs: 0 });
		expect(run2.skippedComplete).toBe(true);
		const forcing = forcingSuffixesFor(roundsDir);
		const roundFileNames = fs.readdirSync(roundsDir).filter((fn) => fn.endsWith(".json") && !fn.startsWith("index"));
		const embedPlan = planStartupPendingScan(
			roundFileNames,
			() => false,
			(name) => JSON.parse(fs.readFileSync(path.join(roundsDir, name), "utf-8")),
		);
		expect(embedPlan.mode).toBe("defer");
		expect(embedPlan.pendingCount).toBe(11);
		expect(startupEmbedStatusMessage(embedPlan)).toBe("🧠 11 rounds pending embedding backfill — run just index");

		// The sweep (`just index`) that consumes the deferral embeds everything:
		// after it every stranded round has current-model rows.
		for (const fileName of embedPlan.pendingFiles) {
			await embedRecoveredRounds([fileName], roundsDir, {
				embed: (text) => Promise.resolve([text.length, 1]),
				appendIndexRow: (label, vec) => appendIndexRowToShards(roundsDir, label, vec, {}, CURRENT_MODEL),
				writeRoundEmbedding: (name, vec) => {
					const p = path.join(roundsDir, name);
					const existing = JSON.parse(fs.readFileSync(p, "utf-8"));
					existing.promptEmbedding = vec;
					fs.writeFileSync(p, JSON.stringify(existing, null, 2));
				},
			});
		}
		for (const fileName of run1.recoveredFiles) {
			expect(hasCurrentModelRows(roundsDir, fileName, forcing)).toBe(true);
		}
	});
});
