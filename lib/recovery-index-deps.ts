import type * as fs from "node:fs";
import type { Bm25Index } from "./bm25-index.ts";
import { roundTextForBm25, upsertBm25Round, writeBm25Index } from "./bm25-index.ts";
import type { RoundData } from "./round-data.ts";
import { readRoundJson } from "./round-io.ts";
import { appendToolIndexRows, buildToolIndexRows } from "./search-tools.ts";
import type { IndexRecoveredRoundsDeps } from "./session-backfill.ts";

export interface RecoveryIndexDepsOptions {
	/** Rounds directory the recovered files live in (also the tool-index base). */
	roundsDir: string;
	/** Path of the bm25 sidecar written exactly once per batch (the flush). */
	bm25IndexPath: string;
	/** Path of the tool-call fulltext index. */
	toolIndexPath: string;
	/**
	 * Memoized bm25 loader shared with the query path (issue #137): upserts
	 * mutate the cached index in memory and the single flush persists it.
	 */
	loadBm25Index: () => Bm25Index;
	/** Storage seam for tests (defaults to node:fs). */
	fsImpl?: Pick<typeof fs, "existsSync" | "mkdirSync" | "readFileSync" | "readdirSync" | "writeFileSync">;
}

/**
 * Production wiring for `indexRecoveredRounds` (issue #137, PR !138 F3):
 * memory-only bm25 upserts against the shared cached index, one file write
 * per batch (flushBm25), and tool-index rows appended per round. Extracted
 * so the exactly-once contract of the real deps object is testable.
 */
export function createRecoveryIndexDeps(options: RecoveryIndexDepsOptions): IndexRecoveredRoundsDeps {
	return {
		readRoundData: (fileName) => readRoundJson(options.roundsDir, fileName),
		upsertBm25: (fileName, roundData) => {
			const index = options.loadBm25Index();
			upsertBm25Round(index, fileName, roundTextForBm25(roundData as unknown as RoundData));
		},
		flushBm25: () => writeBm25Index(options.bm25IndexPath, options.loadBm25Index(), options.fsImpl),
		appendToolRows: (fileName, toolCalls) =>
			appendToolIndexRows(options.toolIndexPath, options.roundsDir, buildToolIndexRows(fileName, toolCalls)),
	};
}
