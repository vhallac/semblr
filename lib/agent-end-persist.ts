import * as nodeFs from "node:fs";
import { createRoundFilePath } from "./hash.ts";
import {
	type AgentEndRoundFile,
	buildAgentEndRoundData,
	buildAgentEndRoundFile,
	extractAndStripFollowupMarker,
} from "./round-capture.ts";
import type { CheckpointSummary, ResponseSegment, ToolCallDetail } from "./round-data.ts";

/**
 * Minimal fs surface used by the persist step — injectable for tests.
 */
export interface AgentEndPersistFs {
	mkdirSync(path: string, opts: { recursive: boolean }): void;
	existsSync(path: string): boolean;
	writeFileSync(path: string, data: string): void;
	readFileSync(path: string, encoding: "utf-8"): string;
}

/**
 * Gap-free write ordering state (F1, PR !131 round 3): once a round write
 * fails, no later round may be persisted ahead of it — otherwise the rounds
 * directory has a hole that only a startup backfill can fill while later
 * rounds keep landing. The gate is set by persistAgentEndRound on any write
 * failure and cleared only by a fresh process (startup backfill then
 * recovers the blocked rounds from the session JSONL).
 */
export interface WriteGapGate {
	blockedBy: string | null;
}

export interface AgentEndPersistDeps {
	fs: AgentEndPersistFs;
	roundsDir: string;
	/** Optional gap-free ordering gate (F1). Callers pass one per process. */
	gate?: WriteGapGate;
	/** Injectable assembly step (defaults to buildAgentEndRoundFile) — tests make this throw. */
	buildRoundFile?: typeof buildAgentEndRoundFile;
}

export interface AgentEndPersistInput {
	cachedUserPrompt: string | null;
	accumulatedText: readonly string[];
	messages: readonly unknown[] | undefined;
	turnIndex: number | null;
	toolCallCount: number;
	toolCallNames: string[];
	toolCalls: ToolCallDetail[];
	responseSegments: ResponseSegment[];
	parentId: string | null;
	summary?: CheckpointSummary;
}

/**
 * Runs after the round file has been written. Receives the saved descriptor
 * and the written round data. Exceptions here cost derived data only (index
 * rows, chain push) — never the round itself (issue #130). The caller may set
 * status messages inside; persistAgentEndRound also captures the error
 * message into the result for the caller to report.
 */
export type PostWriteStep = (saved: AgentEndRoundFile, roundData: Record<string, unknown>) => void;

/**
 * Runs only on the dedup path, after the content-hash collision is detected.
 * Used for in-memory bookkeeping that must happen even when the round file is
 * not rewritten (e.g. the causal-chain push). Exceptions are captured and
 * reported on the result — they never change the dedup outcome.
 */
export type DedupStep = (saved: AgentEndRoundFile) => void;

export type PersistAgentEndResult =
	| { kind: "saved"; saved: AgentEndRoundFile; roundData: Record<string, unknown>; postWriteError?: string }
	| { kind: "dedup"; saved: AgentEndRoundFile; dedupError?: string }
	| { kind: "emergency"; fileName: string; message: string }
	| { kind: "blocked"; blockedBy: string }
	| { kind: "failed"; message: string }
	| { kind: "no-prompt" };

/**
 * Issue #130 write-first persist step for agent_end:
 *
 * 1. Assemble the round (prompt/response extraction, content-hash file name).
 * 2. On assembly failure: emergency raw write from in-memory state.
 * 3. Write the round file.
 * 4. Only then run the caller's fallible post-write steps, each failure
 *    captured — never propagated as a round loss.
 *
 * Dedup (content-hash file already exists) is detected here; the caller keeps
 * its grouping/derived-data logic for that branch.
 */
export function persistAgentEndRound(
	deps: AgentEndPersistDeps,
	input: AgentEndPersistInput,
	postWrite?: PostWriteStep,
	onDedup?: DedupStep,
): PersistAgentEndResult {
	const { fs, roundsDir } = deps;
	const buildRoundFile = deps.buildRoundFile ?? buildAgentEndRoundFile;

	// F1: an earlier write failure must never be skipped over — later rounds
	// stay unpersisted (recoverable from the session JSONL at next startup).
	if (deps.gate?.blockedBy) {
		return { kind: "blocked", blockedBy: deps.gate.blockedBy };
	}

	let saved: AgentEndRoundFile | null = null;
	try {
		saved = buildRoundFile(input.cachedUserPrompt, input.accumulatedText, input.messages, input.toolCalls);
	} catch (err) {
		// Emergency: write a raw round from whatever in-memory state survived.
		const fbPrompt = input.cachedUserPrompt ?? "";
		// F2 (PR #131): strip the followup marker and hash the toolCalls so the
		// emergency filename matches what the normal path would have produced for
		// the same round — startup backfill reconstruction relies on this parity.
		const { cleanedText: fbText, needsFollowup: fbNeedsFollowup } = extractAndStripFollowupMarker(
			input.accumulatedText.join("\n\n").trim(),
		);
		let fbName: string | null = null;
		try {
			fs.mkdirSync(roundsDir, { recursive: true });
			fbName = fbPrompt ? createRoundFilePath(fbPrompt, fbText, input.toolCalls) : `emergency-${Date.now()}.json`;
			const fbPath = `${roundsDir}/${fbName}`;
			if (!fs.existsSync(fbPath)) {
				fs.writeFileSync(
					fbPath,
					JSON.stringify(
						buildAgentEndRoundData({
							userPrompt: fbPrompt || fbName,
							responseText: fbText,
							turnIndex: input.turnIndex,
							toolCallCount: input.toolCallCount,
							toolCallNames: input.toolCallNames,
							toolCalls: input.toolCalls,
							responseSegments: input.responseSegments,
							parentId: input.parentId,
							needsFollowup: fbNeedsFollowup,
						}),
						null,
						2,
					),
				);
			}
			return { kind: "emergency", fileName: fbName, message: (err as Error).message };
		} catch (fbErr) {
			// F1: even the emergency write failed — this round is a gap.
			if (deps.gate) deps.gate.blockedBy = fbName ?? "unknown";
			return { kind: "failed", message: (fbErr as Error).message };
		}
	}
	if (!saved) {
		return { kind: "no-prompt" };
	}

	const roundPath = `${roundsDir}/${saved.fileName}`;
	fs.mkdirSync(roundsDir, { recursive: true });

	// Skip if already saved (deduplication by content hash)
	if (fs.existsSync(roundPath)) {
		if (onDedup) {
			try {
				onDedup(saved);
			} catch (err) {
				return { kind: "dedup", saved, dedupError: (err as Error).message };
			}
		}
		return { kind: "dedup", saved };
	}

	const roundData = buildAgentEndRoundData({
		userPrompt: saved.userPrompt,
		responseText: saved.responseText,
		turnIndex: input.turnIndex,
		toolCallCount: input.toolCallCount,
		toolCallNames: input.toolCallNames,
		toolCalls: input.toolCalls,
		responseSegments: input.responseSegments,
		parentId: input.parentId,
		needsFollowup: saved.needsFollowup,
		...(input.summary ? { summary: input.summary } : {}),
	});

	try {
		fs.writeFileSync(roundPath, JSON.stringify(roundData, null, 2));
	} catch (err) {
		// F1: record the gap so later rounds are never persisted ahead of it.
		if (deps.gate) deps.gate.blockedBy = saved.fileName;
		return { kind: "failed", message: (err as Error).message };
	}

	if (postWrite) {
		try {
			postWrite(saved, roundData);
		} catch (err) {
			return { kind: "saved", saved, roundData, postWriteError: (err as Error).message };
		}
	}
	return { kind: "saved", saved, roundData };
}

export const defaultAgentEndPersistFs: AgentEndPersistFs = nodeFs;
