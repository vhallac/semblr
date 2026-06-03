export interface RoundAssistantSelectionParams {
	userPrompt: string;
	responseSequence: string;
	assistantOutput: string;
	fromLine?: number;
	lineCount?: number;
	match?: string;
	maxMatches?: number;
}

export type RoundAssistantSelection =
	| {
			ok: true;
			assistantOutput: string;
			responseTotalLines: number;
			paginationMarker: string;
			matchHeader: string;
	  }
	| { ok: false; error: string };

export interface ToolResultSelectionParams {
	resultText: string;
	fromLine?: number;
	lineCount?: number;
	match?: string;
	maxMatches?: number;
}

export type ToolResultSelection =
	| {
			ok: true;
			mode: "page";
			resultBlock: string;
			footer: string;
			fromLine: number;
			endLine: number;
			totalLines: number;
	  }
	| {
			ok: true;
			mode: "match";
			resultBlock: string;
			matchSummary: string;
			matchCount: number;
			totalMatches: number;
	  }
	| { ok: true; mode: "full" }
	| { ok: false; error: string };

function hasMatch(match?: string): match is string {
	return match !== undefined && match.length > 0;
}

export function hasLineSelectionConflict(match?: string, fromLine?: number): boolean {
	return hasMatch(match) && fromLine !== undefined;
}

export function selectRoundAssistantOutput(params: RoundAssistantSelectionParams): RoundAssistantSelection {
	const useMatch = hasMatch(params.match);
	const useFromLine = params.fromLine !== undefined;
	let responseTotalLines = 0;
	let paginationMarker = "";
	let matchHeader = "";
	let assistantOutput = params.assistantOutput;

	if (hasLineSelectionConflict(params.match, params.fromLine)) {
		return { ok: false, error: "Error: match and from_line are mutually exclusive. Use one or the other, not both." };
	}

	if (useFromLine) {
		const allLines = params.responseSequence.split("\n");
		responseTotalLines = allLines.length;
		const fromLine = params.fromLine ?? 1;
		const lineCount = params.lineCount ?? 200;
		const startIdx = Math.max(0, fromLine - 1);
		const endIdx = Math.min(responseTotalLines, startIdx + lineCount);
		const pageLines = allLines.slice(startIdx, endIdx);
		const remaining = responseTotalLines - endIdx;

		assistantOutput = pageLines.length > 0 ? pageLines.join("\n") : "(empty)";

		if (remaining > 0) {
			paginationMarker = `[Truncated — use from_line=${endIdx + 1}, line_count=${lineCount} to continue]`;
		}
	} else if (useMatch) {
		try {
			const regex = new RegExp(params.match as string, "gm");
			const allLines = params.responseSequence.split("\n");
			responseTotalLines = allLines.length;
			const lineCount = params.lineCount ?? 0;
			const maxMatches = params.maxMatches ?? 1;

			const userLines = params.userPrompt.split("\n");
			const searchLines: Array<{ text: string; source: "user" | "assistant"; originalIndex: number }> = [];
			for (let i = 0; i < userLines.length; i++) {
				searchLines.push({ text: userLines[i], source: "user", originalIndex: i });
			}
			searchLines.push({ text: "", source: "assistant", originalIndex: -1 });
			for (let i = 0; i < allLines.length; i++) {
				searchLines.push({ text: allLines[i], source: "assistant", originalIndex: i });
			}

			const matchResults: Array<{ lineIdx: number; source: "user" | "assistant"; text: string }> = [];
			for (const line of searchLines) {
				if (regex.test(line.text)) {
					matchResults.push({ lineIdx: line.originalIndex, source: line.source, text: line.text });
					regex.lastIndex = 0;
				}
			}

			const userMatchCount = matchResults.filter((m) => m.source === "user").length;
			let shownMatches = matchResults.slice(0, maxMatches);
			if (userMatchCount > 0) {
				shownMatches = [matchResults[0]];
			}

			const totalMatches = matchResults.length;
			const matchCount = shownMatches.length;
			const matchParts: string[] = [];
			for (let mi = 0; mi < matchCount; mi++) {
				const match = shownMatches[mi];
				if (match.source === "user") {
					matchParts.push(`[M ${mi + 1}/${matchCount} in user prompt] ${match.text}`);
				} else {
					const startIdx = match.lineIdx;
					const endIdx = Math.min(responseTotalLines, startIdx + 1 + lineCount);
					const context = allLines.slice(startIdx, endIdx).join("\n");
					if (lineCount > 0) {
						matchParts.push(
							`[M ${mi + 1}/${matchCount} at assistant line ${match.lineIdx + 1} (${lineCount} lines of context)]\n${context}`,
						);
					} else {
						matchParts.push(`[M ${mi + 1}/${matchCount} at assistant line ${match.lineIdx + 1}] ${match.text}`);
					}
				}
			}

			assistantOutput = matchParts.length > 0 ? matchParts.join("\n\n") : "(no matches)";
			matchHeader = formatMatchSummary(matchCount, totalMatches);
		} catch (err) {
			return {
				ok: false,
				error: `Invalid regexp pattern: ${(err as Error).message}. Provide a valid JavaScript regexp string.`,
			};
		}
	}

	return { ok: true, assistantOutput, responseTotalLines, paginationMarker, matchHeader };
}

export function selectToolResultOutput(params: ToolResultSelectionParams): ToolResultSelection {
	const useMatch = hasMatch(params.match);
	const useFromLine = params.fromLine !== undefined;

	if (hasLineSelectionConflict(params.match, params.fromLine)) {
		return {
			ok: false,
			error: "Error: match and out__from_line are mutually exclusive. Use one or the other, not both.",
		};
	}

	if (useFromLine) {
		const allLines = params.resultText.split("\n");
		const totalLines = allLines.length;
		const fromLine = params.fromLine ?? 1;
		const lineCount = params.lineCount ?? 200;
		const startIdx = Math.max(0, fromLine - 1);
		const endIdx = Math.min(totalLines, startIdx + lineCount);
		const pageLines = allLines.slice(startIdx, endIdx);
		const remaining = totalLines - endIdx;
		const resultBlock = pageLines.length > 0 ? pageLines.join("\n") : "(empty)";
		const footer =
			remaining > 0
				? `\n\n[Truncated — lines remaining: ${remaining}. Use out__from_line=${endIdx + 1} and out_line_count=${lineCount} to continue.]`
				: "";

		return { ok: true, mode: "page", resultBlock, footer, fromLine, endLine: endIdx, totalLines };
	}

	if (useMatch) {
		try {
			const regex = new RegExp(params.match as string, "gm");
			const allLines = params.resultText.split("\n");
			const totalLines = allLines.length;
			const lineCount = params.lineCount ?? 0;
			const maxMatches = params.maxMatches ?? 1;
			const matchResults: number[] = [];
			for (let i = 0; i < allLines.length; i++) {
				if (regex.test(allLines[i])) {
					matchResults.push(i);
					regex.lastIndex = 0;
				}
			}

			const totalMatches = matchResults.length;
			const shownIndices = matchResults.slice(0, maxMatches);
			const matchCount = shownIndices.length;
			const matchParts: string[] = [];
			for (let mi = 0; mi < matchCount; mi++) {
				const lineIdx = shownIndices[mi];
				const startIdx = lineIdx;
				const endIdx = Math.min(totalLines, startIdx + 1 + lineCount);
				const context = allLines.slice(startIdx, endIdx).join("\n");
				if (lineCount > 0) {
					matchParts.push(
						`[M ${mi + 1}/${matchCount} at line ${lineIdx + 1} (${lineCount} lines of context)]\n${context}`,
					);
				} else {
					matchParts.push(`[M ${mi + 1}/${matchCount} at line ${lineIdx + 1}] ${allLines[lineIdx]}`);
				}
			}

			const resultBlock = matchParts.length > 0 ? matchParts.join("\n\n") : "(no matches)";
			return {
				ok: true,
				mode: "match",
				resultBlock,
				matchSummary: formatMatchSummary(matchCount, totalMatches),
				matchCount,
				totalMatches,
			};
		} catch (err) {
			return {
				ok: false,
				error: `Invalid regexp pattern: ${(err as Error).message}. Provide a valid JavaScript regexp string.`,
			};
		}
	}

	return { ok: true, mode: "full" };
}

function formatMatchSummary(shown: number, total: number): string {
	const remaining = total - shown;
	if (remaining > 0) {
		return ` (${shown} match${shown !== 1 ? "es" : ""} shown of ${total} total)`;
	}
	if (total > 0) {
		const label = total === 1 ? "match" : "matches";
		return ` (${total} ${label})`;
	}
	return "";
}
