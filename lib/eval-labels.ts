import { type ParsedPiRound, parsePiSessionJsonl, reconstructPiSessionRounds } from "./pi-session.ts";

function extractExpandedRoundFile(argumentsText: string): string | null {
	try {
		const parsed = JSON.parse(argumentsText) as unknown;
		if (Array.isArray(parsed) && typeof parsed[0] === "string") return parsed[0];
		if (
			parsed &&
			typeof parsed === "object" &&
			"round" in parsed &&
			typeof (parsed as { round?: unknown }).round === "string"
		) {
			return (parsed as { round: string }).round;
		}
	} catch {}
	return null;
}

export function collectWeakLabelExpansionsFromRounds(rounds: readonly ParsedPiRound[]): Map<string, string[]> {
	return new Map(
		reconstructPiSessionRounds(rounds).map(({ roundFile, round }) => [
			roundFile,
			round.toolCalls
				.filter((toolCall) => toolCall.name === "get_round_details")
				.flatMap((toolCall) => {
					const expanded = extractExpandedRoundFile(toolCall.arguments);
					return expanded ? [expanded] : [];
				}),
		]),
	);
}

export function collectWeakLabelExpansionsFromSessionJsonl(raw: string): Map<string, string[]> {
	return collectWeakLabelExpansionsFromRounds(parsePiSessionJsonl(raw));
}
