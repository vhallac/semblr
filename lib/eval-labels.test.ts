import { describe, expect, it } from "vitest";
import { collectWeakLabelExpansionsFromSessionJsonl } from "./eval-labels.ts";
import { createRoundFilePath } from "./hash.ts";

const line = (value: unknown) => JSON.stringify(value);

describe("eval weak-label mining", () => {
	it("collects expanded round files from get_round_details calls within each reconstructed round", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "first prompt words here" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "toolCall", name: "get_round_details", arguments: { round: "older-a.json" } },
						{ type: "toolCall", name: "get_round_details", arguments: { round: "older-b.json" } },
						{ type: "toolCall", name: "search_interactions", arguments: { query: "ignored" } },
						{ type: "text", text: "response one" },
					],
				},
			}),
			line({
				type: "message",
				id: "u2",
				message: { role: "user", timestamp: 20, content: [{ type: "text", text: "second prompt words here" }] },
			}),
			line({
				type: "message",
				message: { role: "assistant", content: [{ type: "text", text: "response two" }] },
			}),
		].join("\n");

		const labels = collectWeakLabelExpansionsFromSessionJsonl(raw);
		expect(labels).toEqual(
			new Map([
				[
					createRoundFilePath(
						"first prompt words here",
						"response one",
						['{"round":"older-a.json"}', '{"round":"older-b.json"}', '{"query":"ignored"}'].map(
							(argumentsText) => ({ arguments: argumentsText, result_summary: "" }),
						),
					),
					["older-a.json", "older-b.json"],
				],
				[createRoundFilePath("second prompt words here", "response two", []), []],
			]),
		);
	});

	it("skips malformed and non-JSON lines while mining labels", () => {
		const raw = [
			"not json",
			line({ type: "event", message: { role: "assistant", content: [{ type: "text", text: "ignored" }] } }),
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "prompt words here" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "toolCall", name: "get_round_details", arguments: { round: "older.json" } },
						{ type: "text", text: "response body" },
					],
				},
			}),
		].join("\n");

		expect(collectWeakLabelExpansionsFromSessionJsonl(raw)).toEqual(
			new Map([
				[
					createRoundFilePath("prompt words here", "response body", [
						{ arguments: '{"round":"older.json"}', result_summary: "" },
					]),
					["older.json"],
				],
			]),
		);
	});

	it("ignores get_round_details calls whose arguments are unparsable", () => {
		const raw = [
			line({
				type: "message",
				id: "u1",
				message: { role: "user", timestamp: 10, content: [{ type: "text", text: "prompt words here" }] },
			}),
			line({
				type: "message",
				message: {
					role: "assistant",
					content: [
						{ type: "toolCall", name: "get_round_details", arguments: { round: "older.json" } },
						{ type: "toolCall", name: "get_round_details", arguments: "{" },
						{ type: "text", text: "response body" },
					],
				},
			}),
		].join("\n");

		expect(collectWeakLabelExpansionsFromSessionJsonl(raw)).toEqual(
			new Map([
				[
					createRoundFilePath("prompt words here", "response body", [
						{ arguments: '{"round":"older.json"}', result_summary: "" },
						{ arguments: '"{"', result_summary: "" },
					]),
					["older.json"],
				],
			]),
		);
	});
});
