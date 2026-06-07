import { describe, expect, it } from "vitest";
import {
	applyMessageEndToState,
	buildAgentEndChainEntry,
	buildAgentEndEmbeddingTexts,
	buildAgentEndRoundData,
	buildAgentEndToolSummary,
	embeddingMaxTokensToResponseBytes,
	extractAgentEndResponseText,
	extractAgentEndUserPrompt,
	extractAndStripFollowupMarker,
	getAgentEndParentId,
	getRelatedParentIdFromGroup,
	type MessageEndProcessingState,
	readAndClearFollowupFlag,
} from "./round-capture.ts";

describe("extractAndStripFollowupMarker", () => {
	it("returns cleaned text and needsFollowup=true when marker is present", () => {
		const result = extractAndStripFollowupMarker("The capital is Paris.\n\nround_needs_followup");
		expect(result.needsFollowup).toBe(true);
		expect(result.cleanedText).toBe("The capital is Paris.");
	});

	it("returns original text and needsFollowup=false when marker is absent", () => {
		const result = extractAndStripFollowupMarker("The capital is Paris.");
		expect(result.needsFollowup).toBe(false);
		expect(result.cleanedText).toBe("The capital is Paris.");
	});

	it("trims trailing whitespace after stripping the marker", () => {
		const result = extractAndStripFollowupMarker("The capital is Paris.  \n\nround_needs_followup");
		expect(result.needsFollowup).toBe(true);
		expect(result.cleanedText).toBe("The capital is Paris.");
	});

	it("does not match marker mid-text", () => {
		const result = extractAndStripFollowupMarker("round_needs_followup is a marker\nThe capital is Paris.");
		expect(result.needsFollowup).toBe(false);
		expect(result.cleanedText).toBe("round_needs_followup is a marker\nThe capital is Paris.");
	});

	it("handles empty string", () => {
		const result = extractAndStripFollowupMarker("");
		expect(result.needsFollowup).toBe(false);
		expect(result.cleanedText).toBe("");
	});

	it("handles marker without leading newline (when response ends with it)", () => {
		const result = extractAndStripFollowupMarker("some textround_needs_followup");
		expect(result.needsFollowup).toBe(false);
		expect(result.cleanedText).toBe("some textround_needs_followup");
	});
});

describe("readAndClearFollowupFlag", () => {
	it("returns null when file does not exist", () => {
		const fsMock = {
			existsSync: () => false,
			readFileSync: () => "",
			writeFileSync: () => {},
			renameSync: () => {},
		};
		expect(readAndClearFollowupFlag("/nonexistent.json", fsMock)).toBeNull();
	});

	it("returns null when file has no needsFollowup flag", () => {
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => JSON.stringify({ userPrompt: "hi", responseSequence: "hello" }),
			writeFileSync: () => {},
			renameSync: () => {},
		};
		expect(readAndClearFollowupFlag("/file.json", fsMock)).toBeNull();
	});

	it("returns the round data without clearing the flag", () => {
		let writeCalled = false;
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => JSON.stringify({ userPrompt: "hi", responseSequence: "hello", needsFollowup: true }),
			writeFileSync: () => {
				writeCalled = true;
			},
			renameSync: () => {},
		};
		const result = readAndClearFollowupFlag("/file.json", fsMock);
		expect(result).not.toBeNull();
		expect(result?.userPrompt).toBe("hi");
		expect(result?.responseSequence).toBe("hello");
		expect(result?.needsFollowup).toBe(true);
		expect(writeCalled).toBe(false);
	});

	it("returns null when file has invalid JSON", () => {
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => "not valid json",
			writeFileSync: () => {},
			renameSync: () => {},
		};
		expect(readAndClearFollowupFlag("/file.json", fsMock)).toBeNull();
	});
});

describe("extractAgentEndUserPrompt", () => {
	it("returns cached prompt when provided", () => {
		expect(extractAgentEndUserPrompt("cached prompt")).toBe("cached prompt");
	});

	it("extracts from messages when no cached prompt", () => {
		const messages = [
			{ role: "assistant", content: "hi" },
			{ role: "user", content: "hello there" },
		];
		expect(extractAgentEndUserPrompt(null, messages)).toBe("hello there");
	});

	it("extracts from array content in messages", () => {
		const messages = [
			{
				role: "user",
				content: [
					{ type: "text", text: "hello" },
					{ type: "text", text: "world" },
				],
			},
		];
		expect(extractAgentEndUserPrompt(null, messages)).toBe("hello world");
	});

	it("returns empty string when no prompt or messages", () => {
		expect(extractAgentEndUserPrompt(null)).toBe("");
	});

	it("returns empty string when no user messages found", () => {
		const messages = [{ role: "assistant", content: "hi" }];
		expect(extractAgentEndUserPrompt(null, messages)).toBe("");
	});
});

describe("extractAgentEndResponseText", () => {
	it("joins accumulated text", () => {
		expect(extractAgentEndResponseText(["Hello", "World"])).toBe("Hello\n\nWorld");
	});

	it("extracts from messages when no accumulated text", () => {
		const messages = [
			{ role: "user", content: "prompt" },
			{ role: "assistant", content: "response text" },
		];
		expect(extractAgentEndResponseText([], messages)).toBe("response text");
	});

	it("extracts from array content in messages", () => {
		const messages = [
			{
				role: "assistant",
				content: [
					{ type: "text", text: "hello" },
					{ type: "text", text: "world" },
				],
			},
		];
		expect(extractAgentEndResponseText([], messages)).toBe("hello world");
	});

	it("returns empty string when nothing available", () => {
		expect(extractAgentEndResponseText([])).toBe("");
	});
});

describe("buildAgentEndToolSummary", () => {
	it("formats with tool count", () => {
		expect(buildAgentEndToolSummary(3, ["bash", "edit", "read"])).toBe("3 tools (bash, edit, read)");
	});

	it("formats discussion-only", () => {
		expect(buildAgentEndToolSummary(0, [])).toBe("0 tools (discussion)");
	});
});

describe("buildAgentEndChainEntry", () => {
	it("builds a chain entry", () => {
		const entry = buildAgentEndChainEntry("round.json", "prompt", "response", 1, ["bash"]);
		expect(entry.fileName).toBe("round.json");
		expect(entry.userPrompt).toBe("prompt");
		expect(entry.responseSequence).toBe("response");
		expect(entry.toolSummary).toBe("1 tools (bash)");
	});
});

describe("getAgentEndParentId", () => {
	it("returns null for chain shorter than 2", () => {
		expect(getAgentEndParentId([])).toBeNull();
		expect(getAgentEndParentId([{ fileName: "a.json" }])).toBeNull();
	});

	it("returns second-to-last filename for longer chains", () => {
		const chain = [{ fileName: "a.json" }, { fileName: "b.json" }, { fileName: "c.json" }];
		expect(getAgentEndParentId(chain)).toBe("b.json");
	});
});

describe("embeddingMaxTokensToResponseBytes", () => {
	it("preserves the existing 8K-token clipping default", () => {
		expect(embeddingMaxTokensToResponseBytes(8000)).toBe(24000);
	});

	it("converts configured embedding max tokens into the response byte clip budget", () => {
		expect(embeddingMaxTokensToResponseBytes(1234)).toBe(3702);
	});
});

describe("buildAgentEndEmbeddingTexts", () => {
	it("combines prompt and clipped response", () => {
		const result = buildAgentEndEmbeddingTexts("What is 2+2?", "4");
		expect(result.clippedResponse).toBe("4");
		expect(result.combinedText).toBe("What is 2+2?\n\n4");
	});

	it("strips REDACTED markers", () => {
		const response = "Before[Tool call REDACTED: expand]\nAfter[REDACTED: something]\nDone";
		const result = buildAgentEndEmbeddingTexts("prompt", response);
		expect(result.clippedResponse).not.toContain("REDACTED");
		expect(result.clippedResponse).toContain("Before");
		expect(result.clippedResponse).toContain("After");
		expect(result.clippedResponse).toContain("Done");
	});

	it("clips response to maxResponseBytes", () => {
		const longResponse = "x".repeat(50000);
		const result = buildAgentEndEmbeddingTexts("prompt", longResponse, 100);
		expect(Buffer.byteLength(result.clippedResponse, "utf-8")).toBeLessThanOrEqual(120);
	});
});

describe("buildAgentEndRoundData", () => {
	it("builds round data with all fields", () => {
		const result = buildAgentEndRoundData({
			userPrompt: "test prompt",
			responseText: "test response",
			turnIndex: 5,
			toolCallCount: 2,
			toolCallNames: ["bash", "edit"],
			toolCalls: [
				{ index: 0, name: "bash", arguments: "{}", result_summary: "ok" },
				{ index: 1, name: "edit", arguments: "{}", result_summary: "ok" },
			],
			responseSegments: [{ type: "text", text: "test response" }],
			parentId: "parent.json",
		});

		expect(result.userPrompt).toBe("test prompt");
		expect(result.responseSequence).toBe("test response");
		expect(result.turnIndex).toBe(5);
		expect(result.toolCallCount).toBe(2);
		expect(result.toolCallNames).toEqual(["bash", "edit"]);
		expect(result.parentId).toBe("parent.json");
		expect(result.needsFollowup).toBe(false);
		expect(result.relatedParentId).toBeNull();
		expect(result.promptEmbedding).toBeUndefined();
	});

	it("uses defaults for optional fields", () => {
		const result = buildAgentEndRoundData({
			userPrompt: "test",
			responseText: "resp",
			turnIndex: null,
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
			parentId: null,
		});

		expect(result.turnIndex).toBe(0);
		expect(result.needsFollowup).toBe(false);
	});

	it("passes through needsFollowup flag", () => {
		const result = buildAgentEndRoundData({
			userPrompt: "test",
			responseText: "resp",
			turnIndex: 0,
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
			parentId: null,
			needsFollowup: true,
		});

		expect(result.needsFollowup).toBe(true);
	});
});

describe("getRelatedParentIdFromGroup", () => {
	it("returns null for single round groups", () => {
		const group = { rounds: [{ fileName: "a.json" }] };
		expect(getRelatedParentIdFromGroup(group, { fileName: "a.json" })).toBeNull();
	});

	it("returns null when round is first in group", () => {
		const group = { rounds: [{ fileName: "a.json" }, { fileName: "b.json" }] };
		expect(getRelatedParentIdFromGroup(group, { fileName: "a.json" })).toBeNull();
	});

	it("returns previous round fileName when round is not first", () => {
		const roundA = { fileName: "a.json" };
		const roundB = { fileName: "b.json" };
		const group = { rounds: [roundA, roundB] };
		expect(getRelatedParentIdFromGroup(group, roundB)).toBe("a.json");
	});

	it("returns null for empty groups", () => {
		expect(getRelatedParentIdFromGroup({ rounds: [] }, { fileName: "a.json" })).toBeNull();
	});
});

describe("applyMessageEndToState", () => {
	it("handles null/undefined message", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState(null, state);
		expect(state.accumulatedText).toEqual([]);
	});

	it("handles user messages (no side effects)", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState({ role: "user", content: "hello" }, state);
		expect(state.accumulatedText).toEqual([]);
	});

	it("extracts text from assistant messages with array content", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState(
			{
				role: "assistant",
				content: [{ type: "text", text: "Hello world" }],
			},
			state,
		);
		expect(state.accumulatedText).toEqual(["Hello world"]);
		expect(state.responseSegments).toEqual([{ type: "text", text: "Hello world" }]);
	});

	it("handles assistant toolCall blocks", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState(
			{
				role: "assistant",
				content: [
					{ type: "text", text: "Let me check." },
					{ type: "toolCall", name: "bash", id: "tc1", arguments: { command: "ls" } },
				],
			},
			state,
		);
		expect(state.toolCallCount).toBe(1);
		expect(state.toolCallNames).toEqual(["bash"]);
		expect(state.toolCalls).toHaveLength(1);
		expect(state.toolCalls[0].name).toBe("bash");
		expect(state.toolCalls[0].arguments).toBe('{"command":"ls"}');
		expect(state.responseSegments).toHaveLength(2);
		expect(state.responseSegments[1]).toEqual({ type: "toolCall", toolCallIndex: 0 });
	});

	it("skips toolCall blocks without name or id", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState(
			{
				role: "assistant",
				content: [{ type: "toolCall" }],
			},
			state,
		);
		expect(state.toolCallCount).toBe(1);
		expect(state.toolCalls).toHaveLength(0);
	});

	it("handles toolResult messages by pairing with tool calls", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: ["thinking..."],
			toolCallCount: 1,
			toolCallNames: ["bash"],
			toolCalls: [{ index: 0, name: "bash", arguments: "{}", result_summary: "" }],
			responseSegments: [{ type: "text", text: "thinking..." }],
		};
		applyMessageEndToState(
			{
				role: "toolResult",
				toolCallId: "tc1",
				content: [{ type: "text", text: "file1.txt\nfile2.txt" }],
			},
			state,
		);
		expect(state.toolCalls[0].result_summary).toBe("file1.txt\nfile2.txt".slice(0, 300));
		expect(state.toolCalls[0].result_full).toBe("file1.txt\nfile2.txt");
	});

	it("handles toolResult without toolCallId", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState(
			{
				role: "toolResult",
				content: [{ type: "text", text: "result" }],
			},
			state,
		);
		// No toolCallId, so nothing should happen
		expect(state.toolCalls).toHaveLength(0);
	});

	it("skips text blocks without text", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: [],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState(
			{
				role: "assistant",
				content: [{ type: "text" }],
			},
			state,
		);
		expect(state.accumulatedText).toEqual([]);
		expect(state.responseSegments).toEqual([]);
	});

	it("does not add duplicate tool call names", () => {
		const state: MessageEndProcessingState = {
			accumulatedText: [],
			toolCallCount: 0,
			toolCallNames: ["bash"],
			toolCalls: [],
			responseSegments: [],
		};
		applyMessageEndToState(
			{
				role: "assistant",
				content: [
					{ type: "toolCall", name: "bash", id: "tc2", arguments: {} },
					{ type: "toolCall", name: "edit", id: "tc3", arguments: {} },
				],
			},
			state,
		);
		expect(state.toolCallNames).toEqual(["bash", "edit"]);
		expect(state.toolCallCount).toBe(2);
	});
});
