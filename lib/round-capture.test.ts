import { describe, expect, it } from "vitest";
import { extractAndStripFollowupMarker, readAndClearFollowupFlag } from "./round-capture.ts";

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
		// The marker detection uses \n prefix, so a response ending with "round_needs_followup"
		// without a newline should NOT match (it's embedded in text)
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

	it("clears the needsFollowup flag and returns the data", () => {
		let writtenData = "";
		let renamedFrom = "";
		let renamedTo = "";
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => JSON.stringify({ userPrompt: "hi", responseSequence: "hello", needsFollowup: true }),
			writeFileSync: (_path: string, data: string) => {
				writtenData = data;
			},
			renameSync: (from: string, to: string) => {
				renamedFrom = from;
				renamedTo = to;
			},
		};
		const result = readAndClearFollowupFlag("/file.json", fsMock);
		expect(result).not.toBeNull();
		expect(result?.userPrompt).toBe("hi");
		expect(result?.needsFollowup).toBe(true); // returns original flag value before clearing

		// Verify the written data has needsFollowup=false
		const written = JSON.parse(writtenData);
		expect(written.needsFollowup).toBe(false);
		expect(renamedFrom).toContain(".tmp.");
		expect(renamedTo).toBe("/file.json");
	});

	it("returns data even when write fails (best-effort)", () => {
		const fsMock = {
			existsSync: () => true,
			readFileSync: () => JSON.stringify({ userPrompt: "hi", responseSequence: "hello", needsFollowup: true }),
			writeFileSync: () => {
				throw new Error("write failed");
			},
			renameSync: () => {},
		};
		const result = readAndClearFollowupFlag("/file.json", fsMock);
		expect(result).not.toBeNull();
		expect(result?.userPrompt).toBe("hi");
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
