import { describe, expect, it } from "vitest";
import { getModelForPhase, MVP_PHASE_MODEL_MAP } from "./phase-model-map.ts";
import type { PhaseModelMap, PhaseName } from "./semblr-config.ts";

const ALL_PHASES: PhaseName[] = ["exploring", "planning", "executing", "stuck", "verifying", "reporting"];

describe("MVP_PHASE_MODEL_MAP", () => {
	it("has exactly 6 phase entries", () => {
		expect(Object.keys(MVP_PHASE_MODEL_MAP)).toHaveLength(6);
	});

	it("includes all six phase names as keys", () => {
		for (const phase of ALL_PHASES) {
			expect(MVP_PHASE_MODEL_MAP).toHaveProperty(phase);
		}
	});

	it("exploring maps to null (stay on current model)", () => {
		expect(MVP_PHASE_MODEL_MAP.exploring).toBeNull();
	});

	it("planning maps to a :cloud model ID", () => {
		expect(MVP_PHASE_MODEL_MAP.planning).toBe("deepseek-v4-flash:cloud");
	});

	it("executing maps to glm-5.2:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.executing).toBe("glm-5.2:cloud");
	});

	it("stuck maps to kimi-k2.6:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.stuck).toBe("kimi-k2.6:cloud");
	});

	it("verifying maps to minimax-m3:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.verifying).toBe("minimax-m3:cloud");
	});

	it("reporting maps to gemma4:12b:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.reporting).toBe("gemma4:12b:cloud");
	});

	it("all values are either null or a non-empty string", () => {
		for (const [, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value === null) continue;
			expect(typeof value).toBe("string");
			expect(value.length).toBeGreaterThan(0);
		}
	});

	it("all cloud model values end with :cloud", () => {
		for (const [, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value === null) continue;
			expect(value).toMatch(/:cloud$/);
		}
	});

	it("all non-null values are distinct (no duplicate models)", () => {
		const values = Object.values(MVP_PHASE_MODEL_MAP).filter((v) => v !== null);
		expect(new Set(values).size).toBe(values.length);
	});

	it("no phase maps to an empty string", () => {
		for (const [, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value !== null) {
				expect(value.length).toBeGreaterThan(0);
			}
		}
	});
});

describe("getModelForPhase", () => {
	it("maps exploring to null", () => {
		expect(getModelForPhase("exploring", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("maps planning to deepseek-v4-flash:cloud", () => {
		expect(getModelForPhase("planning", MVP_PHASE_MODEL_MAP)).toBe("deepseek-v4-flash:cloud");
	});

	it("maps all non-null phases to the correct model IDs", () => {
		const expected: Record<string, string | null> = {
			exploring: null,
			planning: "deepseek-v4-flash:cloud",
			executing: "glm-5.2:cloud",
			stuck: "kimi-k2.6:cloud",
			verifying: "minimax-m3:cloud",
			reporting: "gemma4:12b:cloud",
		};
		for (const phase of ALL_PHASES) {
			expect(getModelForPhase(phase, MVP_PHASE_MODEL_MAP)).toBe(expected[phase]);
		}
	});

	it("is deterministic (same input -> same output)", () => {
		for (const phase of ALL_PHASES) {
			const first = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			const second = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			expect(first).toBe(second);
		}
	});

	it("returns null for arbitrary strings not in the phase union", () => {
		expect(getModelForPhase("thinking" as PhaseName, MVP_PHASE_MODEL_MAP)).toBeNull();
		expect(getModelForPhase("reviewing" as PhaseName, MVP_PHASE_MODEL_MAP)).toBeNull();
		expect(getModelForPhase("" as PhaseName, MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("uses a custom map when provided", () => {
		const customMap: PhaseModelMap = {
			exploring: "custom-explorer:cloud",
			planning: null,
			executing: null,
			stuck: "custom-model:cloud",
			verifying: null,
			reporting: "custom-reporter:cloud",
		};
		expect(getModelForPhase("exploring", customMap)).toBe("custom-explorer:cloud");
		expect(getModelForPhase("planning", customMap)).toBeNull();
		expect(getModelForPhase("stuck", customMap)).toBe("custom-model:cloud");
		expect(getModelForPhase("reporting", customMap)).toBe("custom-reporter:cloud");
	});
});
