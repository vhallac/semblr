import { describe, expect, it } from "vitest";
import { getModelForPhase, MVP_PHASE_MODEL_MAP } from "./phase-model-map.ts";
import type { PhaseModelMap, PhaseName } from "./semblr-config.ts";

const ALL_PHASES: PhaseName[] = ["exploring", "planning", "executing", "verifying", "reporting"];

describe("MVP_PHASE_MODEL_MAP", () => {
	it("has exactly 5 phase entries", () => {
		expect(Object.keys(MVP_PHASE_MODEL_MAP)).toHaveLength(5);
	});

	it("includes all five phase names as keys", () => {
		for (const phase of ALL_PHASES) {
			expect(MVP_PHASE_MODEL_MAP).toHaveProperty(phase);
		}
	});

	it("exploring maps to null (stay on current model)", () => {
		expect(MVP_PHASE_MODEL_MAP.exploring).toBeNull();
	});

	it("planning maps to deepseek/deepseek-v4-flash@openrouter", () => {
		expect(MVP_PHASE_MODEL_MAP.planning).toBe("deepseek/deepseek-v4-flash@openrouter");
	});

	it("executing maps to z-ai/glm-5.2@openrouter", () => {
		expect(MVP_PHASE_MODEL_MAP.executing).toBe("z-ai/glm-5.2@openrouter");
	});

	it("verifying maps to minimax/minimax-m3@openrouter", () => {
		expect(MVP_PHASE_MODEL_MAP.verifying).toBe("minimax/minimax-m3@openrouter");
	});

	it("reporting maps to google/gemma-4-31b-it@openrouter", () => {
		expect(MVP_PHASE_MODEL_MAP.reporting).toBe("google/gemma-4-31b-it@openrouter");
	});

	it("all values are either null or a non-empty string", () => {
		for (const [, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value === null) continue;
			expect(typeof value).toBe("string");
			expect(value.length).toBeGreaterThan(0);
		}
	});

	it("all non-null model values specify an explicit provider via @provider", () => {
		for (const [, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value === null) continue;
			expect(value).toMatch(/@\w+/);
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

	it("maps planning to deepseek/deepseek-v4-flash@openrouter", () => {
		expect(getModelForPhase("planning", MVP_PHASE_MODEL_MAP)).toBe("deepseek/deepseek-v4-flash@openrouter");
	});

	it("maps all non-null phases to the correct model IDs", () => {
		const expected: Record<string, string | null> = {
			exploring: null,
			planning: "deepseek/deepseek-v4-flash@openrouter",
			executing: "z-ai/glm-5.2@openrouter",
			verifying: "minimax/minimax-m3@openrouter",
			reporting: "google/gemma-4-31b-it@openrouter",
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
			verifying: null,
			reporting: "custom-reporter:cloud",
		};
		expect(getModelForPhase("exploring", customMap)).toBe("custom-explorer:cloud");
		expect(getModelForPhase("planning", customMap)).toBeNull();
		expect(getModelForPhase("reporting", customMap)).toBe("custom-reporter:cloud");
	});
});
