import { describe, expect, it } from "vitest";
import type { PhaseName } from "./semblr-config.ts";
import { getModelForPhase, MVP_PHASE_MODEL_MAP } from "./phase-model-map.ts";

const ALL_PHASES: PhaseName[] = ["thinking", "executing", "stuck", "reporting", "reviewing", "verifying"];

describe("MVP_PHASE_MODEL_MAP", () => {
	it("has entries for all six phases", () => {
		for (const phase of ALL_PHASES) {
			expect(phase in MVP_PHASE_MODEL_MAP).toBe(true);
		}
	});

	it("has exactly six keys", () => {
		expect(Object.keys(MVP_PHASE_MODEL_MAP)).toHaveLength(6);
	});

	it("maps thinking to null (no switch)", () => {
		expect(MVP_PHASE_MODEL_MAP.thinking).toBeNull();
	});

	it("maps executing to glm-5.2:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.executing).toBe("glm-5.2:cloud");
	});

	it("maps stuck to kimi-k2.6:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.stuck).toBe("kimi-k2.6:cloud");
	});

	it("maps reporting to gemma4:12b:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.reporting).toBe("gemma4:12b:cloud");
	});

	it("maps reviewing to deepseek-v4-pro:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.reviewing).toBe("deepseek-v4-pro:cloud");
	});

	it("maps verifying to deepseek-v4-flash:cloud", () => {
		expect(MVP_PHASE_MODEL_MAP.verifying).toBe("deepseek-v4-flash:cloud");
	});

	it("all values are either null or a non-empty string", () => {
		for (const [phase, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value === null) continue;
			expect(typeof value).toBe("string");
			expect((value as string).length).toBeGreaterThan(0);
		}
	});

	it("all cloud model values end with :cloud", () => {
		for (const [phase, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value === null) continue;
			expect(value).toMatch(/:cloud$/);
		}
	});



	it("thinking is the only null-mapped phase in the MVP map", () => {
		const nullPhases = Object.entries(MVP_PHASE_MODEL_MAP)
			.filter(([, value]) => value === null)
			.map(([phase]) => phase);
		expect(nullPhases).toEqual(["thinking"]);
	});

	it("no phase maps to an empty string", () => {
		for (const [phase, value] of Object.entries(MVP_PHASE_MODEL_MAP)) {
			if (value !== null) {
				expect(value!.length).toBeGreaterThan(0);
			}
		}
	});

	it("phase strings in the map match the PhaseName union exactly", () => {
		const mapKeys = Object.keys(MVP_PHASE_MODEL_MAP).sort();
		expect(mapKeys).toEqual([...ALL_PHASES].sort());
	});
});

describe("getModelForPhase", () => {
	const customMap = {
		thinking: null,
		executing: "custom-exec:cloud",
		stuck: null,
		reporting: "custom-report:cloud",
		reviewing: "custom-review:cloud",
		verifying: null,
	} as const satisfies Record<PhaseName, string | null>;

	it("returns the model ID for a mapped phase", () => {
		expect(getModelForPhase("executing", customMap)).toBe("custom-exec:cloud");
		expect(getModelForPhase("reporting", customMap)).toBe("custom-report:cloud");
		expect(getModelForPhase("reviewing", customMap)).toBe("custom-review:cloud");
	});

	it("returns null for a phase mapped to null", () => {
		expect(getModelForPhase("thinking", customMap)).toBeNull();
		expect(getModelForPhase("stuck", customMap)).toBeNull();
		expect(getModelForPhase("verifying", customMap)).toBeNull();
	});

	it("returns null for a missing phase (undefined fallback)", () => {
		const partialMap = { executing: "model:cloud" } as unknown as Record<PhaseName, string | null>;
		expect(getModelForPhase("thinking", partialMap)).toBeNull();
	});

	it("works with the MVP map for all phases", () => {
		expect(getModelForPhase("thinking", MVP_PHASE_MODEL_MAP)).toBeNull();
		expect(getModelForPhase("executing", MVP_PHASE_MODEL_MAP)).toBe("glm-5.2:cloud");
		expect(getModelForPhase("stuck", MVP_PHASE_MODEL_MAP)).toBe("kimi-k2.6:cloud");
		expect(getModelForPhase("reporting", MVP_PHASE_MODEL_MAP)).toBe("gemma4:12b:cloud");
		expect(getModelForPhase("reviewing", MVP_PHASE_MODEL_MAP)).toBe("deepseek-v4-pro:cloud");
		expect(getModelForPhase("verifying", MVP_PHASE_MODEL_MAP)).toBe("deepseek-v4-flash:cloud");
	});

	it("returns null for unknown phase names that are not in the map", () => {
		// @ts-expect-error — testing runtime behaviour with a non-PhaseName string
		expect(getModelForPhase("planning", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error
		expect(getModelForPhase("exploring", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error
		expect(getModelForPhase("done", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("returns null for undefined input (edge case)", () => {
		// @ts-expect-error — testing runtime behaviour
		expect(getModelForPhase(undefined, MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("getModelForPhase is a pure function (no side effects)", () => {
		// Calling getModelForPhase should not mutate the map
		const mapBefore = { ...MVP_PHASE_MODEL_MAP };
		getModelForPhase("executing", MVP_PHASE_MODEL_MAP);
		getModelForPhase("thinking", MVP_PHASE_MODEL_MAP);
		expect(MVP_PHASE_MODEL_MAP).toEqual(mapBefore);
	});

	it("getModelForPhase does not mutate the provided map", () => {
		const map = { ...MVP_PHASE_MODEL_MAP };
		const snapshot = JSON.stringify(map);
		getModelForPhase("executing", map);
		expect(JSON.stringify(map)).toBe(snapshot);
	});
});
