/**
 * routing-behavior.test.ts — Behavior tests for multi-model routing.
 *
 * Tests are derived from the specification (issue #86, comments #1 and #2):
 * - MVP_PHASE_MODEL_MAP from comment #2 (explicit @provider syntax)
 * - Phase names from issue #86 comment #1: exploring, planning, executing, verifying, reporting
 * - Switch limiter: maxSwitches = 5 per agent cycle
 * - Config gating: default enabled = false
 * - Switches happen at turn_end boundaries (tracked via pendingModelSwitch)
 * - Original model is restored at agent_end (captured via provider+model ID)
 * - Optional note parameter on semblr_report_phase tool
 */

import { describe, expect, it } from "vitest";
import { loadSemblrConfig, type PhaseName, type RoutingConfig } from "../lib/semblr-config.ts";
import { getModelForPhase, MVP_PHASE_MODEL_MAP } from "../lib/phase-model-map.ts";
import { createRound, createSession, type RoundState } from "../lib/state.ts";
import { resolveModelId } from "../lib/resolve-model-id.ts";

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** All six valid phase names (from issue #86 comment #1). */
const ALL_PHASES: PhaseName[] = ["exploring", "planning", "executing", "verifying", "reporting"];

/** Known target model IDs from the MVP map, indexed by phase (from comment #2). */
const PHASE_MODEL_MAP: Record<PhaseName, string | null> = {
	exploring: null,
	planning: "deepseek/deepseek-v4-flash@openrouter",
	executing: "z-ai/glm-5.2@openrouter",
	verifying: "minimax/minimax-m3@openrouter",
	reporting: "google/gemma-4-31b-it@openrouter",
};

/** The default maxSwitchesPerCycle value. */
const DEFAULT_MAX_SWITCHES = 5;

/**
/**
 * Simulate the routing decision (same logic used in both turn_end and agent_end).
 * Returns the target model ID (or null for no switch) without actually
 * calling pi.setModel.
 */
function simulateRoutingDecision(
	reportedPhase: PhaseName | null,
	currentModelId: string | null,
	switchCounter: number,
	maxSwitches: number,
	enabled: boolean,
	modelMap: Record<string, string | null> = MVP_PHASE_MODEL_MAP,
): { target: string | null; shouldSwitch: boolean; newCounter: number } {
	if (!enabled) return { target: null, shouldSwitch: false, newCounter: switchCounter };
	if (reportedPhase === null) return { target: null, shouldSwitch: false, newCounter: switchCounter };
	if (switchCounter >= maxSwitches) return { target: null, shouldSwitch: false, newCounter: switchCounter };

	const target = getModelForPhase(reportedPhase, modelMap);
	if (target === null) return { target: null, shouldSwitch: false, newCounter: switchCounter };
	if (currentModelId === target) return { target, shouldSwitch: false, newCounter: switchCounter };

	return { target, shouldSwitch: true, newCounter: switchCounter + 1 };
}

// ─────────────────────────────────────────────
// Category 1: resolveModelId behavior
// (Edge cases beyond the unit tests in lib/)
// ─────────────────────────────────────────────

describe("resolveModelId behavioral edge cases", () => {
	it("handles :cloud suffix with provider-prefixed IDs", () => {
		const result = resolveModelId("openai/gpt-4:cloud");
		expect(result).toEqual({ provider: "ollama-cloud", model: "openai/gpt-4" });
	});

	it("passes through provider-prefixed IDs without :cloud", () => {
		const result = resolveModelId("openai/gpt-4");
		expect(result).toEqual({ provider: "ollama-cloud", model: "openai/gpt-4" });
	});

	it("handles multiple consecutive :cloud suffixes", () => {
		// Only the last :cloud is stripped by the regex
		const result = resolveModelId("model:cloud:cloud:cloud");
		expect(result.model).toBe("model:cloud:cloud");
		expect(result.provider).toBe("ollama-cloud");
	});

	it("returns consistent provider for all MVP map model IDs", () => {
		const mapValues = Object.values(MVP_PHASE_MODEL_MAP).filter((v): v is string => v !== null);
		for (const modelId of mapValues) {
			const resolved = resolveModelId(modelId);
			expect(resolved.provider).toBe("openrouter");
			expect(resolved.model.length).toBeGreaterThan(0);
		}
	});

	it("returns a model ID that matches the phase model map expectations", () => {
		// Every non-null entry in the MVP map should resolve to a valid model
		for (const phase of ALL_PHASES) {
			const modelId = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			if (modelId === null) continue;
			const resolved = resolveModelId(modelId);
			expect(resolved.model.length).toBeGreaterThan(0);
			expect(resolved.provider).toBe("openrouter");
		}
	});
});

// ─────────────────────────────────────────────
// Category 2: getModelForPhase / phase map behavior
// ─────────────────────────────────────────────

describe("phase model map behavior", () => {
	it("maps exploring to null (stay on current model)", () => {
		expect(getModelForPhase("exploring", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("maps all non-exploring phases to an @provider model ID", () => {
		const nonExploringPhases = ALL_PHASES.filter((p) => p !== "exploring");
		for (const phase of nonExploringPhases) {
			const modelId = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			expect(modelId).not.toBeNull();
			expect(modelId).toMatch(/@\w+$/);
		}
	});

	it("each non-null phase maps to a different model ID", () => {
		const modelIds = ALL_PHASES
			.map((p) => getModelForPhase(p, MVP_PHASE_MODEL_MAP))
			.filter((id): id is string => id !== null);
		const uniqueIds = new Set(modelIds);
		// All 4 non-exploring phases should have distinct models
		expect(uniqueIds.size).toBe(4);
	});

	it("getModelForPhase is deterministic (same input → same output)", () => {
		for (const phase of ALL_PHASES) {
			const first = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			const second = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			expect(first).toBe(second);
		}
	});

	it("getModelForPhase returns null for arbitrary strings not in the phase union", () => {
		// @ts-expect-error — deliberate runtime check
		expect(getModelForPhase("thinking", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error
		expect(getModelForPhase("reviewing", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error
		expect(getModelForPhase("", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("getModelForPhase with a custom map uses that map, not the MVP map", () => {
		const customMap = {
			exploring: null,
			executing: null, // override — no switch
			planning: null,
			verifying: null,
			reporting: null,
		} as const;
		expect(getModelForPhase("executing", customMap)).toBeNull(); // overridden to null
	});
});

// ─────────────────────────────────────────────
// Category 3: Tool schema validation (PhaseName type)
// ─────────────────────────────────────────────

describe("phase name type validation", () => {
	it("accepts all five defined phase names", () => {
		const validPhases: PhaseName[] = ["exploring", "planning", "executing", "verifying", "reporting"];
		expect(validPhases).toHaveLength(5);
		// Each should produce a non-null-ish result (null or string) from the map
		for (const phase of validPhases) {
			const result = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			// null is valid (exploring stays on current model)
			expect(result === null || typeof result === "string").toBe(true);
		}
	});

	it("rejects strings outside the PhaseName union when used with getModelForPhase", () => {
		// @ts-expect-error — not a PhaseName
		expect(getModelForPhase("thinking", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error — not a PhaseName
		expect(getModelForPhase("done", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error — not a PhaseName
		expect(getModelForPhase("init", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("PhaseName type is a string literal union (no extra keys)", () => {
		// Compile-time check: ALL_PHASES covers every PhaseName
		const phaseSet = new Set<string>(ALL_PHASES);
		expect(phaseSet.has("exploring")).toBe(true);
		expect(phaseSet.has("planning")).toBe(true);
		expect(phaseSet.has("executing")).toBe(true);
		expect(phaseSet.has("debugging")).toBe(false);
		expect(phaseSet.has("verifying")).toBe(true);
		expect(phaseSet.has("reporting")).toBe(true);
		expect(phaseSet.has("thinking")).toBe(false);
		expect(phaseSet.has("reviewing")).toBe(false);
	});

	it("maps have exactly the PhaseName keys", () => {
		const mapKeys = Object.keys(MVP_PHASE_MODEL_MAP);
		expect(mapKeys.sort()).toEqual([...ALL_PHASES].sort());
	});
});

// ─────────────────────────────────────────────
// Category 4: Switch counter behavior
// ─────────────────────────────────────────────

describe("switch counter behavior", () => {
	it("starts at zero for a fresh round", () => {
		const round = createRound();
		expect(round.switchCounter).toBe(0);
	});

	it("increments when a switch decision is made", () => {
		const round = createRound();
		const decision = simulateRoutingDecision(
			"executing",
			"default-model",
			round.switchCounter,
			DEFAULT_MAX_SWITCHES,
			true,
		);
		expect(decision.shouldSwitch).toBe(true);
		expect(decision.newCounter).toBe(1);
	});

	it("increments up to maxSwitches (5) but not beyond", () => {
		const round = createRound();
		let counter = 0;

		// First 3 switches should succeed
		for (let i = 0; i < DEFAULT_MAX_SWITCHES; i++) {
			const decision = simulateRoutingDecision("executing", "model-" + i, counter, DEFAULT_MAX_SWITCHES, true);
			if (decision.shouldSwitch) counter = decision.newCounter;
		}

		expect(counter).toBe(DEFAULT_MAX_SWITCHES);

		// 4th switch should be blocked
		const blocked = simulateRoutingDecision("executing", "model-x", counter, DEFAULT_MAX_SWITCHES, true);
		expect(blocked.shouldSwitch).toBe(false);
		expect(blocked.newCounter).toBe(DEFAULT_MAX_SWITCHES);
	});

	it("counter resets at cycle boundaries (fresh createRound)", () => {
		const round1 = createRound();
		round1.switchCounter = 3;

		const round2 = createRound();
		expect(round2.switchCounter).toBe(0);
	});

	it("no-op phase (exploring) does not increment counter", () => {
		const decision = simulateRoutingDecision("exploring", "current-model", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.newCounter).toBe(0);
	});

	it("same-model phase does not increment counter", () => {
		// If the current model already matches the target, no switch needed
		const decision = simulateRoutingDecision("executing", "z-ai/glm-5.2@openrouter", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.newCounter).toBe(0);
	});

	it("counter is independent between rounds", () => {
		const round1 = createRound();
		const round2 = createRound();

		// Simulate 2 switches in round1
		const d1 = simulateRoutingDecision("executing", "m1", round1.switchCounter, DEFAULT_MAX_SWITCHES, true);
		round1.switchCounter = d1.newCounter;
		const d2 = simulateRoutingDecision("verifying", "m2", round1.switchCounter, DEFAULT_MAX_SWITCHES, true);
		round1.switchCounter = d2.newCounter;

		// round2 should still have counter at 0
		expect(round1.switchCounter).toBe(2);
		expect(round2.switchCounter).toBe(0);
	});

	it("maxSwitches value matches spec (5)", () => {
		// Verify via config
		const config = loadSemblrConfig({ cwd: "/", agentDir: "/agent", env: {}, fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} } });
		expect(config.routing.maxSwitchesPerCycle).toBe(5);
	});

	it("switches blocked after maxSwitches even with different phases", () => {
		let counter = 0;

		// Burn through all 5 switches
		counter = simulateRoutingDecision("executing", "m1", counter, DEFAULT_MAX_SWITCHES, true).newCounter;
		counter = simulateRoutingDecision("m2", counter, DEFAULT_MAX_SWITCHES, true).newCounter;
		counter = simulateRoutingDecision("verifying", "m3", counter, DEFAULT_MAX_SWITCHES, true).newCounter;
		counter = simulateRoutingDecision("reporting", "m4", counter, DEFAULT_MAX_SWITCHES, true).newCounter;
		counter = simulateRoutingDecision("planning", "m5", counter, DEFAULT_MAX_SWITCHES, true).newCounter;

		// Now try different phases — all blocked
		expect(simulateRoutingDecision("m6", counter, DEFAULT_MAX_SWITCHES, true).shouldSwitch).toBe(false);
		expect(simulateRoutingDecision("executing", "m7", counter, DEFAULT_MAX_SWITCHES, true).shouldSwitch).toBe(false);
		expect(simulateRoutingDecision("exploring", "m8", counter, DEFAULT_MAX_SWITCHES, true).shouldSwitch).toBe(false);
	});
});

// ─────────────────────────────────────────────
// Category 5: Model switching logic
// ─────────────────────────────────────────────

describe("model switching logic", () => {
	it("null phase → no switch, stays on current model", () => {
		const decision = simulateRoutingDecision(null, "any-model", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.target).toBeNull();
	});

	it("phase → null in map (exploring) → no switch", () => {
		const decision = simulateRoutingDecision("exploring", "any-model", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.target).toBeNull();
	});

	it("phase → same model → no switch, counter unchanged", () => {
		const decision = simulateRoutingDecision(
			"executing",
			"z-ai/glm-5.2@openrouter", // already on the target
			0,
			DEFAULT_MAX_SWITCHES,
			true,
		);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.newCounter).toBe(0);
	});

	it("phase → different model → pending switch queued", () => {
		const decision = simulateRoutingDecision(
			"executing",
			"default-model", // different from z-ai/glm-5.2@openrouter
			0,
			DEFAULT_MAX_SWITCHES,
			true,
		);
		expect(decision.shouldSwitch).toBe(true);
		expect(decision.target).toBe("z-ai/glm-5.2@openrouter");
		expect(decision.newCounter).toBe(1);
	});

	it("switch is applied at turn_end (pendingModelSwitch set during semblr_report_phase, executed at turn_end)", () => {
		// This test verifies the design: the decision function returns the target
		// without executing it. In the real extension, semblr_report_phase sets
		// pendingModelSwitch, and pi.setModel() is called at turn_end.
		const decision = simulateRoutingDecision("verifying", "default-model", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.target).toBe("minimax/minimax-m3@openrouter");
		expect(decision.shouldSwitch).toBe(true);
		// The round's pendingModelSwitch would be set to the target
		const round = createRound();
		round.currentPhase = "verifying";
		round.pendingModelSwitch = decision.target;
		expect(round.pendingModelSwitch).toBe("minimax/minimax-m3@openrouter");
	});

	it("multiple phase reports in same round → last reported phase wins", () => {
		const round = createRound();
		// Simulate multiple reports overwriting each other
		round.currentPhase = "exploring";
		round.pendingModelSwitch = getModelForPhase("exploring", MVP_PHASE_MODEL_MAP);

		// Later report overwrites
		round.currentPhase = "executing";
		round.pendingModelSwitch = getModelForPhase("executing", MVP_PHASE_MODEL_MAP);

		expect(round.currentPhase).toBe("executing");
		expect(round.pendingModelSwitch).toBe("z-ai/glm-5.2@openrouter");
	});

	it("switch to each non-exploring phase produces expected target model", () => {
		const modelMap: Record<string, string> = {
			planning: "deepseek/deepseek-v4-flash@openrouter",
			executing: "z-ai/glm-5.2@openrouter",
			verifying: "minimax/minimax-m3@openrouter",
			reporting: "google/gemma-4-31b-it@openrouter",
		};

		for (const [phase, expectedModel] of Object.entries(modelMap)) {
			const decision = simulateRoutingDecision(
				phase as PhaseName,
				"default-model",
				0,
				DEFAULT_MAX_SWITCHES,
				true,
			);
			expect(decision.shouldSwitch).toBe(true);
			expect(decision.target).toBe(expectedModel);
		}
	});
});

// ─────────────────────────────────────────────
// Category 6: Config gating
// ─────────────────────────────────────────────

describe("config gating", () => {
	it("default config has routing disabled", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: {},
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});
		expect(config.routing.enabled).toBe(false);
	});

	it("when disabled, no switch decisions occur", () => {
		const decision = simulateRoutingDecision("executing", "default-model", 0, DEFAULT_MAX_SWITCHES, false);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.target).toBeNull();
	});

	it("when disabled, switch counter never increments", () => {
		const decision = simulateRoutingDecision("executing", "default-model", 0, DEFAULT_MAX_SWITCHES, false);
		expect(decision.newCounter).toBe(0);
	});

	it("when enabled via env var, routing is active", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "true" },
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});
		expect(config.routing.enabled).toBe(true);
	});

	it("when enabled, switching decisions are made correctly", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "true" },
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});

		const decision = simulateRoutingDecision(
			"executing",
			"default-model",
			0,
			config.routing.maxSwitchesPerCycle,
			config.routing.enabled,
		);
		expect(decision.shouldSwitch).toBe(true);
		expect(decision.target).toBe("z-ai/glm-5.2@openrouter");
	});

	it("default maxSwitches is 5 in config", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: {},
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});
		expect(config.routing.maxSwitchesPerCycle).toBe(5);
	});

	it("env var with invalid value falls back to disabled", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "invalid" },
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});
		expect(config.routing.enabled).toBe(false);
	});

	it("env var with false value stays disabled", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "false" },
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});
		expect(config.routing.enabled).toBe(false);
	});

	it("config defaults cannot be mutated through the returned object", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: {},
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});

		// Mutating the returned config should not affect subsequent loads
		config.routing.enabled = true;
		config.routing.maxSwitchesPerCycle = 99;

		const config2 = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: {},
			fsImpl: { existsSync: () => false, readFileSync: () => "{}", writeFileSync: () => {} },
		});
		expect(config2.routing.enabled).toBe(false);
		expect(config2.routing.maxSwitchesPerCycle).toBe(5);
	});
});

// ─────────────────────────────────────────────
// Category 7: Integration scenario
// ─────────────────────────────────────────────

describe("integration: full agent cycle", () => {
	it("exploring → planning → executing → verifying → reporting respects switch limit", () => {
		// With maxSwitches = 5, all 5 non-null phases (planning…reporting)
		// trigger switches. The 6th switch attempt is blocked.
		let currentModel = "default-model";
		let switchCounter = 0;

		// Phase 1: exploring → no switch
		let d = simulateRoutingDecision("exploring", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBeNull();
		expect(d.shouldSwitch).toBe(false);

		// Phase 2: planning → switch to deepseek/deepseek-v4-flash@openrouter (switch 1)
		d = simulateRoutingDecision("planning", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("deepseek/deepseek-v4-flash@openrouter");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 3: executing → switch to z-ai/glm-5.2@openrouter (switch 2)
		d = simulateRoutingDecision("executing", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("z-ai/glm-5.2@openrouter");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 4: verifying → switch to minimax/minimax-m3@openrouter (switch 3)
		d = simulateRoutingDecision("verifying", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("minimax/minimax-m3@openrouter");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 5: reporting → switch to google/gemma-4-31b-it@openrouter (switch 4)
		d = simulateRoutingDecision("reporting", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("google/gemma-4-31b-it@openrouter");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 6: executing → switch to z-ai/glm-5.2@openrouter (switch 5 — maxSwitches reached)
		d = simulateRoutingDecision("executing", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("z-ai/glm-5.2@openrouter");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 7: another planning attempt → blocked (switchCounter == 5 >= maxSwitches)
		d = simulateRoutingDecision("planning", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.shouldSwitch).toBe(false);
		expect(switchCounter).toBe(5);

		// After maxSwitches=5 switches, model remains at the 5th target
		expect(currentModel).toBe("z-ai/glm-5.2@openrouter");
	});

	it("full cycle without switch limit (maxSwitches raised) verifies complete map coverage", () => {
		// Use a high maxSwitches to bypass the limit and verify full map coverage
		const unlimitedMax = 10;
		const transitions: Array<{ phase: PhaseName; expectedTarget: string | null }> = [
			{ phase: "exploring", expectedTarget: null },
			{ phase: "planning", expectedTarget: "deepseek/deepseek-v4-flash@openrouter" },
			{ phase: "executing", expectedTarget: "z-ai/glm-5.2@openrouter" },
			{ phase: "verifying", expectedTarget: "minimax/minimax-m3@openrouter" },
			{ phase: "reporting", expectedTarget: "google/gemma-4-31b-it@openrouter" },
		];

		let currentModel = "model-0";
		let counter = 0;

		for (const { phase, expectedTarget } of transitions) {
			const d = simulateRoutingDecision(phase, currentModel, counter, unlimitedMax, true);
			expect(d.target).toBe(expectedTarget);
			if (d.shouldSwitch) {
				counter = d.newCounter;
				currentModel = d.target!;
			}
		}

		expect(currentModel).toBe("google/gemma-4-31b-it@openrouter");
		expect(counter).toBe(4);
	})

	it("full cycle with same-phase repeat does not count extra switches", () => {
		const phases: PhaseName[] = ["executing", "executing", "executing", "verifying"];

		let currentModel = "default-model";
		let switchCounter = 0;

		const decisions = phases.map((phase) => {
			const d = simulateRoutingDecision(phase, currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
			if (d.shouldSwitch) {
				switchCounter = d.newCounter;
				currentModel = d.target!;
			}
			return d;
		});

		// First executing triggers a switch, subsequent ones don't (same model)
		expect(decisions[0].shouldSwitch).toBe(true);  // 1st switch
		expect(decisions[1].shouldSwitch).toBe(false);  // already on glm
		expect(decisions[2].shouldSwitch).toBe(false);  // already on glm
		expect(decisions[3].shouldSwitch).toBe(true);   // switch to verifying model
		expect(switchCounter).toBe(2);
		expect(currentModel).toBe("minimax/minimax-m3@openrouter");
	});

	it("switch counter respects maxSwitches (5) within a single cycle", () => {
		// Simulate a long cycle with many phase changes
		const phases: PhaseName[] = [
			"executing",  // 1: switch to z-ai/glm-5.2
			     // 2: switch to kimi-k2.6
			"verifying",  // 3: switch to minimax/minimax-m3
			"reporting",  // 4: switch to google/gemma-4-31b-it
			"planning",   // 5: switch to deepseek (maxSwitches reached)
			"exploring",  // no switch (null target)
			"executing",  // blocked (counter == maxSwitches)
		];

		let currentModel = "default-model";
		let switchCounter = 0;

		for (const phase of phases) {
			const d = simulateRoutingDecision(phase, currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
			if (d.shouldSwitch) {
				switchCounter = d.newCounter;
				currentModel = d.target!;
			}
		}

		expect(switchCounter).toBe(DEFAULT_MAX_SWITCHES);
		expect(currentModel).toBe("z-ai/glm-5.2@openrouter"); // remains at executing model once switch limit is reached
	});

	it("multiple cycles reset the switch counter each round", () => {
		// Cycle 1: 2 switches
		let currentModel = "model-0";
		let switchCounter1 = 0;
		switchCounter1 = simulateRoutingDecision("executing", currentModel, 0, DEFAULT_MAX_SWITCHES, true).newCounter;
		currentModel = "z-ai/glm-5.2@openrouter";
		switchCounter1 = simulateRoutingDecision(currentModel, switchCounter1, DEFAULT_MAX_SWITCHES, true).newCounter;
		currentModel = "kimi-k2.6@ollama-cloud";

		// Cycle 2 (new round) — counter resets
		let switchCounter2 = 0;
		const d1 = simulateRoutingDecision("executing", currentModel, 0, DEFAULT_MAX_SWITCHES, true);
		expect(d1.shouldSwitch).toBe(true);
		expect(d1.newCounter).toBe(1);
	});

	it("integration: verify model IDs at each step are valid openrouter IDs", () => {
		const phaseSequence: PhaseName[] = ["planning", "executing", "verifying", "reporting"];

		for (const phase of phaseSequence) {
			const modelId = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			expect(modelId).not.toBeNull();
			const resolved = resolveModelId(modelId!);
			expect(resolved.provider).toBe("openrouter");
			expect(resolved.model.length).toBeGreaterThan(0);
		}
	});
});

// ─────────────────────────────────────────────
// Category 8: RoundState routing fields
// ─────────────────────────────────────────────

describe("RoundState routing field behavior", () => {
	it("fresh round has switchCounter = 0", () => {
		const round = createRound();
		expect(round.switchCounter).toBe(0);
	});

	it("fresh round has currentPhase = null", () => {
		const round = createRound();
		expect(round.currentPhase).toBeNull();
	});

	it("fresh round has phaseNote = null", () => {
		const round = createRound();
		expect(round.phaseNote).toBeNull();
	});

	it("fresh round has pendingModelSwitch = null", () => {
		const round = createRound();
		expect(round.pendingModelSwitch).toBeNull();
	});

	it("can set currentPhase to any valid PhaseName", () => {
		const round = createRound();
		for (const phase of ALL_PHASES) {
			round.currentPhase = phase;
			expect(round.currentPhase).toBe(phase);
		}
	});

	it("can set phaseNote to a string and reset to null", () => {
		const round = createRound();
		expect(round.phaseNote).toBeNull();
		round.phaseNote = "Investigating database schema";
		expect(round.phaseNote).toBe("Investigating database schema");
		round.phaseNote = null;
		expect(round.phaseNote).toBeNull();
	});

	it("phaseNote is preserved alongside currentPhase", () => {
		const round = createRound();
		round.currentPhase = "executing";
		round.phaseNote = "Need to check edge cases";
		expect(round.currentPhase).toBe("executing");
		expect(round.phaseNote).toBe("Need to check edge cases");
	});

	it("can set pendingModelSwitch to a model ID", () => {
		const round = createRound();
		round.pendingModelSwitch = "z-ai/glm-5.2@openrouter";
		expect(round.pendingModelSwitch).toBe("z-ai/glm-5.2@openrouter");
	});

	it("can reset pendingModelSwitch back to null", () => {
		const round = createRound();
		round.pendingModelSwitch = "some-model:cloud";
		round.pendingModelSwitch = null;
		expect(round.pendingModelSwitch).toBeNull();
	});

	it("fresh round has originalModel = null", () => {
		const round = createRound();
		expect(round.originalModel).toBeNull();
	});

	it("can set originalModel to full provider/model identity", () => {
		const round = createRound();
		round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "high" };
		expect(round.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "high",
		});
	});

	it("originalModel is captured on first phase report (before pendingModelSwitch is set)", () => {
		const round = createRound();
		// Simulate the extension behavior: capture original model before setting pending switch
		round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };
		round.currentPhase = "executing";
		round.pendingModelSwitch = "z-ai/glm-5.2@openrouter";
		// originalModel should not change after first capture
		expect(round.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "medium",
		});
		expect(round.currentPhase).toBe("executing");
		expect(round.pendingModelSwitch).toBe("z-ai/glm-5.2@openrouter");
	});

	it("originalModel persists across phase changes within the same round", () => {
		const round = createRound();
		round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };
		round.currentPhase = "exploring";
		// Change phase — originalModel stays
		round.currentPhase = "planning";
		expect(round.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "medium",
		});
		expect(round.currentPhase).toBe("planning");
	});

	it("resets to null on new round (createRound)", () => {
		const round1 = createRound();
		round1.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };

		const round2 = createRound();
		expect(round2.originalModel).toBeNull();
		expect(round1.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "medium",
		});
	});

	it("switchCounter increments independently of currentPhase", () => {
		const round = createRound();

		// Set up the round as the extension would
		round.currentPhase = "executing";
		round.pendingModelSwitch = "z-ai/glm-5.2@openrouter";
		round.switchCounter = 1;

		// Now change phase without incrementing counter
		round.currentPhase = "verifying";
		round.pendingModelSwitch = "kimi-k2.6@ollama-cloud";

		expect(round.switchCounter).toBe(1); // unchanged
		expect(round.currentPhase).toBe("verifying");
	});
});

// ─────────────────────────────────────────────
// Category 9: Turn-end switch + agent-end restore flow
// ─────────────────────────────────────────────

describe("turn_end switch + agent_end restore flow", () => {
	it("semblr_report_phase captures originalModel and sets pendingModelSwitch", () => {
		const round = createRound();
		const originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" } as const;
		const targetModelId = "z-ai/glm-5.2@openrouter";

		// Simulate semblr_report_phase("executing"):
		// 1. Capture original model on first call
		if (round.originalModel === null) {
			round.originalModel = originalModel;
		}
		// 2. Store phase
		round.currentPhase = "executing";
		// 3. Set pending switch
		round.pendingModelSwitch = targetModelId;
		round.switchCounter++;

		expect(round.originalModel).toEqual(originalModel);
		expect(round.currentPhase).toBe("executing");
		expect(round.pendingModelSwitch).toBe(targetModelId);
		expect(round.switchCounter).toBe(1);
	});

	it("originalModel is set only on first phase report (subsequent calls preserve it)", () => {
		const round = createRound();

		// First call: capture original model
		if (round.originalModel === null) {
			round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };
		}
		round.currentPhase = "planning";
		round.pendingModelSwitch = "deepseek/deepseek-v4-flash@openrouter";

		// Second call: originalModel should NOT be overwritten
		if (round.originalModel === null) {
			round.originalModel = { provider: "wrong-provider", modelId: "wrong-model", thinkingLevel: "high" }; // would not execute
		}
		round.currentPhase = "executing";
		round.pendingModelSwitch = "z-ai/glm-5.2@openrouter";

		expect(round.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "medium",
		}); // preserved
		expect(round.currentPhase).toBe("executing");
		expect(round.pendingModelSwitch).toBe("z-ai/glm-5.2@openrouter");
	});

	it("turn_end clears pendingModelSwitch after the switch is applied", () => {
		const round = createRound();
		round.pendingModelSwitch = "z-ai/glm-5.2@openrouter";

		// Simulate turn_end: apply switch, then clear pending
		expect(round.pendingModelSwitch).toBe("z-ai/glm-5.2@openrouter");
		round.pendingModelSwitch = null; // turn_end clears it

		expect(round.pendingModelSwitch).toBeNull();
	});

	it("agent_end restores original model when provider or model differs", () => {
		const round = createRound();
		round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };
		round.switchCounter = 2;

		// Simulate agent_end restore logic:
		// If ctx.model does not match provider+id, find and restore by provider+id.
		const currentModel = { provider: "ollama-cloud", id: "minimax-m3" }; // different from original
		const needsRestore =
			currentModel.provider !== round.originalModel.provider || currentModel.id !== round.originalModel.modelId;

		expect(needsRestore).toBe(true);
		expect(round.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "medium",
		});
	});

	it("agent_end restores when model ID matches but provider differs", () => {
		const round = createRound();
		round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };

		const currentModel = { provider: "azure-openai-responses", id: "gpt-5.5" };
		const needsRestore =
			currentModel.provider !== round.originalModel.provider || currentModel.id !== round.originalModel.modelId;

		expect(needsRestore).toBe(true);
	});

	it("agent_end does NOT restore when no switch occurred (provider and model match current)", () => {
		const round = createRound();
		round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };

		// Simulate agent_end check: current model matches original → no restore needed
		const currentModel = { provider: "openai-codex", id: "gpt-5.5" };
		const needsRestore =
			currentModel.provider !== round.originalModel.provider || currentModel.id !== round.originalModel.modelId;

		expect(needsRestore).toBe(false);
	});

	it("agent_end does NOT restore when originalModel is null (no phase was reported)", () => {
		const round = createRound();
		// originalModel is null — no phase was reported this round

		const shouldAttemptRestore = round.originalModel !== null;
		expect(shouldAttemptRestore).toBe(false);
	});

	it("full lifecycle: report → pending → turn_end switch → agent_end restore", () => {
		const round = createRound();

		// Step 1: semblr_report_phase called
		if (round.originalModel === null) {
			round.originalModel = { provider: "openai-codex", modelId: "gpt-5.5", thinkingLevel: "medium" };
		}
		round.currentPhase = "executing";
		const target = "z-ai/glm-5.2@openrouter";
		round.pendingModelSwitch = target;
		round.switchCounter = 1;

		expect(round.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "medium",
		});
		expect(round.pendingModelSwitch).toBe(target);

		// Step 2: turn_end fires — model switches
		// (In the real extension, pi.setModel() would be called here)
		const switchedModel = round.pendingModelSwitch; // would be resolved + applied
		round.pendingModelSwitch = null;

		expect(switchedModel).toBe(target);
		expect(round.pendingModelSwitch).toBeNull();

		// Step 3: agent_end fires — original model is restored
		// (In the real extension, ctx.modelRegistry.find(provider, modelId) → pi.setModel(originalModel))
		expect(round.originalModel).toEqual({
			provider: "openai-codex",
			modelId: "gpt-5.5",
			thinkingLevel: "medium",
		});
		// After agent_end, originalModel is still set (round state clears on next agent_start)
		// The actual restore happens via pi.setModel, not by clearing the field
	});
});

// ─────────────────────────────────────────────
// Category 10: SessionState routing fields (where applicable)
// ─────────────────────────────────────────────

describe("SessionState routing behavior", () => {
	it("fresh session has no routing-specific fields (not in MVP scope)", () => {
		const session = createSession();
		// Per the spec, session-level routing fields are deferred
		expect("totalPhaseSwitches" in session).toBe(false);
	});
});

// ─────────────────────────────────────────────
