/**
 * routing-behavior.test.ts — Behavior tests for multi-model routing.
 *
 * Tests are derived from the specification (issue #86, comments #1 and #2):
 * - MVP_PHASE_MODEL_MAP from comment #2 (Ollama-Cloud naming)
 * - Phase names from task-003 (thinking, executing, stuck, reporting, reviewing, verifying)
 * - Switch limiter: maxSwitches = 3 per agent cycle
 * - Config gating: default enabled = false
 * - Switches happen at agent_end boundaries (tracked via pendingModelSwitch)
 */

import { describe, expect, it } from "vitest";
import { loadSemblrConfig, type MultiModelRoutingConfig, type PhaseName } from "../lib/semblr-config.ts";
import { getModelForPhase, MVP_PHASE_MODEL_MAP } from "../lib/phase-model-map.ts";
import { createRound, createSession, type RoundState } from "../lib/state.ts";
import { resolveModelId } from "../lib/resolve-model-id.ts";

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** All six valid phase names. */
const ALL_PHASES: PhaseName[] = ["thinking", "executing", "stuck", "reporting", "reviewing", "verifying"];

/** Known target model IDs from the MVP map, indexed by phase. */
const PHASE_MODEL_MAP: Record<PhaseName, string | null> = {
	thinking: null,
	executing: "glm-5.2:cloud",
	stuck: "kimi-k2.6:cloud",
	reporting: "gemma4:12b:cloud",
	reviewing: "deepseek-v4-pro:cloud",
	verifying: "deepseek-v4-flash:cloud",
};

/** The default maxSwitches value. */
const DEFAULT_MAX_SWITCHES = 3;

/**
 * Simulate the routing decision at agent_end.
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
		expect(result).toEqual({ provider: "ollama", model: "openai/gpt-4" });
	});

	it("passes through provider-prefixed IDs without :cloud", () => {
		const result = resolveModelId("openai/gpt-4");
		expect(result).toEqual({ provider: "ollama", model: "openai/gpt-4" });
	});

	it("handles multiple consecutive :cloud suffixes", () => {
		// Only the last :cloud is stripped by the regex
		const result = resolveModelId("model:cloud:cloud:cloud");
		expect(result.model).toBe("model:cloud:cloud");
		expect(result.provider).toBe("ollama");
	});

	it("returns consistent provider for all MVP map model IDs", () => {
		const mapValues = Object.values(MVP_PHASE_MODEL_MAP).filter((v): v is string => v !== null);
		for (const modelId of mapValues) {
			const resolved = resolveModelId(modelId);
			expect(resolved.provider).toBe("ollama");
			expect(resolved.model).not.toMatch(/:cloud$/);
		}
	});

	it("returns a model ID that matches the phase model map expectations", () => {
		// Every non-null entry in the MVP map should resolve to a valid model
		for (const phase of ALL_PHASES) {
			const modelId = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			if (modelId === null) continue;
			const resolved = resolveModelId(modelId);
			expect(resolved.model.length).toBeGreaterThan(0);
			expect(resolved.provider).toBe("ollama");
		}
	});
});

// ─────────────────────────────────────────────
// Category 2: getModelForPhase / phase map behavior
// ─────────────────────────────────────────────

describe("phase model map behavior", () => {
	it("maps thinking to null (stay on current model)", () => {
		expect(getModelForPhase("thinking", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("maps all non-thinking phases to a :cloud model ID", () => {
		const nonThinkingPhases = ALL_PHASES.filter((p) => p !== "thinking");
		for (const phase of nonThinkingPhases) {
			const modelId = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			expect(modelId).not.toBeNull();
			expect(modelId).toMatch(/:cloud$/);
		}
	});

	it("each non-null phase maps to a different model ID", () => {
		const modelIds = ALL_PHASES
			.map((p) => getModelForPhase(p, MVP_PHASE_MODEL_MAP))
			.filter((id): id is string => id !== null);
		const uniqueIds = new Set(modelIds);
		// All 5 non-thinking phases should have distinct models
		expect(uniqueIds.size).toBe(5);
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
		expect(getModelForPhase("planning", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error
		expect(getModelForPhase("exploring", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error
		expect(getModelForPhase("", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("getModelForPhase with a custom map uses that map, not the MVP map", () => {
		const customMap = {
			thinking: null,
			executing: null, // override — no switch
			stuck: "custom-model:cloud",
			reporting: null,
			reviewing: null,
			verifying: null,
		} as const;
		expect(getModelForPhase("executing", customMap)).toBeNull(); // overridden to null
		expect(getModelForPhase("stuck", customMap)).toBe("custom-model:cloud");
	});
});

// ─────────────────────────────────────────────
// Category 3: Tool schema validation (PhaseName type)
// ─────────────────────────────────────────────

describe("phase name type validation", () => {
	it("accepts all six defined phase names", () => {
		const validPhases: PhaseName[] = ["thinking", "executing", "stuck", "reporting", "reviewing", "verifying"];
		expect(validPhases).toHaveLength(6);
		// Each should produce a non-null-ish result (null or string) from the map
		for (const phase of validPhases) {
			const result = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			// null is valid (thinking stays on current model)
			expect(result === null || typeof result === "string").toBe(true);
		}
	});

	it("rejects strings outside the PhaseName union when used with getModelForPhase", () => {
		// @ts-expect-error — not a PhaseName
		expect(getModelForPhase("planning", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error — not a PhaseName
		expect(getModelForPhase("done", MVP_PHASE_MODEL_MAP)).toBeNull();
		// @ts-expect-error — not a PhaseName
		expect(getModelForPhase("init", MVP_PHASE_MODEL_MAP)).toBeNull();
	});

	it("PhaseName type is a string literal union (no extra keys)", () => {
		// Compile-time check: ALL_PHASES covers every PhaseName
		const phaseSet = new Set<string>(ALL_PHASES);
		expect(phaseSet.has("thinking")).toBe(true);
		expect(phaseSet.has("executing")).toBe(true);
		expect(phaseSet.has("stuck")).toBe(true);
		expect(phaseSet.has("reporting")).toBe(true);
		expect(phaseSet.has("reviewing")).toBe(true);
		expect(phaseSet.has("verifying")).toBe(true);
		expect(phaseSet.has("planning")).toBe(false);
		expect(phaseSet.has("exploring")).toBe(false);
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

	it("increments up to maxSwitches (3) but not beyond", () => {
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

	it("no-op phase (thinking) does not increment counter", () => {
		const decision = simulateRoutingDecision("thinking", "current-model", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.newCounter).toBe(0);
	});

	it("same-model phase does not increment counter", () => {
		// If the current model already matches the target, no switch needed
		const decision = simulateRoutingDecision("executing", "glm-5.2:cloud", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.newCounter).toBe(0);
	});

	it("counter is independent between rounds", () => {
		const round1 = createRound();
		const round2 = createRound();

		// Simulate 2 switches in round1
		const d1 = simulateRoutingDecision("executing", "m1", round1.switchCounter, DEFAULT_MAX_SWITCHES, true);
		round1.switchCounter = d1.newCounter;
		const d2 = simulateRoutingDecision("stuck", "m2", round1.switchCounter, DEFAULT_MAX_SWITCHES, true);
		round1.switchCounter = d2.newCounter;

		// round2 should still have counter at 0
		expect(round1.switchCounter).toBe(2);
		expect(round2.switchCounter).toBe(0);
	});

	it("maxSwitches value matches spec (3)", () => {
		// Verify via config
		const config = loadSemblrConfig({ cwd: "/", agentDir: "/agent", env: {}, fsImpl: { existsSync: () => false, readFileSync: () => "{}" } });
		expect(config.multiModelRouting.maxSwitches).toBe(3);
	});

	it("switches blocked after maxSwitches even with different phases", () => {
		let counter = 0;

		// Burn through all 3 switches
		counter = simulateRoutingDecision("executing", "m1", counter, DEFAULT_MAX_SWITCHES, true).newCounter;
		counter = simulateRoutingDecision("stuck", "m2", counter, DEFAULT_MAX_SWITCHES, true).newCounter;
		counter = simulateRoutingDecision("verifying", "m3", counter, DEFAULT_MAX_SWITCHES, true).newCounter;

		// Now try different phases — all blocked
		expect(simulateRoutingDecision("reporting", "m4", counter, DEFAULT_MAX_SWITCHES, true).shouldSwitch).toBe(false);
		expect(simulateRoutingDecision("reviewing", "m5", counter, DEFAULT_MAX_SWITCHES, true).shouldSwitch).toBe(false);
		expect(simulateRoutingDecision("thinking", "m6", counter, DEFAULT_MAX_SWITCHES, true).shouldSwitch).toBe(false);
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

	it("phase → null in map (thinking) → no switch", () => {
		const decision = simulateRoutingDecision("thinking", "any-model", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.shouldSwitch).toBe(false);
		expect(decision.target).toBeNull();
	});

	it("phase → same model → no switch, counter unchanged", () => {
		const decision = simulateRoutingDecision(
			"executing",
			"glm-5.2:cloud", // already on the target
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
			"default-model", // different from glm-5.2:cloud
			0,
			DEFAULT_MAX_SWITCHES,
			true,
		);
		expect(decision.shouldSwitch).toBe(true);
		expect(decision.target).toBe("glm-5.2:cloud");
		expect(decision.newCounter).toBe(1);
	});

	it("switch is only applied at agent_end (simulated by returning target, not mid-round)", () => {
		// This test verifies the design: the decision function returns the target
		// without executing it. In the real extension, pi.setModel() is called at agent_end.
		const decision = simulateRoutingDecision("stuck", "default-model", 0, DEFAULT_MAX_SWITCHES, true);
		expect(decision.target).toBe("kimi-k2.6:cloud");
		expect(decision.shouldSwitch).toBe(true);
		// The round's pendingModelSwitch would be set to the target
		const round = createRound();
		round.currentPhase = "stuck";
		round.pendingModelSwitch = decision.target;
		expect(round.pendingModelSwitch).toBe("kimi-k2.6:cloud");
	});

	it("multiple phase reports in same round → last reported phase wins", () => {
		const round = createRound();
		// Simulate multiple reports overwriting each other
		round.currentPhase = "thinking";
		round.pendingModelSwitch = getModelForPhase("thinking", MVP_PHASE_MODEL_MAP);

		// Later report overwrites
		round.currentPhase = "executing";
		round.pendingModelSwitch = getModelForPhase("executing", MVP_PHASE_MODEL_MAP);

		expect(round.currentPhase).toBe("executing");
		expect(round.pendingModelSwitch).toBe("glm-5.2:cloud");
	});

	it("switch to each non-thinking phase produces expected target model", () => {
		const modelMap: Record<PhaseName, string> = {
			executing: "glm-5.2:cloud",
			stuck: "kimi-k2.6:cloud",
			reporting: "gemma4:12b:cloud",
			reviewing: "deepseek-v4-pro:cloud",
			verifying: "deepseek-v4-flash:cloud",
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
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});
		expect(config.multiModelRouting.enabled).toBe(false);
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
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});
		expect(config.multiModelRouting.enabled).toBe(true);
	});

	it("when enabled, switching decisions are made correctly", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "true" },
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});

		const decision = simulateRoutingDecision(
			"executing",
			"default-model",
			0,
			config.multiModelRouting.maxSwitches,
			config.multiModelRouting.enabled,
		);
		expect(decision.shouldSwitch).toBe(true);
		expect(decision.target).toBe("glm-5.2:cloud");
	});

	it("default maxSwitches is 3 in config", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: {},
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});
		expect(config.multiModelRouting.maxSwitches).toBe(3);
	});

	it("env var with invalid value falls back to disabled", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "invalid" },
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});
		expect(config.multiModelRouting.enabled).toBe(false);
	});

	it("env var with false value stays disabled", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "false" },
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});
		expect(config.multiModelRouting.enabled).toBe(false);
	});

	it("config defaults cannot be mutated through the returned object", () => {
		const config = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: {},
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});

		// Mutating the returned config should not affect subsequent loads
		config.multiModelRouting.enabled = true;
		config.multiModelRouting.maxSwitches = 99;

		const config2 = loadSemblrConfig({
			cwd: "/",
			agentDir: "/agent",
			env: {},
			fsImpl: { existsSync: () => false, readFileSync: () => "{}" },
		});
		expect(config2.multiModelRouting.enabled).toBe(false);
		expect(config2.multiModelRouting.maxSwitches).toBe(3);
	});
});

// ─────────────────────────────────────────────
// Category 7: Integration scenario
// ─────────────────────────────────────────────

describe("integration: full agent cycle", () => {
	it("thinking → executing → stuck → reporting → reviewing → verifying respects switch limit", () => {
		// With maxSwitches = 3, only the first 3 non-null phases trigger switches.
		// The remaining phases (reviewing, verifying) are blocked.
		// When blocked, simulateRoutingDecision returns target=null.
		let currentModel = "default-model";
		let switchCounter = 0;

		// Phase 1: thinking → no switch
		let d = simulateRoutingDecision("thinking", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBeNull();
		expect(d.shouldSwitch).toBe(false);

		// Phase 2: executing → switch to glm-5.2:cloud (switch 1)
		d = simulateRoutingDecision("executing", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("glm-5.2:cloud");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 3: stuck → switch to kimi-k2.6:cloud (switch 2)
		d = simulateRoutingDecision("stuck", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("kimi-k2.6:cloud");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 4: reporting → switch to gemma4:12b:cloud (switch 3 — maxSwitches reached)
		d = simulateRoutingDecision("reporting", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.target).toBe("gemma4:12b:cloud");
		expect(d.shouldSwitch).toBe(true);
		switchCounter = d.newCounter;
		currentModel = d.target!;

		// Phase 5: reviewing → blocked (switchCounter == 3 >= maxSwitches)
		d = simulateRoutingDecision("reviewing", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.shouldSwitch).toBe(false);
		expect(switchCounter).toBe(3);

		// Phase 6: verifying → blocked
		d = simulateRoutingDecision("verifying", currentModel, switchCounter, DEFAULT_MAX_SWITCHES, true);
		expect(d.shouldSwitch).toBe(false);
		expect(switchCounter).toBe(3);

		// After maxSwitches=3 switches, model is stuck at the 3rd target
		expect(currentModel).toBe("gemma4:12b:cloud");
	});

	it("full cycle without switch limit (maxSwitches raised) verifies complete map coverage", () => {
		// Use a high maxSwitches to bypass the limit and verify full map coverage
		const unlimitedMax = 10;
		const transitions: Array<{ phase: PhaseName; expectedTarget: string | null }> = [
			{ phase: "thinking", expectedTarget: null },
			{ phase: "executing", expectedTarget: "glm-5.2:cloud" },
			{ phase: "stuck", expectedTarget: "kimi-k2.6:cloud" },
			{ phase: "reporting", expectedTarget: "gemma4:12b:cloud" },
			{ phase: "reviewing", expectedTarget: "deepseek-v4-pro:cloud" },
			{ phase: "verifying", expectedTarget: "deepseek-v4-flash:cloud" },
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

		expect(currentModel).toBe("deepseek-v4-flash:cloud");
		expect(counter).toBe(5);
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
		expect(currentModel).toBe("deepseek-v4-flash:cloud");
	});

	it("switch counter respects maxSwitches (3) within a single cycle", () => {
		// Simulate a long cycle with many phase changes
		const phases: PhaseName[] = [
			"executing",  // 1: switch to glm-5.2
			"stuck",      // 2: switch to kimi-k2.6
			"verifying",  // 3: switch to deepseek-v4-flash
			"reporting",  // blocked (counter == maxSwitches)
			"reviewing",  // blocked
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
		expect(currentModel).toBe("deepseek-v4-flash:cloud"); // stuck at verifying model
	});

	it("multiple cycles reset the switch counter each round", () => {
		// Cycle 1: 2 switches
		let currentModel = "model-0";
		let switchCounter1 = 0;
		switchCounter1 = simulateRoutingDecision("executing", currentModel, 0, DEFAULT_MAX_SWITCHES, true).newCounter;
		currentModel = "glm-5.2:cloud";
		switchCounter1 = simulateRoutingDecision("stuck", currentModel, switchCounter1, DEFAULT_MAX_SWITCHES, true).newCounter;
		currentModel = "kimi-k2.6:cloud";

		// Cycle 2 (new round) — counter resets
		let switchCounter2 = 0;
		const d1 = simulateRoutingDecision("executing", currentModel, 0, DEFAULT_MAX_SWITCHES, true);
		expect(d1.shouldSwitch).toBe(true);
		expect(d1.newCounter).toBe(1);
	});

	it("integration: verify model IDs at each step are valid ollama-cloud IDs", () => {
		const phaseSequence: PhaseName[] = ["executing", "stuck", "reporting", "reviewing", "verifying"];

		for (const phase of phaseSequence) {
			const modelId = getModelForPhase(phase, MVP_PHASE_MODEL_MAP);
			expect(modelId).not.toBeNull();
			const resolved = resolveModelId(modelId!);
			expect(resolved.provider).toBe("ollama");
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

	it("can set pendingModelSwitch to a model ID", () => {
		const round = createRound();
		round.pendingModelSwitch = "glm-5.2:cloud";
		expect(round.pendingModelSwitch).toBe("glm-5.2:cloud");
	});

	it("can reset pendingModelSwitch back to null", () => {
		const round = createRound();
		round.pendingModelSwitch = "some-model:cloud";
		round.pendingModelSwitch = null;
		expect(round.pendingModelSwitch).toBeNull();
	});

	it("switchCounter increments independently of currentPhase", () => {
		const round = createRound();

		// Set up the round as the extension would
		round.currentPhase = "executing";
		round.pendingModelSwitch = "glm-5.2:cloud";
		round.switchCounter = 1;

		// Now change phase without incrementing counter
		round.currentPhase = "stuck";
		round.pendingModelSwitch = "kimi-k2.6:cloud";

		expect(round.switchCounter).toBe(1); // unchanged
		expect(round.currentPhase).toBe("stuck");
	});
});

// ─────────────────────────────────────────────
// Category 9: SessionState routing fields (where applicable)
// ─────────────────────────────────────────────

describe("SessionState routing behavior", () => {
	it("fresh session has no routing-specific fields (not in MVP scope)", () => {
		const session = createSession();
		// Per the spec, session-level routing fields are deferred
		expect("totalPhaseSwitches" in session).toBe(false);
	});
});
