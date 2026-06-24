import type { PhaseModelMap, PhaseName } from "./semblr-config.ts";

/**
 * Hardcoded MVP phase → model map (Ollama-Cloud naming convention).
 *
 * From issue #86 comment #2 (2026-06-24).
 * `null` → stay on the current model (no switch).
 * `"<model>:cloud"` → switch to the named Ollama Cloud model.
 */
export const MVP_PHASE_MODEL_MAP: PhaseModelMap = {
	exploring: null,
	planning: "deepseek-v4-flash:cloud",
	executing: "glm-5.2:cloud",
	stuck: "kimi-k2.6:cloud",
	verifying: "minimax-m3:cloud",
	reporting: "gemma4:12b:cloud",
};

/**
 * Returns the target model ID for a given phase, or `null` if no switch
 * should occur (i.e. the phase maps to `null` or is absent from the map).
 */
export function getModelForPhase(phase: PhaseName, map: PhaseModelMap): string | null {
	return map[phase] ?? null;
}
