import type { PhaseModelMap, PhaseName } from "./semblr-config.ts";

/**
 * Hardcoded MVP phase → model map (Ollama-Cloud naming convention).
 *
 * `null` → stay on the current model (no switch).
 * `"<model>:cloud"` → switch to the named Ollama Cloud model.
 */
export const MVP_PHASE_MODEL_MAP: PhaseModelMap = {
	thinking: null, // stay on default model for thinking
	executing: "glm-5.2:cloud",
	stuck: "kimi-k2.6:cloud",
	reporting: "gemma4:12b:cloud",
	reviewing: "deepseek-v4-pro:cloud",
	verifying: "deepseek-v4-flash:cloud",
};

/**
 * Returns the target model ID for a given phase, or `null` if no switch
 * should occur (i.e. the phase maps to `null` or is absent from the map).
 */
export function getModelForPhase(phase: PhaseName, map: PhaseModelMap): string | null {
	return map[phase] ?? null;
}
