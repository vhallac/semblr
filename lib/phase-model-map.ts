import type { PhaseModelMap, PhaseName } from "./semblr-config.ts";

/**
 * Hardcoded MVP phase → model map.
 *
 * From issue #86 comment #2 (2026-06-24).
 * `null` → stay on the current model (no switch).
 * `"<model>@<provider>"` → switch to the named model on the specified provider.
 */
export const MVP_PHASE_MODEL_MAP: PhaseModelMap = {
	exploring: null,
	planning: "deepseek/deepseek-v4-flash@openrouter",
	executing: "z-ai/glm-5.2@openrouter",
	verifying: "minimax/minimax-m3@openrouter",
	reporting: "google/gemma-4-31b-it@openrouter",
};

/**
 * Returns the target model ID for a given phase, or `null` if no switch
 * should occur (i.e. the phase maps to `null` or is absent from the map).
 */
export function getModelForPhase(phase: PhaseName, map: PhaseModelMap): string | null {
	return map[phase] ?? null;
}
