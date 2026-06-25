/**
 * resolve-model-id — resolves Ollama-Cloud model IDs to provider/model pairs.
 *
 * For MVP: strips ":cloud" suffix, uses "ollama-cloud" as the provider.
 * Non-cloud IDs pass through with "ollama-cloud" as the default provider.
 */

export interface ResolvedModelId {
	provider: string;
	model: string;
}

/**
 * Resolves a model ID string to a provider/model pair.
 *
 * Examples:
 *   "glm-5.2:cloud"         → { provider: "ollama-cloud", model: "glm-5.2" }
 *   "kimi-k2.6:cloud"       → { provider: "ollama-cloud", model: "kimi-k2.6" }
 *   "anthropic/claude"      → { provider: "ollama-cloud", model: "anthropic/claude" }
 *
 * In the MVP, all phase→model map entries use Ollama-Cloud naming
 * (`<model>:cloud`), so stripping the suffix and defaulting to
 * "ollama-cloud" is sufficient.
 */
export function resolveModelId(modelId: string): ResolvedModelId {
	if (modelId.includes(":cloud")) {
		return {
			provider: "ollama-cloud",
			model: modelId.replace(/:cloud$/, ""),
		};
	}
	// Pass through as-is — use ollama-cloud as default provider.
	return {
		provider: "ollama-cloud",
		model: modelId,
	};
}
