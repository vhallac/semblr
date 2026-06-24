/**
 * resolve-model-id — resolves Ollama-Cloud model IDs to provider/model pairs.
 *
 * For MVP: strips ":cloud" suffix, uses "ollama" as the provider.
 * Non-cloud IDs pass through with "ollama" as the default provider.
 */

export interface ResolvedModelId {
	provider: string;
	model: string;
}

/**
 * Resolves a model ID string to a provider/model pair.
 *
 * Examples:
 *   "glm-5.2:cloud"         → { provider: "ollama", model: "glm-5.2" }
 *   "kimi-k2.6:cloud"       → { provider: "ollama", model: "kimi-k2.6" }
 *   "anthropic/claude"      → { provider: "ollama", model: "anthropic/claude" }
 *
 * In the MVP, all phase→model map entries use Ollama-Cloud naming
 * (`<model>:cloud`), so stripping the suffix and defaulting to "ollama"
 * is sufficient.
 */
export function resolveModelId(modelId: string): ResolvedModelId {
	if (modelId.includes(":cloud")) {
		return {
			provider: "ollama",
			model: modelId.replace(/:cloud$/, ""),
		};
	}
	// Pass through as-is — use ollama as default provider.
	return {
		provider: "ollama",
		model: modelId,
	};
}
