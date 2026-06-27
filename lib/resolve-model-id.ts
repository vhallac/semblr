/**
 * resolve-model-id — resolves model ID strings to provider/model pairs.
 *
 * Supports two syntaxes:
 * 1. `{model}@{provider}` — explicit provider (e.g. "glm-5.2@ollama-cloud")
 * 2. `{model}:cloud`       — backward-compat shorthand for ollama-cloud
 * 3. `{model}`              — plain model ID, defaults to ollama-cloud
 *
 * `@` is not allowed in model or provider names — no escaping needed.
 */

export interface ResolvedModelId {
	provider: string;
	model: string;
}

/**
 * Resolves a model ID string to a provider/model pair.
 *
 * Examples:
 *   "glm-5.2@ollama-cloud"     → { provider: "ollama-cloud", model: "glm-5.2" }
 *   "anthropic/claude@openai"  → { provider: "openai", model: "anthropic/claude" }
 *   "glm-5.2:cloud"            → { provider: "ollama-cloud", model: "glm-5.2" }
 *   "anthropic/claude"         → { provider: "ollama-cloud", model: "anthropic/claude" }
 */
export function resolveModelId(modelId: string): ResolvedModelId {
	// New syntax: {model}@{provider} takes priority
	const atIndex = modelId.indexOf("@");
	if (atIndex !== -1) {
		return {
			model: modelId.slice(0, atIndex),
			provider: modelId.slice(atIndex + 1),
		};
	}

	// Backward-compat: :cloud suffix → ollama-cloud provider
	if (modelId.endsWith(":cloud")) {
		return {
			provider: "ollama-cloud",
			model: modelId.slice(0, -6), // remove ":cloud"
		};
	}

	// Pass through as-is — use ollama-cloud as default provider.
	return {
		provider: "ollama-cloud",
		model: modelId,
	};
}
