import * as path from "node:path";
import {
	createEmbeddingModelRegistry,
	type EmbeddingClientConfig,
	type EmbeddingModelRegistry,
	getApiKey,
} from "./embedding-client.ts";
import { loadSemblrConfig, type SemblrConfig, type SemblrConfigDeps } from "./semblr-config.ts";

export interface ScriptConfigOptions {
	config?: SemblrConfig;
	configDeps?: SemblrConfigDeps;
}

export interface ScriptEmbeddingOptions extends ScriptConfigOptions {
	apiKey?: string;
	modelRegistry?: EmbeddingModelRegistry;
}

export function resolveScriptConfig(options: ScriptConfigOptions = {}): SemblrConfig {
	return options.config ?? loadSemblrConfig(options.configDeps);
}

export function resolveScriptIndexPath(config: SemblrConfig, roundsDir: string, indexPath?: string): string {
	return indexPath ?? (roundsDir === config.roundsDir ? config.indexPath : path.resolve(roundsDir, "index.csv"));
}

export function resolveScriptModelRegistry(
	config: SemblrConfig,
	options: Pick<ScriptEmbeddingOptions, "apiKey" | "modelRegistry"> = {},
): EmbeddingModelRegistry {
	return (
		options.modelRegistry ??
		createEmbeddingModelRegistry(config, { agentDir: config.agentDir, apiKey: options.apiKey })
	);
}

export async function resolveScriptApiKey(
	config: SemblrConfig,
	options: ScriptEmbeddingOptions = {},
): Promise<string | null> {
	if (options.apiKey !== undefined) return options.apiKey || null;
	return getApiKey(undefined, { config, modelRegistry: resolveScriptModelRegistry(config, options) });
}

export function scriptEmbeddingConfig(config: SemblrConfig): EmbeddingClientConfig {
	return {
		embeddingProvider: config.embeddingProvider,
		embeddingModel: config.embeddingModel,
		embeddingApiUrl: config.embeddingApiUrl,
		embedTimeoutMs: config.embedTimeoutMs,
		embedMaxRetries: config.embedMaxRetries,
		embedBackoffMs: config.embedBackoffMs,
	};
}
