import { spawn } from "node:child_process";
import * as os from "node:os";
import * as path from "node:path";
import { AuthStorage, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { SEMBLR_CONFIG_DEFAULTS, type SemblrConfig } from "./semblr-config.ts";

export const EMBEDDING_MODEL = SEMBLR_CONFIG_DEFAULTS.embeddingModel;

export interface EmbeddingModelLike {
	baseUrl: string;
}

export interface EmbeddingModelRegistry {
	find(provider: string, modelId: string): EmbeddingModelLike | undefined;
	getApiKeyForProvider(provider: string): Promise<string | undefined>;
	registerProvider?(provider: string, config: { apiKey?: string }): void;
}

export type EmbeddingClientConfig = Pick<
	SemblrConfig,
	"embeddingProvider" | "embeddingModel" | "embeddingApiUrl" | "embedTimeoutMs" | "embedMaxRetries" | "embedBackoffMs"
>;

type EmbeddingClientConfigInput = Partial<EmbeddingClientConfig>;

export interface ApiKeyContext {
	modelRegistry?: EmbeddingModelRegistry;
}

export interface ApiKeyLookupDeps {
	env?: { OPENROUTER_API_KEY?: string };
	spawnImpl?: typeof spawn;
	config?: EmbeddingClientConfigInput;
	modelRegistry?: EmbeddingModelRegistry;
}

function normalizeEmbeddingConfig(config: EmbeddingClientConfigInput = {}): EmbeddingClientConfig {
	return {
		embeddingProvider: config.embeddingProvider ?? SEMBLR_CONFIG_DEFAULTS.embeddingProvider,
		embeddingModel: config.embeddingModel ?? SEMBLR_CONFIG_DEFAULTS.embeddingModel,
		embeddingApiUrl: config.embeddingApiUrl,
		embedTimeoutMs: config.embedTimeoutMs ?? SEMBLR_CONFIG_DEFAULTS.embedTimeoutMs,
		embedMaxRetries: config.embedMaxRetries ?? SEMBLR_CONFIG_DEFAULTS.embedMaxRetries,
		embedBackoffMs: config.embedBackoffMs ?? SEMBLR_CONFIG_DEFAULTS.embedBackoffMs,
	};
}

function isDefaultEmbeddingSelection(config: EmbeddingClientConfig): boolean {
	return (
		config.embeddingProvider === SEMBLR_CONFIG_DEFAULTS.embeddingProvider &&
		config.embeddingModel === SEMBLR_CONFIG_DEFAULTS.embeddingModel &&
		config.embeddingApiUrl === undefined
	);
}

function appendEmbeddingsPath(baseUrl: string): string {
	return `${baseUrl.replace(/\/+$/, "")}/embeddings`;
}

export function resolveEmbeddingApiUrl(
	configInput: EmbeddingClientConfigInput = {},
	modelRegistry?: Pick<EmbeddingModelRegistry, "find">,
): string {
	const config = normalizeEmbeddingConfig(configInput);
	if (config.embeddingApiUrl) return config.embeddingApiUrl;

	const model = modelRegistry?.find(config.embeddingProvider, config.embeddingModel);
	if (model) return appendEmbeddingsPath(model.baseUrl);

	if (isDefaultEmbeddingSelection(config)) return SEMBLR_CONFIG_DEFAULTS.defaultEmbeddingApiUrl;

	throw new Error(
		`Semblr embedding model ${config.embeddingProvider}/${config.embeddingModel} was not found. ` +
			"Add the embedding model to pi models.json or set semblr.embeddingApiUrl / SEMBLR_EMBEDDING_API_URL.",
	);
}

function defaultAgentDir(env: NodeJS.ProcessEnv = process.env): string {
	return env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
}

export interface CreateEmbeddingModelRegistryOptions {
	agentDir?: string;
	apiKey?: string;
	env?: NodeJS.ProcessEnv;
}

export function createEmbeddingModelRegistry(
	configInput: EmbeddingClientConfigInput = {},
	options: CreateEmbeddingModelRegistryOptions = {},
): EmbeddingModelRegistry {
	const config = normalizeEmbeddingConfig(configInput);
	const agentDir = options.agentDir ?? defaultAgentDir(options.env);
	const authStorage = AuthStorage.create(path.join(agentDir, "auth.json"));
	const modelRegistry = ModelRegistry.create(authStorage, path.join(agentDir, "models.json"));
	if (options.apiKey) {
		modelRegistry.registerProvider(config.embeddingProvider, { apiKey: options.apiKey });
	}
	return modelRegistry;
}

export async function getApiKey(ctx?: ApiKeyContext, deps: ApiKeyLookupDeps = {}): Promise<string | null> {
	const config = normalizeEmbeddingConfig(deps.config);
	const env = deps.env ?? process.env;
	const modelRegistry = deps.modelRegistry ?? ctx?.modelRegistry;

	if (config.embeddingProvider === SEMBLR_CONFIG_DEFAULTS.embeddingProvider) {
		const envKey = env.OPENROUTER_API_KEY;
		if (envKey) return envKey;
	}

	try {
		const piKey = await modelRegistry?.getApiKeyForProvider(config.embeddingProvider);
		if (piKey) return piKey;
	} catch {
		// Pi auth lookup failed, fall through to legacy OpenRouter fallback if applicable.
	}

	if (config.embeddingProvider !== SEMBLR_CONFIG_DEFAULTS.embeddingProvider) return null;

	try {
		return await new Promise<string | null>((resolve) => {
			const child = (deps.spawnImpl ?? spawn)("pass", ["show", "ai/openrouter"], {
				timeout: 1000,
				stdio: ["pipe", "pipe", "pipe"],
			});
			let stdout = "";
			child.stdout?.on("data", (chunk: Buffer) => {
				stdout += chunk.toString();
			});
			child.on("close", (code) => {
				if (code === 0) {
					const trimmed = stdout.trim();
					resolve(trimmed || null);
				} else {
					resolve(null);
				}
			});
			child.on("error", () => resolve(null));
		});
	} catch {
		// pass not available, fall through.
	}

	return null;
}

interface EmbeddingResponseLike {
	ok: boolean;
	status: number;
	text(): Promise<string>;
	json(): Promise<unknown>;
}

type EmbeddingFetch = (url: string, init: RequestInit) => Promise<EmbeddingResponseLike>;

export interface EmbedTextDeps {
	fetchImpl?: EmbeddingFetch;
	sleep?: (ms: number) => Promise<void>;
	timeoutMs?: number;
	maxRetries?: number;
	backoffMs?: number;
	config?: EmbeddingClientConfigInput;
	modelRegistry?: Pick<EmbeddingModelRegistry, "find">;
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function embedText(text: string, apiKey: string, deps: EmbedTextDeps = {}): Promise<number[]> {
	const config = normalizeEmbeddingConfig(deps.config);
	const fetchImpl: EmbeddingFetch = deps.fetchImpl ?? ((url, init) => fetch(url, init));
	const sleepImpl = deps.sleep ?? sleep;
	const timeoutMs = deps.timeoutMs ?? config.embedTimeoutMs;
	const maxRetries = deps.maxRetries ?? config.embedMaxRetries;
	const backoffMs = deps.backoffMs ?? config.embedBackoffMs;
	const embeddingApiUrl = resolveEmbeddingApiUrl(config, deps.modelRegistry);
	let lastError: Error | undefined;

	for (let attempt = 0; attempt < maxRetries; attempt++) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), timeoutMs);

		try {
			const response = await fetchImpl(embeddingApiUrl, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${apiKey}`,
				},
				body: JSON.stringify({
					model: config.embeddingModel,
					input: text,
				}),
				signal: controller.signal,
			});

			if (!response.ok) {
				const errText = await response.text();
				const err = new Error(`Embedding API error ${response.status}: ${errText}`);
				if (response.status >= 400 && response.status < 500 && response.status !== 429) {
					throw err;
				}
				lastError = err;
			} else {
				const data = (await response.json()) as {
					data: Array<{ embedding: number[] }>;
				};
				return data.data[0].embedding;
			}
		} catch (e) {
			if (e instanceof DOMException && e.name === "AbortError") {
				lastError = new Error(`Embedding API timeout after ${timeoutMs}ms`);
			} else if (e instanceof Error && e.message.startsWith("Embedding API error 4") && !e.message.includes("429")) {
				throw e;
			} else {
				lastError = e instanceof Error ? e : new Error(String(e));
			}
		} finally {
			clearTimeout(timer);
		}

		if (attempt < maxRetries - 1) {
			await sleepImpl(backoffMs * 2 ** attempt);
		}
	}

	throw lastError ?? new Error("Embedding API failed after retries");
}
