import type { spawn } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import {
	createEmbeddingModelRegistry,
	EMBEDDING_MODEL,
	type EmbeddingClientConfig,
	type EmbeddingModelRegistry,
	embedText,
	getApiKey,
	resolveEmbeddingApiUrl,
} from "./embedding-client.ts";

function okEmbeddingFetch(requests: Array<{ url: string; body: unknown }>, vector = [1, 2]): typeof fetch {
	return vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
		requests.push({ url: String(url), body: JSON.parse(String(init?.body)) });
		return new Response(JSON.stringify({ data: [{ embedding: vector }] }), { status: 200 });
	}) as typeof fetch;
}

function config(overrides: Partial<EmbeddingClientConfig> = {}): EmbeddingClientConfig {
	return {
		embeddingProvider: "openrouter",
		embeddingModel: EMBEDDING_MODEL,
		embeddingApiUrl: undefined,
		embedTimeoutMs: 15_000,
		embedMaxRetries: 3,
		embedBackoffMs: 1000,
		...overrides,
	};
}

describe("embedding-client", () => {
	it("preserves the default OpenRouter endpoint and embedding model", async () => {
		const requests: Array<{ url: string; body: unknown }> = [];

		const vector = await embedText("hello", "api-key", { fetchImpl: okEmbeddingFetch(requests, [3, 4]) });

		expect(vector).toEqual([3, 4]);
		expect(requests).toEqual([
			{
				url: "https://openrouter.ai/api/v1/embeddings",
				body: { model: "openai/text-embedding-3-small", input: "hello" },
			},
		]);
	});

	it("uses configured embeddingApiUrl and embeddingModel in requests", async () => {
		const requests: Array<{ url: string; body: unknown }> = [];

		await embedText("hello", "api-key", {
			fetchImpl: okEmbeddingFetch(requests),
			config: config({
				embeddingProvider: "custom-provider",
				embeddingModel: "custom-embedding-model",
				embeddingApiUrl: "https://embeddings.example/custom",
			}),
		});

		expect(requests).toEqual([
			{
				url: "https://embeddings.example/custom",
				body: { model: "custom-embedding-model", input: "hello" },
			},
		]);
	});

	it("lets embeddingApiUrl bypass provider/model lookup", () => {
		const registry = { find: vi.fn() };

		const url = resolveEmbeddingApiUrl(
			config({
				embeddingProvider: "missing-provider",
				embeddingModel: "missing-model",
				embeddingApiUrl: "https://embeddings.example/full-override",
			}),
			registry,
		);

		expect(url).toBe("https://embeddings.example/full-override");
		expect(registry.find).not.toHaveBeenCalled();
	});

	it.each([
		["https://provider.example/root", "https://provider.example/root/embeddings"],
		["https://provider.example/root/", "https://provider.example/root/embeddings"],
		["https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/embeddings"],
	])("derives the embeddings endpoint from provider model baseUrl %s", (baseUrl, expectedUrl) => {
		const registry = {
			find: vi.fn(() => ({ baseUrl })),
		};

		const url = resolveEmbeddingApiUrl(
			config({ embeddingProvider: "provider", embeddingModel: "embedding-model" }),
			registry,
		);

		expect(registry.find).toHaveBeenCalledWith("provider", "embedding-model");
		expect(url).toBe(expectedUrl);
	});

	it("explains how to fix a missing non-default provider/model lookup", () => {
		expect(() =>
			resolveEmbeddingApiUrl(config({ embeddingProvider: "provider", embeddingModel: "embedding-model" }), {
				find: () => undefined,
			}),
		).toThrow(
			"Semblr embedding model provider/embedding-model was not found. Add the embedding model to pi models.json or set semblr.embeddingApiUrl / SEMBLR_EMBEDDING_API_URL.",
		);
	});

	it("gets API keys from the configured provider instead of hardcoded OpenRouter", async () => {
		const registry = {
			find: vi.fn(),
			getApiKeyForProvider: vi.fn(async (provider: string) => `${provider}-key`),
		};

		const apiKey = await getApiKey(
			{ modelRegistry: registry as EmbeddingModelRegistry },
			{ config: config({ embeddingProvider: "custom-provider" }) },
		);

		expect(apiKey).toBe("custom-provider-key");
		expect(registry.getApiKeyForProvider).toHaveBeenCalledWith("custom-provider");
	});

	it("keeps OPENROUTER_API_KEY as the default provider compatibility fallback", async () => {
		const registry = {
			find: vi.fn(),
			getApiKeyForProvider: vi.fn(async () => "registry-key"),
		};

		const apiKey = await getApiKey(
			{ modelRegistry: registry as EmbeddingModelRegistry },
			{ env: { OPENROUTER_API_KEY: "env-key" } },
		);

		expect(apiKey).toBe("env-key");
		expect(registry.getApiKeyForProvider).not.toHaveBeenCalled();
	});

	it("does not use the legacy OpenRouter pass fallback for custom providers", async () => {
		const spawnImpl = vi.fn() as unknown as typeof spawn;

		const apiKey = await getApiKey(undefined, {
			config: config({ embeddingProvider: "custom-provider" }),
			spawnImpl,
		});

		expect(apiKey).toBeNull();
		expect(spawnImpl).not.toHaveBeenCalled();
	});

	it("uses retry and backoff values from config", async () => {
		const sleepCalls: number[] = [];
		const fetchImpl = vi.fn(async () => new Response("busy", { status: 500 })) as typeof fetch;

		await expect(
			embedText("hello", "api-key", {
				fetchImpl,
				sleep: async (ms) => {
					sleepCalls.push(ms);
				},
				config: config({
					embeddingApiUrl: "https://embeddings.example/custom",
					embedMaxRetries: 2,
					embedBackoffMs: 25,
				}),
			}),
		).rejects.toThrow("Embedding API error 500: busy");

		expect(fetchImpl).toHaveBeenCalledTimes(2);
		expect(sleepCalls).toEqual([25]);
	});

	it("uses timeout values from config", async () => {
		vi.useFakeTimers();
		try {
			const fetchImpl = vi.fn(
				(_url: string | URL | Request, init?: RequestInit) =>
					new Promise<Response>((_resolve, reject) => {
						init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
					}),
			) as typeof fetch;

			const promise = embedText("hello", "api-key", {
				fetchImpl,
				config: config({
					embeddingApiUrl: "https://embeddings.example/custom",
					embedMaxRetries: 1,
					embedTimeoutMs: 50,
				}),
			});
			const assertion = expect(promise).rejects.toThrow("Embedding API timeout after 50ms");

			await vi.advanceTimersByTimeAsync(50);
			await assertion;
		} finally {
			vi.useRealTimers();
		}
	});

	it("creates a pi model registry rooted at the configured agent dir and applies API key overrides", async () => {
		const registry = createEmbeddingModelRegistry(config({ embeddingProvider: "custom-provider" }), {
			agentDir: "/tmp/agent-for-semblr-test",
			apiKey: "runtime-key",
		});

		expect(await registry.getApiKeyForProvider("custom-provider")).toBe("runtime-key");
	});
});
