import { spawn } from "node:child_process";

const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";
const EMBED_TIMEOUT_MS = 15_000;
const EMBED_MAX_RETRIES = 3;
const EMBED_BACKOFF_MS = 1000;

export interface ApiKeyContext {
	modelRegistry?: { getApiKeyForProvider(provider: string): Promise<string | undefined> };
}

export interface ApiKeyLookupDeps {
	env?: { OPENROUTER_API_KEY?: string };
	spawnImpl?: typeof spawn;
}

export async function getApiKey(ctx?: ApiKeyContext, deps: ApiKeyLookupDeps = {}): Promise<string | null> {
	const env = deps.env ?? process.env;
	const envKey = env.OPENROUTER_API_KEY;
	if (envKey) return envKey;

	try {
		const piKey = await ctx?.modelRegistry?.getApiKeyForProvider("openrouter");
		if (piKey) return piKey;
	} catch {
		// Pi auth lookup failed, fall through.
	}

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
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function embedText(text: string, apiKey: string, deps: EmbedTextDeps = {}): Promise<number[]> {
	const fetchImpl: EmbeddingFetch = deps.fetchImpl ?? ((url, init) => fetch(url, init));
	const sleepImpl = deps.sleep ?? sleep;
	const timeoutMs = deps.timeoutMs ?? EMBED_TIMEOUT_MS;
	const maxRetries = deps.maxRetries ?? EMBED_MAX_RETRIES;
	const backoffMs = deps.backoffMs ?? EMBED_BACKOFF_MS;
	let lastError: Error | undefined;

	for (let attempt = 0; attempt < maxRetries; attempt++) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), timeoutMs);

		try {
			const response = await fetchImpl(OPENROUTER_URL, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${apiKey}`,
				},
				body: JSON.stringify({
					model: EMBEDDING_MODEL,
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
