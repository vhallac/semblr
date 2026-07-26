import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { loadSemblrConfig } from "./semblr-config.ts";

function fsFromFiles(files: Record<string, unknown>) {
	const normalized = new Map(Object.entries(files).map(([filePath, content]) => [path.normalize(filePath), content]));
	return {
		existsSync: (filePath: string) => normalized.has(path.normalize(filePath)),
		readFileSync: (filePath: string) => {
			const content = normalized.get(path.normalize(filePath));
			if (content === undefined) throw new Error(`Missing file: ${filePath}`);
			return typeof content === "string" ? content : JSON.stringify(content);
		},
	} as Pick<typeof import("node:fs"), "existsSync" | "readFileSync">;
}

describe("loadSemblrConfig", () => {
	it("returns defaults that preserve current behavior", () => {
		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl: fsFromFiles({}) });

		expect(config).toEqual({
			agentDir: "/agent",
			embeddingProvider: "openrouter",
			embeddingModel: "openai/text-embedding-3-small",
			embeddingMaxTokens: 8000,
			embeddingApiUrl: undefined,
			roundsDir: "/agent/semblr/rounds",
			indexPath: "/agent/semblr/rounds/index.csv",
			groupThreshold: 0.77,
			minSimilarity: 0.3,
			embedTimeoutMs: 15_000,
			embedMaxRetries: 3,
			embedBackoffMs: 1000,
			hybridSemanticWeight: 0.7,
			summaryThresholdExtra: 0,
		});
	});

	it("lets env values override settings and defaults", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": { semblr: { embeddingProvider: "global-provider", embeddingMaxTokens: 1000 } },
			"/repo/.pi/settings.json": { semblr: { embeddingProvider: "project-provider", embeddingMaxTokens: 2000 } },
		});

		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {
				SEMBLR_EMBEDDING_PROVIDER: "env-provider",
				SEMBLR_EMBEDDING_MAX_TOKENS: "3000",
				SEMBLR_EMBEDDING_API_URL: "https://embeddings.example/v1",
				SEMBLR_HYBRID_SEMANTIC_WEIGHT: "0.4",
			},
			fsImpl,
		});

		expect(config.embeddingProvider).toBe("env-provider");
		expect(config.embeddingMaxTokens).toBe(3000);
		expect(config.embeddingApiUrl).toBe("https://embeddings.example/v1");
		expect(config.hybridSemanticWeight).toBe(0.4);
	});

	it("lets project settings override global settings per key", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": {
				semblr: { embeddingProvider: "global-provider", embeddingModel: "global-model", minSimilarity: 0.4 },
			},
			"/repo/.pi/settings.json": { semblr: { embeddingModel: "project-model" } },
		});

		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		expect(config.embeddingProvider).toBe("global-provider");
		expect(config.embeddingModel).toBe("project-model");
		expect(config.minSimilarity).toBe(0.4);
	});

	it("merges global and project semblr sections without reading unrelated settings", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": { theme: "ignored", semblr: { groupThreshold: 0.82, embedBackoffMs: 2500 } },
			"/repo/.pi/settings.json": { providers: { ignored: true }, semblr: { embedBackoffMs: 500 } },
		});

		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		expect(config.groupThreshold).toBe(0.82);
		expect(config.embedBackoffMs).toBe(500);
		expect(config.embeddingProvider).toBe("openrouter");
	});

	it("resolves project relative roundsDir under the project cwd", () => {
		const fsImpl = fsFromFiles({
			"/repo/.pi/settings.json": { semblr: { roundsDir: "local-rounds" } },
		});

		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		expect(config.roundsDir).toBe("/repo/local-rounds");
		expect(config.indexPath).toBe("/repo/local-rounds/index.csv");
	});

	it("resolves global relative roundsDir under the pi agent dir", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": { semblr: { roundsDir: "global-rounds" } },
		});

		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		expect(config.roundsDir).toBe("/agent/global-rounds");
	});

	it("uses absolute roundsDir values unchanged", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": { semblr: { roundsDir: "/absolute/rounds" } },
		});

		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		expect(config.roundsDir).toBe("/absolute/rounds");
		expect(config.indexPath).toBe("/absolute/rounds/index.csv");
	});

	it("resolves env relative roundsDir under the pi agent dir", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUNDS_DIR: "env-rounds" },
			fsImpl: fsFromFiles({}),
		});

		expect(config.roundsDir).toBe("/agent/env-rounds");
	});

	it("warns and falls back to defaults for invalid numeric values", () => {
		const warnings: string[] = [];
		const fsImpl = fsFromFiles({
			"/repo/.pi/settings.json": { semblr: { embeddingMaxTokens: "not-a-number", minSimilarity: false } },
		});

		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl,
			warn: (message) => warnings.push(message),
		});

		expect(config.embeddingMaxTokens).toBe(8000);
		expect(config.minSimilarity).toBe(0.3);
		expect(config.summaryThresholdExtra).toBe(0);
		expect(warnings).toEqual([
			"Invalid numeric Semblr setting embeddingMaxTokens; using default 8000",
			"Invalid numeric Semblr setting minSimilarity; using default 0.3",
		]);
	});

	it("uses PI_CODING_AGENT_DIR for the default global agent dir", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			env: { PI_CODING_AGENT_DIR: "/configured-agent" },
			fsImpl: fsFromFiles({}),
		});

		expect(config.roundsDir).toBe("/configured-agent/semblr/rounds");
	});
});
