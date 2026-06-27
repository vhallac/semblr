import type * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import {
	collectPhaseModels,
	getPresetDescription,
	loadSemblrConfig,
	resolvePreset,
	validateRoutingConfig,
} from "./semblr-config.ts";

function fsFromFiles(files: Record<string, unknown>) {
	const normalized = new Map(Object.entries(files).map(([filePath, content]) => [path.normalize(filePath), content]));
	return {
		existsSync: (filePath: string) => normalized.has(path.normalize(filePath)),
		readFileSync: (filePath: string) => {
			const content = normalized.get(path.normalize(filePath));
			if (content === undefined) throw new Error(`Missing file: ${filePath}`);
			return typeof content === "string" ? content : JSON.stringify(content);
		},
		writeFileSync: (filePath: string, data: string) => {
			// Parse JSON back to match the initial map's value types (objects, not strings)
			try {
				normalized.set(path.normalize(filePath), JSON.parse(data));
			} catch {
				normalized.set(path.normalize(filePath), data);
			}
		},
		// Expose the map for test assertions
		_normalized: normalized,
	} as Pick<typeof import("node:fs"), "existsSync" | "readFileSync" | "writeFileSync"> & {
		_normalized: Map<string, unknown>;
	};
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
			summaryThresholdExtra: 0,
			routing: {
				enabled: false,
				preset: null,
				phaseModels: {},
				maxSwitchesPerCycle: 3,
				minTurnsPerPhase: 1,
				agentCycleTimeoutSec: 0,
			},
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
			},
			fsImpl,
		});

		expect(config.embeddingProvider).toBe("env-provider");
		expect(config.embeddingMaxTokens).toBe(3000);
		expect(config.embeddingApiUrl).toBe("https://embeddings.example/v1");
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

	it("defaults routing.enabled to false", () => {
		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl: fsFromFiles({}) });
		expect(config.routing.enabled).toBe(false);
	});

	it("enables routing via SEMBLR_ROUTING_ENABLED env var", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "true" },
			fsImpl: fsFromFiles({}),
		});
		expect(config.routing.enabled).toBe(true);
	});

	it("enables routing via SEMBLR_ROUTING_ENABLED=1", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "1" },
			fsImpl: fsFromFiles({}),
		});
		expect(config.routing.enabled).toBe(true);
	});

	it("disables routing via SEMBLR_ROUTING_ENABLED=false", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "false" },
			fsImpl: fsFromFiles({}),
		});
		expect(config.routing.enabled).toBe(false);
	});

	it("falls back to disabled for invalid SEMBLR_ROUTING_ENABLED values", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_ENABLED: "invalid" },
			fsImpl: fsFromFiles({}),
		});
		expect(config.routing.enabled).toBe(false);
	});

	it("defaults maxSwitchesPerCycle to 3", () => {
		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl: fsFromFiles({}) });
		expect(config.routing.maxSwitchesPerCycle).toBe(3);
	});

	it("defaults routing.phaseModels to empty when no preset", () => {
		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl: fsFromFiles({}) });
		expect(Object.keys(config.routing.phaseModels)).toHaveLength(0);
	});
});

// ═══════════════════════════════════════════
// Preset resolution tests
// ═══════════════════════════════════════════

describe("resolvePreset", () => {
	it("resolves fast-sonnet to all nulls (all current)", () => {
		const map = resolvePreset("fast-sonnet");
		expect(map).not.toBeNull();
		expect(map?.exploring).toBeNull();
		expect(map?.planning).toBeNull();
		expect(map?.executing).toBeNull();
		expect(map?.verifying).toBeNull();
		expect(map?.reporting).toBeNull();
	});

	it("resolves dual-model with actual model IDs", () => {
		const map = resolvePreset("dual-model");
		expect(map).not.toBeNull();
		expect(map?.exploring).toBeNull();
		expect(map?.planning).toBe("deepseek-v4-flash:cloud");
		expect(map?.executing).toBe("glm-5.2:cloud");
		expect(map?.verifying).toBe("minimax-m3:cloud");
		expect(map?.reporting).toBe("gemma4:31b:cloud");
	});

	it("resolves escalation-only to all nulls", () => {
		const resolved = resolvePreset("escalation-only");
		expect(resolved).not.toBeNull();
		if (!resolved) return;
		for (const val of Object.values(resolved)) {
			expect(val).toBeNull();
		}
	});

	it("returns null for unknown preset name", () => {
		expect(resolvePreset("nonexistent")).toBeNull();
		expect(resolvePreset("")).toBeNull();
	});

	it("getPresetDescription returns descriptions for known presets", () => {
		expect(getPresetDescription("fast-sonnet")).toContain("single fast model");
		expect(getPresetDescription("dual-model")).toContain("Three tiers");
		expect(getPresetDescription("escalation-only")).toContain("Minimal intervention");
		expect(getPresetDescription("unknown")).toBeNull();
	});

	it("collectPhaseModels returns all non-null model IDs", () => {
		const fastSonnet = resolvePreset("fast-sonnet");
		expect(fastSonnet).not.toBeNull();
		if (!fastSonnet) return;
		expect(collectPhaseModels(fastSonnet)).toHaveLength(0);
		const dualMap = resolvePreset("dual-model");
		expect(dualMap).not.toBeNull();
		if (!dualMap) return;
		const dualModels = collectPhaseModels(dualMap);
		expect(dualModels).toHaveLength(4);
		expect(dualModels).toContain("deepseek-v4-flash:cloud");
		expect(dualModels).toContain("glm-5.2:cloud");
	});

	it("resolvePreset is pure (same input, same output, different objects)", () => {
		const a = resolvePreset("dual-model");
		const b = resolvePreset("dual-model");
		expect(a).toEqual(b);
		expect(a).not.toBe(b);
	});
});

// ═══════════════════════════════════════════
// Routing config loading from settings.json
// ═══════════════════════════════════════════

// ═══════════════════════════════════════════
// Auto-init tests
// ═══════════════════════════════════════════

describe("autoInitSemblrSettings", () => {
	it("writes defaults when global settings.json exists but has no semblr section", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": { theme: "dark" },
		});

		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		// The file should now have a semblr section with defaults
		const settings = fsImpl._normalized.get(path.normalize("/agent/settings.json")) as Record<string, unknown>;
		expect(settings).toBeDefined();
		expect(settings.theme).toBe("dark"); // preserves existing keys
		expect(settings.semblr).toBeDefined();
		expect((settings.semblr as Record<string, unknown>).embeddingProvider).toBe("openrouter");
		expect((settings.semblr as Record<string, unknown>).embeddingModel).toBe("openai/text-embedding-3-small");

		// Config should be read from the auto-init defaults
		expect(config.embeddingProvider).toBe("openrouter");
	});

	it("is a no-op when the global settings file does not exist", () => {
		const fsImpl = fsFromFiles({});

		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl,
		});

		// File shouldn't have been created (auto-init only writes to EXISTING files)
		expect(fsImpl.existsSync("/agent/settings.json")).toBe(false);
		expect(config.embeddingProvider).toBe("openrouter");
	});

	it("is a no-op when the global settings file already has a semblr section", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": {
				semblr: { embeddingProvider: "custom-provider", embeddingMaxTokens: 12000 },
			},
		});

		const config = loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		// Should preserve existing custom settings
		expect(config.embeddingProvider).toBe("custom-provider");
		expect(config.embeddingMaxTokens).toBe(12000);
	});

	it("handles write failure gracefully with a warning", () => {
		const warnings: string[] = [];
		const fsImpl = {
			existsSync: (_p: unknown) => true,
			readFileSync: (_p: unknown) => JSON.stringify({ theme: "dark" }),
			writeFileSync: (_p: unknown, _data: unknown) => {
				throw new Error("EACCES: permission denied");
			},
		} as unknown as Pick<typeof fs, "existsSync" | "readFileSync" | "writeFileSync">;

		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl,
			warn: (msg) => warnings.push(msg),
		});

		expect(warnings.some((w) => w.includes("Failed to auto-init"))).toBe(true);
		// Config should still load with defaults
		expect(config.embeddingProvider).toBe("openrouter");
	});

	it("skips auto-init when the global settings file contains malformed JSON", () => {
		const fsImpl = {
			existsSync: (_p: unknown) => true,
			readFileSync: (_p: unknown) => "not valid json",
			writeFileSync: (_p: unknown, _data: unknown) => {
				throw new Error("should not be called");
			},
		} as unknown as Pick<typeof fs, "existsSync" | "readFileSync" | "writeFileSync">;

		// Should not throw
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl,
		});

		expect(config.embeddingProvider).toBe("openrouter");
	});

	it("writes routing defaults into the auto-init section", () => {
		const fsImpl = fsFromFiles({
			"/agent/settings.json": {},
		});

		loadSemblrConfig({ cwd: "/repo", agentDir: "/agent", env: {}, fsImpl });

		const settings = fsImpl._normalized.get(path.normalize("/agent/settings.json")) as Record<string, unknown>;
		const routing = (settings.semblr as Record<string, unknown>).routing as Record<string, unknown>;
		expect(routing).toBeDefined();
		expect(routing.enabled).toBe(false);
		expect(routing.maxSwitchesPerCycle).toBe(3);
		expect(routing.minTurnsPerPhase).toBe(1);
	});
});

describe("routing config loading", () => {
	it("loads routing.preset from settings.json", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl: fsFromFiles({
				"/repo/.pi/settings.json": { semblr: { routing: { preset: "dual-model" } } },
			}),
		});
		expect(config.routing.preset).toBe("dual-model");
		expect(config.routing.phaseModels.planning).toBe("deepseek-v4-flash:cloud");
	});

	it("loads routing.phaseModels from settings.json", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl: fsFromFiles({
				"/repo/.pi/settings.json": {
					semblr: {
						routing: {
							phaseModels: {
								exploring: null,
								planning: "custom-model:cloud",
								executing: null,
								verifying: null,
								reporting: "custom-reporter:cloud",
							},
						},
					},
				},
			}),
		});
		expect(config.routing.phaseModels.planning).toBe("custom-model:cloud");
		expect(config.routing.phaseModels.reporting).toBe("custom-reporter:cloud");
		expect(config.routing.phaseModels.exploring).toBeNull();
	});

	it("phaseModels takes precedence over preset with warning", () => {
		const warnings: string[] = [];
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl: fsFromFiles({
				"/repo/.pi/settings.json": {
					semblr: {
						routing: {
							preset: "dual-model",
							phaseModels: {
								exploring: null,
								planning: "override:cloud",
								executing: null,
								verifying: null,
								reporting: null,
							},
						},
					},
				},
			}),
			warn: (msg: string) => warnings.push(msg),
		});
		expect(config.routing.phaseModels.planning).toBe("override:cloud");
		expect(warnings.some((w) => w.includes("preset") && w.includes("ignored"))).toBe(true);
	});

	it("warns on unknown preset name", () => {
		const warnings: string[] = [];
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl: fsFromFiles({
				"/repo/.pi/settings.json": { semblr: { routing: { preset: "bogus-preset" } } },
			}),
			warn: (msg: string) => warnings.push(msg),
		});
		expect(config.routing.preset).toBeNull();
		expect(Object.keys(config.routing.phaseModels)).toHaveLength(0);
		expect(warnings.some((w) => w.includes("unknown preset"))).toBe(true);
	});

	it("loads numeric routing settings from settings.json", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: {},
			fsImpl: fsFromFiles({
				"/repo/.pi/settings.json": {
					semblr: {
						routing: { maxSwitchesPerCycle: 7, minTurnsPerPhase: 2 },
					},
				},
			}),
		});
		expect(config.routing.maxSwitchesPerCycle).toBe(7);
		expect(config.routing.minTurnsPerPhase).toBe(2);
	});

	it("overrides maxSwitchesPerCycle via SEMBLR_ROUTING_MAX_SWITCHES env var", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_MAX_SWITCHES: "10" },
			fsImpl: fsFromFiles({}),
		});
		expect(config.routing.maxSwitchesPerCycle).toBe(10);
	});

	it("overrides agentCycleTimeoutSec via SEMBLR_ROUTING_TIMEOUT env var", () => {
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_TIMEOUT: "120" },
			fsImpl: fsFromFiles({}),
		});
		expect(config.routing.agentCycleTimeoutSec).toBe(120);
	});

	it("warns and falls back for invalid numeric routing settings", () => {
		const warnings: string[] = [];
		const config = loadSemblrConfig({
			cwd: "/repo",
			agentDir: "/agent",
			env: { SEMBLR_ROUTING_MAX_SWITCHES: "-1" },
			fsImpl: fsFromFiles({}),
			warn: (msg: string) => warnings.push(msg),
		});
		expect(config.routing.maxSwitchesPerCycle).toBe(3);
		expect(warnings.some((w) => w.includes("maxSwitchesPerCycle"))).toBe(true);
	});
});

// ═══════════════════════════════════════════
// Config validation
// ═══════════════════════════════════════════

describe("validateRoutingConfig", () => {
	it("model availability check sets unavailable models to null", () => {
		const warnings: string[] = [];
		const result = validateRoutingConfig(
			{
				phaseModels: {
					exploring: null,
					planning: "nonexistent-model@bad-provider",
					executing: "glm-5.2@ollama-cloud",
					verifying: null,
					reporting: "another-bad-model@other-provider",
				},
			},
			{},
			(msg: string) => warnings.push(msg),
			(_model: string, provider: string) => provider === "ollama-cloud",
		);
		expect(result.phaseModels.planning).toBeNull();
		expect(result.phaseModels.executing).toBe("glm-5.2@ollama-cloud");
		expect(result.phaseModels.reporting).toBeNull();
		expect(warnings.length).toBe(2);
	});

	it("falls back to defaults for negative numeric values", () => {
		const warnings: string[] = [];
		const result = validateRoutingConfig({ maxSwitchesPerCycle: -5 }, {}, (msg: string) => warnings.push(msg));
		expect(result.maxSwitchesPerCycle).toBe(3);
		expect(warnings.length).toBeGreaterThan(0);
	});

	it("returns inert defaults for empty input", () => {
		const result = validateRoutingConfig({}, {}, () => {});
		expect(result.enabled).toBe(false);
		expect(result.preset).toBeNull();
		expect(Object.keys(result.phaseModels)).toHaveLength(0);
	});

	it("respects SEMBLR_ROUTING_ENABLED env for enabled toggle", () => {
		const result = validateRoutingConfig({}, { SEMBLR_ROUTING_ENABLED: "true" }, () => {});
		expect(result.enabled).toBe(true);
	});

	it("resolves preset when no phaseModels are provided", () => {
		const result = validateRoutingConfig({ preset: "dual-model" }, {}, () => {});
		expect(result.phaseModels.planning).toBe("deepseek-v4-flash:cloud");
		expect(result.phaseModels.executing).toBe("glm-5.2:cloud");
	});
});
