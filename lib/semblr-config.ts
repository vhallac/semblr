import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ─────────────────────────────────────────────
// Multi-Model Routing Types
// ─────────────────────────────────────────────

/** Phase names the LLM can self-report via semblr_report_phase tool. */
export type PhaseName = "exploring" | "planning" | "executing" | "verifying" | "reporting";

/** Map from phase to model ID. `null` means stay on the current model. */
export type PhaseModelMap = Record<PhaseName, string | null>;

/** Named preset identifiers. */
export type PresetName = "fast-sonnet" | "dual-model" | "escalation-only";

/** All known preset names. */
export const PRESET_NAMES: readonly PresetName[] = ["fast-sonnet", "dual-model", "escalation-only"] as const;

/** Configuration for multi-model routing. */
export interface RoutingConfig {
	/** Opt-in toggle. Default false. */
	enabled: boolean;
	/** Named preset. Null = use phaseModels directly. */
	preset: PresetName | null;
	/** Per-phase model mapping. Overrides preset if both are set. */
	phaseModels: PhaseModelMap;
	/** Max model switches per agent cycle. */
	maxSwitchesPerCycle: number;
	/** Minimum turns a model must serve before phase change is allowed. */
	minTurnsPerPhase: number;
	/** Timeout in seconds for the agent cycle. 0 = no timeout. */
	agentCycleTimeoutSec: number;
}

/** Inert routing config (disabled, no switches). Used when config validation fails. */
const DISABLED_ROUTING_CONFIG: RoutingConfig = {
	enabled: false,
	preset: null,
	phaseModels: {} as PhaseModelMap,
	maxSwitchesPerCycle: 0,
	minTurnsPerPhase: 0,
	agentCycleTimeoutSec: 0,
};

/** Preset phase→model mappings. "current" values are resolved to null at load time. */
const PRESET_REGISTRY: Record<PresetName, Record<PhaseName, string | "current">> = {
	"fast-sonnet": {
		exploring: "current",
		planning: "current",
		executing: "current",
		verifying: "current",
		reporting: "current",
	},
	"dual-model": {
		exploring: "current",
		planning: "deepseek-v4-flash:cloud",
		executing: "glm-5.2:cloud",
		verifying: "minimax-m3:cloud",
		reporting: "gemma4:31b:cloud",
	},
	"escalation-only": {
		exploring: "current",
		planning: "current",
		executing: "current",
		verifying: "current",
		reporting: "current",
	},
};

/** Preset descriptions (for documentation and command output). */
const PRESET_DESCRIPTIONS: Record<PresetName, string> = {
	"fast-sonnet": "Route everything through a single fast model. Zero switching cost, zero overhead.",
	"dual-model":
		"Exploration and reporting on a cheap/fast model, execution on a mid-tier model, escalation when stuck. Three tiers, cost-conscious.",
	"escalation-only":
		"Minimal intervention. Only switches when the LLM reports it's stuck — otherwise stays on the user's chosen model.",
};

/** Default routing config values. */
const ROUTING_DEFAULTS = {
	maxSwitchesPerCycle: 3,
	minTurnsPerPhase: 1,
	agentCycleTimeoutSec: 0,
} as const;

export interface SemblrConfig {
	agentDir: string;
	embeddingProvider: string;
	embeddingModel: string;
	embeddingMaxTokens: number;
	embeddingApiUrl?: string;
	roundsDir: string;
	indexPath: string;
	groupThreshold: number;
	minSimilarity: number;
	embedTimeoutMs: number;
	embedMaxRetries: number;
	embedBackoffMs: number;
	/** 0 disables the automatic context-size warning; set a positive token count to enable it. */
	summaryThresholdExtra: number;
	/** Multi-model routing configuration. */
	routing: RoutingConfig;
}

export interface SemblrConfigEnv {
	PI_CODING_AGENT_DIR?: string;
	SEMBLR_EMBEDDING_PROVIDER?: string;
	SEMBLR_EMBEDDING_MODEL?: string;
	SEMBLR_EMBEDDING_MAX_TOKENS?: string;
	SEMBLR_EMBEDDING_API_URL?: string;
	SEMBLR_ROUNDS_DIR?: string;
	SEMBLR_GROUP_THRESHOLD?: string;
	SEMBLR_MIN_SIMILARITY?: string;
	SEMBLR_EMBED_TIMEOUT?: string;
	SEMBLR_EMBED_RETRIES?: string;
	SEMBLR_EMBED_BACKOFF?: string;
	SEMBLR_SUMMARY_THRESHOLD_EXTRA?: string;
	SEMBLR_ROUTING_ENABLED?: string;
	SEMBLR_ROUTING_MAX_SWITCHES?: string;
	SEMBLR_ROUTING_TIMEOUT?: string;
}

export interface SemblrConfigDeps {
	cwd?: string;
	agentDir?: string;
	env?: SemblrConfigEnv;
	fsImpl?: Pick<typeof fs, "existsSync" | "readFileSync" | "writeFileSync">;
	warn?: (message: string) => void;
}

type ConfigKey = keyof Omit<SemblrConfig, "agentDir" | "indexPath" | "routing">;
type SettingValue = string | number | boolean | null | SettingRecord | SettingValue[];
type SettingRecord = { [key: string]: SettingValue | undefined };

const DEFAULTS = {
	embeddingProvider: "openrouter",
	embeddingModel: "openai/text-embedding-3-small",
	defaultEmbeddingApiUrl: "https://openrouter.ai/api/v1/embeddings",
	embeddingMaxTokens: 8000,
	groupThreshold: 0.77,
	minSimilarity: 0.3,
	embedTimeoutMs: 15_000,
	embedMaxRetries: 3,
	embedBackoffMs: 1000,
	summaryThresholdExtra: 0,
};

const ENV_KEYS = {
	embeddingProvider: "SEMBLR_EMBEDDING_PROVIDER",
	embeddingModel: "SEMBLR_EMBEDDING_MODEL",
	embeddingMaxTokens: "SEMBLR_EMBEDDING_MAX_TOKENS",
	embeddingApiUrl: "SEMBLR_EMBEDDING_API_URL",
	roundsDir: "SEMBLR_ROUNDS_DIR",
	groupThreshold: "SEMBLR_GROUP_THRESHOLD",
	minSimilarity: "SEMBLR_MIN_SIMILARITY",
	embedTimeoutMs: "SEMBLR_EMBED_TIMEOUT",
	embedMaxRetries: "SEMBLR_EMBED_RETRIES",
	embedBackoffMs: "SEMBLR_EMBED_BACKOFF",
	summaryThresholdExtra: "SEMBLR_SUMMARY_THRESHOLD_EXTRA",
} satisfies Record<ConfigKey, keyof SemblrConfigEnv>;

function defaultAgentDir(env: SemblrConfigEnv): string {
	return env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
}

function isRecord(value: unknown): value is SettingRecord {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function mergeRecords(base: SettingRecord, override: SettingRecord): SettingRecord {
	const merged: SettingRecord = { ...base };
	for (const [key, value] of Object.entries(override)) {
		const existing = merged[key];
		if (isRecord(existing) && isRecord(value)) {
			merged[key] = mergeRecords(existing, value);
		} else {
			merged[key] = value;
		}
	}
	return merged;
}

function loadSemblrSection(
	settingsPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync" | "writeFileSync">,
	warn: (message: string) => void,
): SettingRecord {
	if (!fsImpl.existsSync(settingsPath)) return {};

	try {
		const parsed = JSON.parse(fsImpl.readFileSync(settingsPath, "utf-8"));
		if (!isRecord(parsed)) return {};
		const section = parsed.semblr;
		return isRecord(section) ? section : {};
	} catch (error) {
		warn(
			`Failed to read Semblr settings from ${settingsPath}: ${error instanceof Error ? error.message : String(error)}`,
		);
		return {};
	}
}

function asNonEmptyString(value: SettingValue | undefined): string | undefined {
	return typeof value === "string" && value.trim() ? value : undefined;
}

function parseNumericSetting(
	key: ConfigKey,
	value: SettingValue | undefined,
	defaultValue: number,
	warn: (message: string) => void,
): number {
	const numberValue =
		typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : NaN;
	if (Number.isFinite(numberValue)) return numberValue;

	warn(`Invalid numeric Semblr setting ${key}; using default ${defaultValue}`);
	return defaultValue;
}

function resolveValue(
	key: ConfigKey,
	env: SemblrConfigEnv,
	projectSettings: SettingRecord,
	globalSettings: SettingRecord,
) {
	const envValue = env[ENV_KEYS[key]];
	if (envValue !== undefined) return { value: envValue, source: "env" as const };
	if (projectSettings[key] !== undefined) return { value: projectSettings[key], source: "project" as const };
	if (globalSettings[key] !== undefined) return { value: globalSettings[key], source: "global" as const };
	return { value: undefined, source: "default" as const };
}

function resolveString(
	key: ConfigKey,
	defaultValue: string | undefined,
	env: SemblrConfigEnv,
	projectSettings: SettingRecord,
	globalSettings: SettingRecord,
): string | undefined {
	const { value } = resolveValue(key, env, projectSettings, globalSettings);
	return asNonEmptyString(value) ?? defaultValue;
}

function resolveNumber(
	key: ConfigKey,
	defaultValue: number,
	env: SemblrConfigEnv,
	projectSettings: SettingRecord,
	globalSettings: SettingRecord,
	warn: (message: string) => void,
): number {
	const { value } = resolveValue(key, env, projectSettings, globalSettings);
	return value === undefined ? defaultValue : parseNumericSetting(key, value, defaultValue, warn);
}

function resolveRoundsDir(
	env: SemblrConfigEnv,
	projectSettings: SettingRecord,
	globalSettings: SettingRecord,
	cwd: string,
	agentDir: string,
): string {
	const { value, source } = resolveValue("roundsDir", env, projectSettings, globalSettings);
	const configured = asNonEmptyString(value);
	if (!configured) return path.resolve(agentDir, "semblr", "rounds");
	if (path.isAbsolute(configured)) return configured;
	return path.resolve(source === "project" ? cwd : agentDir, configured);
}

// ─────────────────────────────────────────────
// Routing Config Resolution
// ─────────────────────────────────────────────

function resolveBoolean(value: string | undefined, defaultValue: boolean): boolean {
	if (value === undefined) return defaultValue;
	const lower = value.trim().toLowerCase();
	if (lower === "true" || lower === "1") return true;
	if (lower === "false" || lower === "0") return false;
	// Invalid values fall back to default silently (non-critical config).
	return defaultValue;
}

function parseNonNegativeInt(
	value: SettingValue | undefined,
	defaultValue: number,
	name: string,
	warn: (msg: string) => void,
): number {
	if (value === undefined) return defaultValue;
	const num = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : NaN;
	if (Number.isFinite(num) && num >= 0 && Number.isInteger(num)) return num;
	warn(`Invalid routing config: ${name} must be a non-negative integer; using default ${defaultValue}`);
	return defaultValue;
}

/** Resolves a preset name into a PhaseModelMap. Handles "current" → null mapping. */
export function resolvePreset(presetName: string): PhaseModelMap | null {
	if (!isPresetName(presetName)) return null;
	const preset = PRESET_REGISTRY[presetName];
	const map: Partial<PhaseModelMap> = {};
	for (const phase of Object.keys(preset) as PhaseName[]) {
		const val = preset[phase];
		map[phase] = val === "current" ? null : val;
	}
	return map as PhaseModelMap;
}

function isPresetName(name: string): name is PresetName {
	return (PRESET_NAMES as readonly string[]).includes(name);
}

/** Returns the preset description for a given preset name, or null if unknown. */
export function getPresetDescription(name: string): string | null {
	if (!isPresetName(name)) return null;
	return PRESET_DESCRIPTIONS[name];
}

/** Returns the set of all model IDs referenced in a PhaseModelMap (excluding nulls). */
export function collectPhaseModels(map: PhaseModelMap): string[] {
	return Object.values(map).filter((v): v is string => v !== null);
}

/**
 * Validate routing config at load time. Returns validated RoutingConfig and a
 * list of warnings. If the config is unrecoverable, returns the inert disabled config.
 *
 * @param isModelAvailable Optional callback to check model availability at session_start.
 *                         At module load time, this is not available.
 */
export function validateRoutingConfig(
	raw: Partial<RoutingConfig>,
	env: SemblrConfigEnv,
	warn: (msg: string) => void,
	isModelAvailable?: (model: string, provider: string) => boolean,
): RoutingConfig {
	// ── Resolve enabled ──
	const enabled = resolveBoolean(env.SEMBLR_ROUTING_ENABLED, raw.enabled ?? DISABLED_ROUTING_CONFIG.enabled);

	// ── Resolve preset vs phaseModels ──
	let preset: PresetName | null = raw.preset ?? null;
	let phaseModels: PhaseModelMap =
		raw.phaseModels && Object.keys(raw.phaseModels).length > 0
			? (raw.phaseModels as PhaseModelMap)
			: ({} as PhaseModelMap);

	// If both preset and phaseModels are set, phaseModels takes precedence
	if (preset !== null && Object.keys(phaseModels).length > 0) {
		warn(`Routing config: both 'preset' ("${preset}") and 'phaseModels' are set — 'preset' will be ignored.`);
		preset = null;
	}

	// Resolve preset if set (and not overridden by phaseModels)
	if (preset !== null && Object.keys(phaseModels).length === 0) {
		const resolved = resolvePreset(preset);
		if (resolved === null) {
			warn(`Routing config: unknown preset "${preset}" — falling back to empty phaseModels.`);
			preset = null;
			phaseModels = {} as PhaseModelMap;
		} else {
			phaseModels = resolved;
		}
	}

	// ── Validate numeric settings ──
	const maxSwitchesPerCycle = parseNonNegativeInt(
		raw.maxSwitchesPerCycle ?? env.SEMBLR_ROUTING_MAX_SWITCHES,
		ROUTING_DEFAULTS.maxSwitchesPerCycle,
		"maxSwitchesPerCycle",
		warn,
	);
	const minTurnsPerPhase = parseNonNegativeInt(
		raw.minTurnsPerPhase,
		ROUTING_DEFAULTS.minTurnsPerPhase,
		"minTurnsPerPhase",
		warn,
	);
	const agentCycleTimeoutSec = parseNonNegativeInt(
		raw.agentCycleTimeoutSec ?? env.SEMBLR_ROUTING_TIMEOUT,
		ROUTING_DEFAULTS.agentCycleTimeoutSec,
		"agentCycleTimeoutSec",
		warn,
	);

	// ── Validate model availability (best-effort, only when callback provided) ──
	if (isModelAvailable) {
		// Note: resolveModelId import creates a circular dep, so we do inline parsing
		for (const [phase, modelId] of Object.entries(phaseModels)) {
			if (modelId === null) continue;
			const { provider, model } = parseModelId(modelId);
			if (provider && model && !isModelAvailable(model, provider)) {
				warn(`Routing config: model "${modelId}" not available; phase "${phase}" will not trigger a switch.`);
				(phaseModels as Record<string, string | null>)[phase] = null;
			}
		}
	}

	return {
		enabled,
		preset,
		phaseModels,
		maxSwitchesPerCycle,
		minTurnsPerPhase,
		agentCycleTimeoutSec,
	};
}

/** Inline model ID parsing (avoids circular dep with resolve-model-id). */
function parseModelId(modelId: string): { provider: string; model: string } {
	const atIndex = modelId.indexOf("@");
	if (atIndex !== -1) {
		return { model: modelId.slice(0, atIndex), provider: modelId.slice(atIndex + 1) };
	}
	if (modelId.endsWith(":cloud")) {
		return { provider: "ollama-cloud", model: modelId.slice(0, -6) };
	}
	return { provider: "ollama-cloud", model: modelId };
}

/**
 * Auto-init: write semblr defaults into the global settings.json if it exists
 * but has no 'semblr' section. Idempotent — once written, subsequent calls skip.
 * Write failures are warned and execution continues gracefully.
 */
function autoInitSemblrSettings(
	settingsPath: string,
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync" | "writeFileSync">,
	warn: (message: string) => void,
): void {
	// Only auto-init if the file already exists (pi creates it)
	if (!fsImpl.existsSync(settingsPath)) return;

	let parsed: unknown;
	try {
		parsed = JSON.parse(fsImpl.readFileSync(settingsPath, "utf-8"));
	} catch {
		return; // malformed JSON — don't touch
	}

	if (!isRecord(parsed)) return;

	// Already has a semblr section — nothing to do
	if (isRecord(parsed.semblr)) return;

	// Write the full defaults into the settings file
	const semblrDefaults: SettingRecord = {
		embeddingProvider: DEFAULTS.embeddingProvider,
		embeddingModel: DEFAULTS.embeddingModel,
		embeddingMaxTokens: DEFAULTS.embeddingMaxTokens,
		groupThreshold: DEFAULTS.groupThreshold,
		minSimilarity: DEFAULTS.minSimilarity,
		embedTimeoutMs: DEFAULTS.embedTimeoutMs,
		embedMaxRetries: DEFAULTS.embedMaxRetries,
		embedBackoffMs: DEFAULTS.embedBackoffMs,
		summaryThresholdExtra: DEFAULTS.summaryThresholdExtra,
		routing: {
			enabled: false,
			preset: null,
			phaseModels: {},
			maxSwitchesPerCycle: ROUTING_DEFAULTS.maxSwitchesPerCycle,
			minTurnsPerPhase: ROUTING_DEFAULTS.minTurnsPerPhase,
			agentCycleTimeoutSec: ROUTING_DEFAULTS.agentCycleTimeoutSec,
		},
	};

	parsed.semblr = semblrDefaults;

	try {
		const serialized = JSON.stringify(parsed, null, 2);
		fsImpl.writeFileSync(settingsPath, serialized);
	} catch (error) {
		warn(
			`Failed to auto-init semblr settings in ${settingsPath}: ${
				error instanceof Error ? error.message : String(error)
			}`,
		);
	}
}

export function loadSemblrConfig(deps: SemblrConfigDeps = {}): SemblrConfig {
	const env = deps.env ?? process.env;
	const cwd = deps.cwd ?? process.cwd();
	const agentDir = deps.agentDir ?? defaultAgentDir(env);
	const fsImpl = deps.fsImpl ?? fs;
	const warn = deps.warn ?? console.warn;

	// Auto-init global settings before reading (idempotent)
	autoInitSemblrSettings(path.join(agentDir, "settings.json"), fsImpl, warn);

	const globalSettings = loadSemblrSection(path.join(agentDir, "settings.json"), fsImpl, warn);
	const projectSettings = loadSemblrSection(path.join(cwd, ".pi", "settings.json"), fsImpl, warn);
	const mergedSettings = mergeRecords(globalSettings, projectSettings);

	const roundsDir = resolveRoundsDir(env, projectSettings, globalSettings, cwd, agentDir);

	return {
		agentDir,
		embeddingProvider: resolveString(
			"embeddingProvider",
			DEFAULTS.embeddingProvider,
			env,
			mergedSettings,
			{},
		) as string,
		embeddingModel: resolveString("embeddingModel", DEFAULTS.embeddingModel, env, mergedSettings, {}) as string,
		embeddingMaxTokens: resolveNumber(
			"embeddingMaxTokens",
			DEFAULTS.embeddingMaxTokens,
			env,
			mergedSettings,
			{},
			warn,
		),
		embeddingApiUrl: resolveString("embeddingApiUrl", undefined, env, mergedSettings, {}),
		roundsDir,
		indexPath: path.join(roundsDir, "index.csv"),
		groupThreshold: resolveNumber("groupThreshold", DEFAULTS.groupThreshold, env, mergedSettings, {}, warn),
		minSimilarity: resolveNumber("minSimilarity", DEFAULTS.minSimilarity, env, mergedSettings, {}, warn),
		embedTimeoutMs: resolveNumber("embedTimeoutMs", DEFAULTS.embedTimeoutMs, env, mergedSettings, {}, warn),
		embedMaxRetries: resolveNumber("embedMaxRetries", DEFAULTS.embedMaxRetries, env, mergedSettings, {}, warn),
		embedBackoffMs: resolveNumber("embedBackoffMs", DEFAULTS.embedBackoffMs, env, mergedSettings, {}, warn),
		summaryThresholdExtra: resolveNumber(
			"summaryThresholdExtra",
			DEFAULTS.summaryThresholdExtra,
			env,
			mergedSettings,
			{},
			warn,
		),
		routing: resolveRouting(env, mergedSettings, warn),
	};
}

function resolveRouting(
	env: SemblrConfigEnv,
	mergedSettings: SettingRecord,
	warn: (msg: string) => void,
): RoutingConfig {
	const routingSettings = isRecord(mergedSettings.routing) ? mergedSettings.routing : {};

	return validateRoutingConfig(
		{
			enabled: typeof routingSettings.enabled === "boolean" ? routingSettings.enabled : undefined,
			preset: typeof routingSettings.preset === "string" ? (routingSettings.preset as PresetName | null) : null,
			phaseModels: isRecord(routingSettings.phaseModels)
				? (routingSettings.phaseModels as unknown as PhaseModelMap)
				: undefined,
			maxSwitchesPerCycle:
				typeof routingSettings.maxSwitchesPerCycle === "number" ? routingSettings.maxSwitchesPerCycle : undefined,
			minTurnsPerPhase:
				typeof routingSettings.minTurnsPerPhase === "number" ? routingSettings.minTurnsPerPhase : undefined,
			agentCycleTimeoutSec:
				typeof routingSettings.agentCycleTimeoutSec === "number" ? routingSettings.agentCycleTimeoutSec : undefined,
		},
		env,
		warn,
	);
}

export { DEFAULTS as SEMBLR_CONFIG_DEFAULTS };
