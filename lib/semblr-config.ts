import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ─────────────────────────────────────────────
// Multi-Model Routing Types
// ─────────────────────────────────────────────

/** Phase names the LLM can self-report via semblr_report_phase tool. */
export type PhaseName = "exploring" | "planning" | "executing" | "stuck" | "verifying" | "reporting";

/** Map from phase to model ID. `null` means stay on the current model. */
export type PhaseModelMap = Record<PhaseName, string | null>;

/** Configuration for multi-model routing. */
export interface MultiModelRoutingConfig {
	/** Opt-in toggle. Default false. */
	enabled: boolean;
	/** Maximum model switches per agent cycle. */
	maxSwitches: number;
	/** Phase → model ID mapping. */
	phaseModelMap: PhaseModelMap;
}

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
	/** Multi-model routing configuration (experimental). */
	multiModelRouting: MultiModelRoutingConfig;
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
}

export interface SemblrConfigDeps {
	cwd?: string;
	agentDir?: string;
	env?: SemblrConfigEnv;
	fsImpl?: Pick<typeof fs, "existsSync" | "readFileSync">;
	warn?: (message: string) => void;
}

type ConfigKey = keyof Omit<SemblrConfig, "agentDir" | "indexPath" | "multiModelRouting">;
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
	fsImpl: Pick<typeof fs, "existsSync" | "readFileSync">,
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

/** Hardcoded MVP phase → model map (Ollama-Cloud naming convention, from issue #86 comment #2). */
const DEFAULT_PHASE_MODEL_MAP: PhaseModelMap = {
	exploring: null,
	planning: "deepseek-v4-flash:cloud",
	executing: "glm-5.2:cloud",
	stuck: "kimi-k2.6:cloud",
	verifying: "minimax-m3:cloud",
	reporting: "gemma4:12b:cloud",
};

function resolveBoolean(value: string | undefined, defaultValue: boolean): boolean {
	if (value === undefined) return defaultValue;
	const lower = value.trim().toLowerCase();
	if (lower === "true" || lower === "1") return true;
	if (lower === "false" || lower === "0") return false;
	// Invalid values fall back to default silently (non-critical config).
	return defaultValue;
}

function resolveMultiModelRouting(env: SemblrConfigEnv): MultiModelRoutingConfig {
	return {
		enabled: resolveBoolean(env.SEMBLR_ROUTING_ENABLED, false),
		maxSwitches: 3,
		phaseModelMap: DEFAULT_PHASE_MODEL_MAP,
	};
}

export function loadSemblrConfig(deps: SemblrConfigDeps = {}): SemblrConfig {
	const env = deps.env ?? process.env;
	const cwd = deps.cwd ?? process.cwd();
	const agentDir = deps.agentDir ?? defaultAgentDir(env);
	const fsImpl = deps.fsImpl ?? fs;
	const warn = deps.warn ?? console.warn;

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
		multiModelRouting: resolveMultiModelRouting(env),
	};
}

export { DEFAULTS as SEMBLR_CONFIG_DEFAULTS };
