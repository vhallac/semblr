import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { DEFAULT_MAX_RECENCY_ENTRIES, DEFAULT_PROMPT_TRUNCATION } from "./context-format.ts";
import { DEFAULT_PROMPT_NOISE_CLEANUP } from "./round-capture.ts";
import { DEFAULT_CONTEXT_BUDGET_RATIO, DEFAULT_MAX_RELEVANCE_ENTRIES } from "./search-interactions.ts";

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
	hybridSemanticWeight: number;
	/** 0 disables the automatic context-size warning; set a positive token count to enable it. */
	summaryThresholdExtra: number;
	/** Chars kept from the head of each recency/relevance list prompt; non-positive disables truncation. */
	contextPromptHeadChars: number;
	/** Chars kept from the tail of each recency/relevance list prompt. */
	contextPromptTailChars: number;
	/** Fraction of the context window the relevance-list injection may occupy at best score (0–1). */
	contextBudgetRatio: number;
	/** Hard cap on relevance-list entries. */
	contextRelevanceMaxEntries: number;
	/** Hard cap on recency-list entries across all groups. */
	contextRecencyMaxEntries: number;
	/** Code fences longer than this many chars collapse to a placeholder in embedding inputs; 0 disables. */
	promptNoiseFenceMaxChars: number;
	/** JSON dumps longer than this many chars collapse to a placeholder in embedding inputs; 0 disables. */
	promptNoiseJsonMaxChars: number;
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
	SEMBLR_HYBRID_SEMANTIC_WEIGHT?: string;
	SEMBLR_SUMMARY_THRESHOLD_EXTRA?: string;
	SEMBLR_CONTEXT_PROMPT_HEAD_CHARS?: string;
	SEMBLR_CONTEXT_PROMPT_TAIL_CHARS?: string;
	SEMBLR_CONTEXT_BUDGET_RATIO?: string;
	SEMBLR_CONTEXT_RELEVANCE_MAX_ENTRIES?: string;
	SEMBLR_CONTEXT_RECENCY_MAX_ENTRIES?: string;
	SEMBLR_PROMPT_NOISE_FENCE_MAX_CHARS?: string;
	SEMBLR_PROMPT_NOISE_JSON_MAX_CHARS?: string;
}

export interface SemblrConfigDeps {
	cwd?: string;
	agentDir?: string;
	env?: SemblrConfigEnv;
	fsImpl?: Pick<typeof fs, "existsSync" | "readFileSync">;
	warn?: (message: string) => void;
}

type ConfigKey = keyof Omit<SemblrConfig, "agentDir" | "indexPath">;
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
	hybridSemanticWeight: 0.7,
	summaryThresholdExtra: 0,
	contextPromptHeadChars: DEFAULT_PROMPT_TRUNCATION.headChars,
	contextPromptTailChars: DEFAULT_PROMPT_TRUNCATION.tailChars,
	contextBudgetRatio: DEFAULT_CONTEXT_BUDGET_RATIO,
	contextRelevanceMaxEntries: DEFAULT_MAX_RELEVANCE_ENTRIES,
	contextRecencyMaxEntries: DEFAULT_MAX_RECENCY_ENTRIES,
	promptNoiseFenceMaxChars: DEFAULT_PROMPT_NOISE_CLEANUP.fenceMaxChars,
	promptNoiseJsonMaxChars: DEFAULT_PROMPT_NOISE_CLEANUP.jsonMaxChars,
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
	hybridSemanticWeight: "SEMBLR_HYBRID_SEMANTIC_WEIGHT",
	summaryThresholdExtra: "SEMBLR_SUMMARY_THRESHOLD_EXTRA",
	contextPromptHeadChars: "SEMBLR_CONTEXT_PROMPT_HEAD_CHARS",
	contextPromptTailChars: "SEMBLR_CONTEXT_PROMPT_TAIL_CHARS",
	contextBudgetRatio: "SEMBLR_CONTEXT_BUDGET_RATIO",
	contextRelevanceMaxEntries: "SEMBLR_CONTEXT_RELEVANCE_MAX_ENTRIES",
	contextRecencyMaxEntries: "SEMBLR_CONTEXT_RECENCY_MAX_ENTRIES",
	promptNoiseFenceMaxChars: "SEMBLR_PROMPT_NOISE_FENCE_MAX_CHARS",
	promptNoiseJsonMaxChars: "SEMBLR_PROMPT_NOISE_JSON_MAX_CHARS",
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
		hybridSemanticWeight: Math.max(
			0,
			Math.min(
				1,
				resolveNumber("hybridSemanticWeight", DEFAULTS.hybridSemanticWeight, env, mergedSettings, {}, warn),
			),
		),
		summaryThresholdExtra: resolveNumber(
			"summaryThresholdExtra",
			DEFAULTS.summaryThresholdExtra,
			env,
			mergedSettings,
			{},
			warn,
		),
		contextPromptHeadChars: resolveNumber(
			"contextPromptHeadChars",
			DEFAULTS.contextPromptHeadChars,
			env,
			mergedSettings,
			{},
			warn,
		),
		contextPromptTailChars: resolveNumber(
			"contextPromptTailChars",
			DEFAULTS.contextPromptTailChars,
			env,
			mergedSettings,
			{},
			warn,
		),
		contextBudgetRatio: Math.max(
			0,
			Math.min(1, resolveNumber("contextBudgetRatio", DEFAULTS.contextBudgetRatio, env, mergedSettings, {}, warn)),
		),
		contextRelevanceMaxEntries: Math.max(
			0,
			Math.floor(
				resolveNumber(
					"contextRelevanceMaxEntries",
					DEFAULTS.contextRelevanceMaxEntries,
					env,
					mergedSettings,
					{},
					warn,
				),
			),
		),
		contextRecencyMaxEntries: Math.max(
			0,
			Math.floor(
				resolveNumber("contextRecencyMaxEntries", DEFAULTS.contextRecencyMaxEntries, env, mergedSettings, {}, warn),
			),
		),
		promptNoiseFenceMaxChars: Math.max(
			0,
			Math.floor(
				resolveNumber("promptNoiseFenceMaxChars", DEFAULTS.promptNoiseFenceMaxChars, env, mergedSettings, {}, warn),
			),
		),
		promptNoiseJsonMaxChars: Math.max(
			0,
			Math.floor(
				resolveNumber("promptNoiseJsonMaxChars", DEFAULTS.promptNoiseJsonMaxChars, env, mergedSettings, {}, warn),
			),
		),
	};
}

export { DEFAULTS as SEMBLR_CONFIG_DEFAULTS };
