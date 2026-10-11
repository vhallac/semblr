/**
 * Context-provider API contract (#111).
 *
 * The public, versioned contract any extension implements to become a
 * _context provider_: the seam that lets third parties register context
 * sections alongside semblr's built-ins.
 *
 * Binding invariants for all implementors (see #111):
 * - No cross-section mutation: a provider reads `buildContext`, never
 *   mutates shared build state.
 * - Frozen during the tool loop: content is computed at build time, not
 *   re-evaluated between tool calls.
 * - Fail-by-omission: a failing provider never corrupts the whole context.
 *
 * `dependsOn` / phase-2 rendering is a unit of its own (see the #111 plan);
 * registration-level dependsOn validation lands there.
 */

/**
 * The apiVersion this contract implementation assigns.
 * Changes after first release are additive, or flagged by an apiVersion bump.
 */
export const API_VERSION = 1;

/** apiVersions this semblr build supports. Ext 1.b: anything else is skipped. */
export const SUPPORTED_API_VERSIONS: readonly number[] = [API_VERSION];

/**
 * Budget hints a provider declares. These are static; dynamic budget curves
 * are provider-internal (F2 resolution) — computed inside `render` within
 * the allocation ceiling. Priority ordering is decided downstream (#113).
 */
export interface ProviderBudgetHints {
	/** Relative ordering hint among providers (lower renders earlier within a phase). */
	priority: number;
	/** Preferred token allocation; the allocation is a ceiling, not a quota. */
	preferredTokens: number;
	/** Below this allocation `render` is not invoked this round (arbitration in #113). */
	minimumViableTokens: number;
}

/** The per-round allocation handed to a provider's `render`. */
export interface ProviderBudget {
	/** The token ceiling allocated to this provider this round. */
	allocatedTokens: number;
}

/**
 * The shared, read-only build context handed to every provider's `render`.
 *
 * FROZEN FIELD LIST (binding): the fields below are the complete set a
 * provider may read. Changes are additive only; removals or type changes
 * require an apiVersion bump.
 */
export interface BuildContext {
	/** The current user prompt for this round. */
	prompt: string;
	/** Epoch milliseconds at the start of this context build. */
	now: number;
}

/**
 * A context provider. The extension author implements this to participate
 * in every context build.
 */
export interface ContextProvider {
	/** Stable unique identity, `semblr.<name>`. */
	identity: string;
	/** The contract version this provider implements. */
	apiVersion: number;
	/** Static budget hints (see {@link ProviderBudgetHints}). */
	budgetHints: ProviderBudgetHints;
	/**
	 * Produce this round's section content, or `null` for nothing to inject.
	 * Ext 2.a: throwing here drops the section and the build continues
	 * (fail-by-omission, enforced by the caller — full semantics in #112).
	 */
	render(buildContext: BuildContext, budget: ProviderBudget): string | null;
}

/** Ordered set of registered providers (registration order = render order). */
export interface ProviderRegistry {
	providers: Map<string, ContextProvider>;
}

export type RegistrationStatus = "registered" | "duplicate-identity" | "unsupported-api-version" | "invalid-identity";

export interface RegistrationResult {
	status: RegistrationStatus;
	/** Human-readable diagnostic when the provider was not registered. */
	diagnostic?: string;
}

/** Identity shape: `semblr.<name>` with a non-empty name segment. */
export function isValidProviderIdentity(identity: string): boolean {
	return /^semblr\.[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(identity);
}

export function createProviderRegistry(): ProviderRegistry {
	return { providers: new Map() };
}

/**
 * Register a provider into the registry.
 *
 * Ext 1.a: a duplicate identity fails with a diagnostic; the original
 * provider stands. Ext 1.b: an unsupported `apiVersion` skips the provider
 * with a diagnostic — registration never hard-fails the build.
 */
export function registerProvider(registry: ProviderRegistry, provider: ContextProvider): RegistrationResult {
	if (!isValidProviderIdentity(provider.identity)) {
		return {
			status: "invalid-identity",
			diagnostic: `invalid provider identity "${provider.identity}" — must match "semblr.<name>"`,
		};
	}
	if (!SUPPORTED_API_VERSIONS.includes(provider.apiVersion)) {
		return {
			status: "unsupported-api-version",
			diagnostic: `provider "${provider.identity}" declares apiVersion ${provider.apiVersion}; supported: ${SUPPORTED_API_VERSIONS.join(", ")} — skipped`,
		};
	}
	if (registry.providers.has(provider.identity)) {
		return {
			status: "duplicate-identity",
			diagnostic: `provider identity "${provider.identity}" is already registered — original provider stands`,
		};
	}
	registry.providers.set(provider.identity, provider);
	return { status: "registered" };
}
