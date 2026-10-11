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
 * Rendering is two-phase: providers without `dependsOn` render first
 * (ordered by `budgetHints.priority`), then dependents render with a
 * read-only view of the sections built so far. A provider whose dependency
 * was dropped is dropped too (fail-by-omission cascades); the build continues.
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

/**
 * Read-only view of the sections built so far in this context build,
 * keyed by provider identity. Present (empty) for phase-1 providers;
 * progressively filled for phase-2 providers. Providers MUST NOT rely on
 * the presence of any particular section — fail-by-omission means a
 * dependency may be absent.
 */
export type SectionsView = Readonly<Record<string, string>>;

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
	/** Additive field: read-only sections built so far (empty for phase 1). */
	readonly sections: SectionsView;
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
	 * Identities this provider's content depends on (phase-2 providers).
	 * All dependencies must be registered providers; cycles are rejected
	 * at registration.
	 */
	dependsOn?: readonly string[];
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

export type RegistrationStatus =
	| "registered"
	| "duplicate-identity"
	| "unsupported-api-version"
	| "invalid-identity"
	| "unknown-dependency"
	| "circular-dependency";

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
	if (provider.dependsOn?.includes(provider.identity)) {
		return {
			status: "circular-dependency",
			diagnostic: `provider "${provider.identity}" depends on itself`,
		};
	}
	const missing = provider.dependsOn?.filter((dep) => !registry.providers.has(dep)) ?? [];
	if (missing.length > 0) {
		return {
			status: "unknown-dependency",
			diagnostic: `provider "${provider.identity}" depends on unregistered provider(s): ${missing.join(", ")}`,
		};
	}
	if (reaches(provider.identity, provider.dependsOn ?? [], registry)) {
		return {
			status: "circular-dependency",
			diagnostic: `provider "${provider.identity}" would create a dependsOn cycle`,
		};
	}
	registry.providers.set(provider.identity, provider);
	return { status: "registered" };
}

/** Frozen record snapshot of the sections map, for the read-only view. */
function snapshot(sections: Map<string, string>): SectionsView {
	return Object.freeze(Object.fromEntries(sections));
}

/** True if following the dependency edges from `deps` reaches `identity`. */
function reaches(identity: string, deps: readonly string[], registry: ProviderRegistry): boolean {
	const seen = new Set<string>();
	const stack = [...deps];
	while (stack.length > 0) {
		const current = stack.pop()!;
		if (current === identity) return true;
		if (seen.has(current)) continue;
		seen.add(current);
		const provider = registry.providers.get(current);
		if (provider?.dependsOn) stack.push(...provider.dependsOn);
	}
	return false;
}

export type RenderStatus = "rendered" | "empty" | "dropped" | "skipped-dependency";

export interface RenderedSection {
	identity: string;
	status: RenderStatus;
	/** Content when status is "rendered". */
	content?: string;
	/** Human-readable diagnostic when the section was dropped. */
	diagnostic?: string;
}

export interface RenderResult {
	/** Successfully rendered content, in render order (identity -> content). */
	sections: Map<string, string>;
	/** Per-provider outcome, in render order. */
	outcomes: RenderedSection[];
}

/**
 * Two-phase render walk over the registry.
 *
 * Phase 1: providers without `dependsOn`, ordered by priority (lower first),
 * ties by registration order. Phase 2: providers with `dependsOn`, same
 * ordering, rendered after all independents with a read-only view of the
 * sections built so far.
 *
 * Fail-by-omission (ext 2.a): a provider returning `null` yields no section;
 * a provider throwing has its section dropped with a diagnostic. Either way
 * the build continues — but dependents of a dropped provider are skipped,
 * so the omission cascades.
 */
export function renderProviders(registry: ProviderRegistry, base: Omit<BuildContext, "sections">): RenderResult {
	const all = [...registry.providers.values()];
	const byPriority = (a: ContextProvider, b: ContextProvider) => a.budgetHints.priority - b.budgetHints.priority;
	const phase1 = all.filter((p) => !p.dependsOn || p.dependsOn.length === 0).sort(byPriority);
	const phase2 = all.filter((p) => p.dependsOn && p.dependsOn.length > 0).sort(byPriority);

	const sections = new Map<string, string>();
	const dropped = new Set<string>();
	const outcomes: RenderedSection[] = [];

	const renderOne = (provider: ContextProvider, phaseSections: SectionsView): void => {
		const buildContext: BuildContext = Object.freeze({ ...base, sections: phaseSections });
		let content: string | null;
		try {
			content = provider.render(buildContext, { allocatedTokens: provider.budgetHints.preferredTokens });
		} catch (error) {
			dropped.add(provider.identity);
			outcomes.push({
				identity: provider.identity,
				status: "dropped",
				diagnostic: `provider "${provider.identity}" threw during render: ${error instanceof Error ? error.message : String(error)}`,
			});
			return;
		}
		if (content === null) {
			outcomes.push({ identity: provider.identity, status: "empty" });
			return;
		}
		sections.set(provider.identity, content);
		outcomes.push({ identity: provider.identity, status: "rendered", content });
	};

	for (const provider of phase1) renderOne(provider, snapshot(sections));
	for (const provider of phase2) {
		const failedDep = provider.dependsOn!.find((dep) => dropped.has(dep));
		if (failedDep !== undefined) {
			dropped.add(provider.identity);
			outcomes.push({
				identity: provider.identity,
				status: "skipped-dependency",
				diagnostic: `provider "${provider.identity}" skipped: dependency "${failedDep}" produced no section`,
			});
			continue;
		}
		// Phase-2 providers see only the sections built so far — frozen snapshot.
		renderOne(provider, snapshot(sections));
	}

	return { sections, outcomes };
}
