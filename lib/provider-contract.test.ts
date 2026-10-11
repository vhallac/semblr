import { describe, expect, it } from "vitest";
import {
	API_VERSION,
	type BuildContext,
	type ContextProvider,
	createProviderRegistry,
	isValidProviderIdentity,
	registerProvider,
	renderProviders,
	type SectionsView,
} from "./provider-contract.ts";

function makeProvider(overrides: Partial<ContextProvider> = {}): ContextProvider {
	return {
		identity: "semblr.test",
		apiVersion: API_VERSION,
		budgetHints: { priority: 1, preferredTokens: 100, minimumViableTokens: 10 },
		render: () => "section content",
		...overrides,
	};
}

describe("provider contract registration", () => {
	it("registers a provider with a valid identity and supported apiVersion", () => {
		const registry = createProviderRegistry();
		const provider = makeProvider({ render: () => "hello" });

		const result = registerProvider(registry, provider);

		expect(result.status).toBe("registered");
		expect(registry.providers.get("semblr.test")).toBe(provider);
	});

	it("keeps insertion order of registered providers", () => {
		const registry = createProviderRegistry();
		registerProvider(registry, makeProvider({ identity: "semblr.a" }));
		registerProvider(registry, makeProvider({ identity: "semblr.b" }));

		expect([...registry.providers.keys()]).toEqual(["semblr.a", "semblr.b"]);
	});

	it("rejects a duplicate identity with a diagnostic and the original stands", () => {
		const registry = createProviderRegistry();
		const original = makeProvider({ render: () => "original" });
		registerProvider(registry, original);

		const result = registerProvider(registry, makeProvider({ render: () => "dupe" }));

		expect(result.status).toBe("duplicate-identity");
		expect(result.diagnostic).toMatch(/already registered/);
		expect(registry.providers.get("semblr.test")?.render(null as never, null as never)).toBe("original");
	});

	it("skips a provider declaring an unsupported apiVersion, without hard failure", () => {
		const registry = createProviderRegistry();

		const result = registerProvider(registry, makeProvider({ apiVersion: API_VERSION + 1 }));

		expect(result.status).toBe("unsupported-api-version");
		expect(result.diagnostic).toMatch(/apiVersion 2/);
		expect(registry.providers.size).toBe(0);
	});

	it.each(["", "no-prefix", "semblr.", "semblr.", "other.thing", "semblr. bad name"])(
		"rejects invalid identity %j",
		(identity: string) => {
			const registry = createProviderRegistry();

			const result = registerProvider(registry, makeProvider({ identity }));

			expect(result.status).toBe("invalid-identity");
			expect(registry.providers.size).toBe(0);
		},
	);

	it.each(["semblr.a", "semblr.relevance-section", "semblr.preamble_v2", "semblr.x.1"])(
		"accepts valid identity %j",
		(identity: string) => {
			expect(isValidProviderIdentity(identity)).toBe(true);
		},
	);

	it("rejects an unknown dependency at registration", () => {
		const registry = createProviderRegistry();

		const result = registerProvider(
			registry,
			makeProvider({ identity: "semblr.dep", dependsOn: ["semblr.missing"] }),
		);

		expect(result.status).toBe("unknown-dependency");
		expect(result.diagnostic).toMatch(/semblr\.missing/);
		expect(registry.providers.size).toBe(0);
	});

	it("rejects self-dependency at registration", () => {
		const registry = createProviderRegistry();

		const result = registerProvider(registry, makeProvider({ identity: "semblr.self", dependsOn: ["semblr.self"] }));

		expect(result.status).toBe("circular-dependency");
		expect(registry.providers.size).toBe(0);
	});

	it("guards against a registration that would close a cycle (direct registry mutation)", () => {
		// A cycle cannot be formed through the public API (deps must reference
		// already-registered providers), but the registry can be mutated
		// directly; registration still guards against it.
		const registry = createProviderRegistry();
		registerProvider(registry, makeProvider({ identity: "semblr.a" }));
		registry.providers.set("semblr.x", makeProvider({ identity: "semblr.x", dependsOn: ["semblr.a"] }));

		const backEdge = registerProvider(registry, makeProvider({ identity: "semblr.z", dependsOn: ["semblr.x"] }));
		expect(backEdge.status).toBe("registered"); // x→a, z→x: no cycle

		// Mutate x to depend on the not-yet-registered w; registering w then
		// would close w→x→w.
		registry.providers.set("semblr.x", makeProvider({ identity: "semblr.x", dependsOn: ["semblr.w"] }));
		const cycle = registerProvider(registry, makeProvider({ identity: "semblr.w", dependsOn: ["semblr.x"] }));
		expect(cycle.status).toBe("circular-dependency");
		expect(cycle.diagnostic).toMatch(/cycle/);
		expect(registry.providers.size).toBe(3); // a, x, z — w not added
	});
});

describe("two-phase render walk", () => {
	const base: Omit<BuildContext, "sections"> = { prompt: "hello", now: 1_000 };

	function populatedRegistry(...providers: ContextProvider[]) {
		const registry = createProviderRegistry();
		for (const p of providers) {
			const result = registerProvider(registry, p);
			if (result.status !== "registered") throw new Error(`unexpected: ${result.diagnostic}`);
		}
		return registry;
	}

	it("renders independents first (by priority), then dependents", () => {
		const order: string[] = [];
		const registry = populatedRegistry(
			makeProvider({
				identity: "semblr.low",
				budgetHints: { priority: 5, preferredTokens: 100, minimumViableTokens: 10 },
				render: () => {
					order.push("low");
					return "low-content";
				},
			}),
			makeProvider({
				identity: "semblr.high",
				budgetHints: { priority: 1, preferredTokens: 100, minimumViableTokens: 10 },
				render: () => {
					order.push("high");
					return "high-content";
				},
			}),
			makeProvider({
				identity: "semblr.second",
				dependsOn: ["semblr.high"],
				render: () => {
					order.push("second");
					return "second-content";
				},
			}),
		);

		const result = renderProviders(registry, base);

		expect(order).toEqual(["high", "low", "second"]);
		expect([...result.sections]).toEqual([
			["semblr.high", "high-content"],
			["semblr.low", "low-content"],
			["semblr.second", "second-content"],
		]);
	});

	it("gives phase-2 providers a sections view of what was built so far", () => {
		const registry = populatedRegistry(
			makeProvider({ identity: "semblr.base", render: () => "base-content" }),
			makeProvider({
				identity: "semblr.reader",
				dependsOn: ["semblr.base"],
				render: (ctx) => `saw:${ctx.sections["semblr.base"]}`,
			}),
		);

		const result = renderProviders(registry, base);

		expect(result.sections.get("semblr.reader")).toBe("saw:base-content");
	});

	it("gives phase-1 providers an empty sections view", () => {
		const registry = populatedRegistry(
			makeProvider({ identity: "semblr.first", render: (ctx) => `count:${Object.keys(ctx.sections).length}` }),
		);

		const result = renderProviders(registry, base);

		expect(result.sections.get("semblr.first")).toBe("count:0");
	});

	it("sections view is frozen and does not expose later sections", () => {
		let captured: SectionsView | undefined;
		const registry = populatedRegistry(
			makeProvider({
				identity: "semblr.capture",
				render: (ctx) => {
					captured = ctx.sections;
					return "captured";
				},
			}),
			makeProvider({ identity: "semblr.later", render: () => "later" }),
		);

		renderProviders(registry, base);

		expect(Object.isFrozen(captured)).toBe(true);
		expect("semblr.later" in captured!).toBe(false);
		expect(captured!["semblr.capture"]).toBeUndefined(); // own section not in its own view
	});

	it("drops a throwing provider with a diagnostic and the build continues", () => {
		const registry = populatedRegistry(
			makeProvider({
				identity: "semblr.boom",
				render: () => {
					throw new Error("kaboom");
				},
			}),
			makeProvider({ identity: "semblr.survivor", render: () => "still-here" }),
		);

		const result = renderProviders(registry, base);

		expect(result.sections.get("semblr.boom")).toBeUndefined();
		expect(result.sections.get("semblr.survivor")).toBe("still-here");
		const boom = result.outcomes.find((o) => o.identity === "semblr.boom");
		expect(boom?.status).toBe("dropped");
		expect(boom?.diagnostic).toMatch(/kaboom/);
	});

	it("skips dependents of a dropped provider (fail-by-omission cascades)", () => {
		const registry = populatedRegistry(
			makeProvider({
				identity: "semblr.boom",
				render: () => {
					throw new Error("kaboom");
				},
			}),
			makeProvider({ identity: "semblr.dep", dependsOn: ["semblr.boom"], render: () => "never" }),
		);

		const result = renderProviders(registry, base);

		expect(result.sections.has("semblr.dep")).toBe(false);
		expect(result.outcomes.find((o) => o.identity === "semblr.dep")?.status).toBe("skipped-dependency");
	});

	it("treats a null render as empty — no section, no cascade", () => {
		const registry = populatedRegistry(
			makeProvider({ identity: "semblr.nothing", render: () => null }),
			makeProvider({ identity: "semblr.dep", dependsOn: ["semblr.nothing"], render: () => "dep-content" }),
		);

		const result = renderProviders(registry, base);

		expect(result.sections.has("semblr.nothing")).toBe(false);
		// null is "nothing to inject", not a failure: the dependent still renders.
		expect(result.sections.get("semblr.dep")).toBe("dep-content");
		expect(result.outcomes.find((o) => o.identity === "semblr.nothing")?.status).toBe("empty");
	});

	it("passes the provider's preferredTokens as the allocated ceiling", () => {
		let seen: number | undefined;
		const registry = populatedRegistry(
			makeProvider({
				identity: "semblr.budgeted",
				budgetHints: { priority: 1, preferredTokens: 333, minimumViableTokens: 10 },
				render: (_ctx, budget) => {
					seen = budget.allocatedTokens;
					return "ok";
				},
			}),
		);

		renderProviders(registry, base);

		expect(seen).toBe(333);
	});
});
