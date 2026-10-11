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

	it("accepts a required edge to an unregistered dependency (render-time resolution, decision 022)", () => {
		const registry = createProviderRegistry();

		const result = registerProvider(
			registry,
			makeProvider({ identity: "semblr.dep", dependsOn: [{ name: "semblr.missing", required: true }] }),
		);

		// Decision 022: existence is resolved at render time, not registration —
		// an unregistered dependency name is accepted (required-absent disables
		// the provider statically later).
		expect(result.status).toBe("registered");
		expect(registry.providers.get("semblr.dep")).toBeDefined();
	});

	it("registers a provider with an optional edge to an unregistered dependency", () => {
		const registry = createProviderRegistry();

		const result = registerProvider(
			registry,
			makeProvider({ identity: "semblr.opt", dependsOn: [{ name: "semblr.maybe", required: false }] }),
		);

		expect(result.status).toBe("registered");
		expect(registry.providers.get("semblr.opt")).toBeDefined();
	});

	it("rejects self-dependency at registration", () => {
		const registry = createProviderRegistry();

		const result = registerProvider(registry, makeProvider({ identity: "semblr.self", dependsOn: [{ name: "semblr.self", required: true }] }));

		expect(result.status).toBe("circular-dependency");
		expect(registry.providers.size).toBe(0);
	});

	it("guards against a registration that would close a cycle (direct registry mutation)", () => {
		// A cycle cannot be formed through the public API (deps must reference
		// already-registered providers), but the registry can be mutated
		// directly; registration still guards against it.
		const registry = createProviderRegistry();
		registerProvider(registry, makeProvider({ identity: "semblr.a" }));
		registry.providers.set("semblr.x", makeProvider({ identity: "semblr.x", dependsOn: [{ name: "semblr.a", required: true }] }));

		const backEdge = registerProvider(registry, makeProvider({ identity: "semblr.z", dependsOn: [{ name: "semblr.x", required: true }] }));
		expect(backEdge.status).toBe("registered"); // x→a, z→x: no cycle

		// Mutate x to depend on the not-yet-registered w; registering w then
		// would close w→x→w.
		registry.providers.set("semblr.x", makeProvider({ identity: "semblr.x", dependsOn: [{ name: "semblr.w", required: true }] }));
		const cycle = registerProvider(registry, makeProvider({ identity: "semblr.w", dependsOn: [{ name: "semblr.x", required: true }] }));
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
				dependsOn: [{ name: "semblr.high", required: true }],
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
				dependsOn: [{ name: "semblr.base", required: true }],
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
			makeProvider({ identity: "semblr.dep", dependsOn: [{ name: "semblr.boom", required: true }], render: () => "never" }),
		);

		const result = renderProviders(registry, base);

		expect(result.sections.has("semblr.dep")).toBe(false);
		expect(result.outcomes.find((o) => o.identity === "semblr.dep")?.status).toBe("skipped-dependency");
	});

	it("treats a null render as empty — no section, no cascade", () => {
		const registry = populatedRegistry(
			makeProvider({ identity: "semblr.nothing", render: () => null }),
			makeProvider({ identity: "semblr.dep", dependsOn: [{ name: "semblr.nothing", required: true }], render: () => "dep-content" }),
		);

		const result = renderProviders(registry, base);

		expect(result.sections.has("semblr.nothing")).toBe(false);
		// null is "nothing to inject", not a failure: the dependent still renders.
		expect(result.sections.get("semblr.dep")).toBe("dep-content");
		expect(result.outcomes.find((o) => o.identity === "semblr.nothing")?.status).toBe("empty");
	});

	it("renders a phase-2-on-phase-2 dependency in view (dep's section present)", () => {
		const registry = populatedRegistry(
			makeProvider({ identity: "semblr.c", render: () => "c-content" }),
			makeProvider({
				identity: "semblr.a",
				dependsOn: [{ name: "semblr.c", required: true }],
				budgetHints: { priority: 5, preferredTokens: 100, minimumViableTokens: 10 },
				render: (ctx) => `A saw c=${ctx.sections["semblr.c"]}`,
			}),
			makeProvider({
				identity: "semblr.b",
				dependsOn: [{ name: "semblr.a", required: true }],
				budgetHints: { priority: 0, preferredTokens: 100, minimumViableTokens: 10 },
				render: (ctx) => `B saw a=${ctx.sections["semblr.a"]}`,
			}),
		);

		const result = renderProviders(registry, base);

		// Topological order: c → a → b, despite b's higher priority than a.
		expect(result.outcomes.map((o) => o.identity)).toEqual(["semblr.c", "semblr.a", "semblr.b"]);
		expect(result.sections.get("semblr.b")).toBe("B saw a=A saw c=c-content");
	});

	it("skips the whole phase-2 mid-chain when a middle provider throws", () => {
		const registry = populatedRegistry(
			makeProvider({ identity: "semblr.root", render: () => "root" }),
			makeProvider({
				identity: "semblr.mid",
				dependsOn: [{ name: "semblr.root", required: true }],
				render: () => {
					throw new Error("mid kaboom");
				},
			}),
			makeProvider({ identity: "semblr.leaf", dependsOn: [{ name: "semblr.mid", required: true }], render: () => "never" }),
		);

		const result = renderProviders(registry, base);

		expect(result.outcomes.find((o) => o.identity === "semblr.mid")?.status).toBe("dropped");
		expect(result.outcomes.find((o) => o.identity === "semblr.leaf")?.status).toBe("skipped-dependency");
		expect(result.outcomes.find((o) => o.identity === "semblr.leaf")?.diagnostic).toMatch(/semblr.mid/);
	});

	it("renders a dependent whose declared dependency returned null, observing undefined", () => {
		const registry = populatedRegistry(
			makeProvider({ identity: "semblr.void", render: () => null }),
			makeProvider({
				identity: "semblr.dep",
				dependsOn: [{ name: "semblr.void", required: true }],
				render: (ctx) => `dep saw ${ctx.sections["semblr.void"]}`,
			}),
		);

		const result = renderProviders(registry, base);

		// null is "nothing to inject", not a failure: the dependent renders and
		// observes undefined for the absent section (decision 021 semantics).
		expect(result.outcomes.find((o) => o.identity === "semblr.dep")?.status).toBe("rendered");
		expect(result.sections.get("semblr.dep")).toBe("dep saw undefined");
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
