import { describe, expect, it } from "vitest";
import {
	API_VERSION,
	type ContextProvider,
	createProviderRegistry,
	isValidProviderIdentity,
	registerProvider,
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
});
