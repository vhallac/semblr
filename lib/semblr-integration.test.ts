import { describe, expect, it } from "vitest";
import { embeddingClientDeps } from "../src/semblr.ts";

describe("embeddingClientDeps", () => {
	it("overrides only the embedding model for a model-specific query", () => {
		const modelRegistry = { marker: "registry" };
		const deps = embeddingClientDeps({ modelRegistry } as never, "historical-model");

		expect(deps.modelRegistry).toBe(modelRegistry);
		expect(deps.config.embeddingModel).toBe("historical-model");
		expect(deps.config.embeddingProvider).toBeTruthy();
		expect(deps.config.embedMaxRetries).toBeGreaterThan(0);
	});
});
