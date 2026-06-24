import { describe, expect, it } from "vitest";
import { resolveModelId } from "./resolve-model-id.ts";

describe("resolveModelId", () => {
	it("strips :cloud suffix and returns ollama provider", () => {
		expect(resolveModelId("glm-5.2:cloud")).toEqual({ provider: "ollama", model: "glm-5.2" });
		expect(resolveModelId("kimi-k2.6:cloud")).toEqual({ provider: "ollama", model: "kimi-k2.6" });
		expect(resolveModelId("gemma4:12b:cloud")).toEqual({ provider: "ollama", model: "gemma4:12b" });
		expect(resolveModelId("deepseek-v4-pro:cloud")).toEqual({ provider: "ollama", model: "deepseek-v4-pro" });
		expect(resolveModelId("deepseek-v4-flash:cloud")).toEqual({ provider: "ollama", model: "deepseek-v4-flash" });
	});

	it("passes through non-cloud IDs unchanged with ollama provider", () => {
		expect(resolveModelId("glm-5.2")).toEqual({ provider: "ollama", model: "glm-5.2" });
		expect(resolveModelId("llama3:8b")).toEqual({ provider: "ollama", model: "llama3:8b" });
		expect(resolveModelId("anthropic/claude-sonnet")).toEqual({
			provider: "ollama",
			model: "anthropic/claude-sonnet",
		});
	});

	it("only strips :cloud at the end, not in the middle", () => {
		expect(resolveModelId("some:cloud-model")).toEqual({ provider: "ollama", model: "some:cloud-model" });
		expect(resolveModelId("some:cloud-model:cloud")).toEqual({ provider: "ollama", model: "some:cloud-model" });
	});

	it("handles empty string", () => {
		expect(resolveModelId("")).toEqual({ provider: "ollama", model: "" });
	});

	it("handles model ID that is just :cloud", () => {
		expect(resolveModelId(":cloud")).toEqual({ provider: "ollama", model: "" });
		expect(resolveModelId("model:cloud:cloud")).toEqual({ provider: "ollama", model: "model:cloud" });
	});

	it("passes through openai/gpt-4 (no :cloud) unchanged with ollama provider", () => {
		expect(resolveModelId("openai/gpt-4")).toEqual({ provider: "ollama", model: "openai/gpt-4" });
	});

	it("strips trailing :cloud even after multiple colons", () => {
		expect(resolveModelId("model:cloud:cloud:cloud")).toEqual({ provider: "ollama", model: "model:cloud:cloud" });
	});

	it("handles :cloud suffix with provider/model style IDs", () => {
		expect(resolveModelId("openai/gpt-4:cloud")).toEqual({ provider: "ollama", model: "openai/gpt-4" });
	});

	it("handles numeric model names with :cloud suffix", () => {
		expect(resolveModelId("minimax-m3:cloud")).toEqual({ provider: "ollama", model: "minimax-m3" });
	});

	it("handles :cloud suffix when model has version tag", () => {
		expect(resolveModelId("qwen2.5:32b:cloud")).toEqual({ provider: "ollama", model: "qwen2.5:32b" });
	});

	it("passes through provider-prefixed IDs without :cloud", () => {
		expect(resolveModelId("anthropic/claude-sonnet-4-20250514")).toEqual({
			provider: "ollama",
			model: "anthropic/claude-sonnet-4-20250514",
		});
	});

	it("returns consistent provider for cloud vs non-cloud variants", () => {
		const cloudResult = resolveModelId("gpt-4:cloud");
		const nonCloudResult = resolveModelId("gpt-4");
		expect(cloudResult.provider).toBe("ollama");
		expect(nonCloudResult.provider).toBe("ollama");
		expect(cloudResult.model).toBe("gpt-4");
		expect(nonCloudResult.model).toBe("gpt-4");
	});
});
