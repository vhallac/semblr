import { describe, expect, it } from "vitest";
import { resolveModelId } from "./resolve-model-id.ts";

describe("resolveModelId", () => {
	// ── New {model}@{provider} syntax ──

	it("parses {model}@{provider} syntax", () => {
		expect(resolveModelId("glm-5.2@ollama-cloud")).toEqual({ provider: "ollama-cloud", model: "glm-5.2" });
		expect(resolveModelId("kimi-k2.6@ollama-cloud")).toEqual({ provider: "ollama-cloud", model: "kimi-k2.6" });
		expect(resolveModelId("anthropic/claude-sonnet@openai")).toEqual({
			provider: "openai",
			model: "anthropic/claude-sonnet",
		});
	});

	it("parses @ syntax with arbitrary providers", () => {
		expect(resolveModelId("gpt-4@openai")).toEqual({ provider: "openai", model: "gpt-4" });
		expect(resolveModelId("claude@anthropic")).toEqual({ provider: "anthropic", model: "claude" });
		expect(resolveModelId("gemma@local")).toEqual({ provider: "local", model: "gemma" });
	});

	it("handles @ at start (empty model)", () => {
		expect(resolveModelId("@provider")).toEqual({ provider: "provider", model: "" });
	});

	it("handles @ at end (empty provider)", () => {
		expect(resolveModelId("model@")).toEqual({ provider: "", model: "model" });
	});

	it("handles only @", () => {
		expect(resolveModelId("@")).toEqual({ provider: "", model: "" });
	});

	it("splits on first @ only", () => {
		expect(resolveModelId("model@provider@extra")).toEqual({ provider: "provider@extra", model: "model" });
	});

	// ── Backward-compat: :cloud suffix ──

	it("strips :cloud suffix and returns ollama provider", () => {
		expect(resolveModelId("glm-5.2:cloud")).toEqual({ provider: "ollama-cloud", model: "glm-5.2" });
		expect(resolveModelId("kimi-k2.6:cloud")).toEqual({ provider: "ollama-cloud", model: "kimi-k2.6" });
		expect(resolveModelId("gemma4:31b:cloud")).toEqual({ provider: "ollama-cloud", model: "gemma4:31b" });
		expect(resolveModelId("deepseek-v4-pro:cloud")).toEqual({ provider: "ollama-cloud", model: "deepseek-v4-pro" });
		expect(resolveModelId("deepseek-v4-flash:cloud")).toEqual({
			provider: "ollama-cloud",
			model: "deepseek-v4-flash",
		});
	});

	it("passes through non-cloud IDs unchanged with ollama provider", () => {
		expect(resolveModelId("glm-5.2")).toEqual({ provider: "ollama-cloud", model: "glm-5.2" });
		expect(resolveModelId("llama3:8b")).toEqual({ provider: "ollama-cloud", model: "llama3:8b" });
		expect(resolveModelId("anthropic/claude-sonnet")).toEqual({
			provider: "ollama-cloud",
			model: "anthropic/claude-sonnet",
		});
	});

	it("only strips :cloud at the end, not in the middle", () => {
		expect(resolveModelId("some:cloud-model")).toEqual({ provider: "ollama-cloud", model: "some:cloud-model" });
		expect(resolveModelId("some:cloud-model:cloud")).toEqual({ provider: "ollama-cloud", model: "some:cloud-model" });
	});

	it("handles empty string", () => {
		expect(resolveModelId("")).toEqual({ provider: "ollama-cloud", model: "" });
	});

	it("handles model ID that is just :cloud", () => {
		expect(resolveModelId(":cloud")).toEqual({ provider: "ollama-cloud", model: "" });
		expect(resolveModelId("model:cloud:cloud")).toEqual({ provider: "ollama-cloud", model: "model:cloud" });
	});

	it("passes through openai/gpt-4 (no :cloud) unchanged with ollama provider", () => {
		expect(resolveModelId("openai/gpt-4")).toEqual({ provider: "ollama-cloud", model: "openai/gpt-4" });
	});

	it("strips trailing :cloud even after multiple colons", () => {
		expect(resolveModelId("model:cloud:cloud:cloud")).toEqual({
			provider: "ollama-cloud",
			model: "model:cloud:cloud",
		});
	});

	it("handles :cloud suffix with provider/model style IDs", () => {
		expect(resolveModelId("openai/gpt-4:cloud")).toEqual({ provider: "ollama-cloud", model: "openai/gpt-4" });
	});

	it("handles numeric model names with :cloud suffix", () => {
		expect(resolveModelId("minimax-m3:cloud")).toEqual({ provider: "ollama-cloud", model: "minimax-m3" });
	});

	it("handles :cloud suffix when model has version tag", () => {
		expect(resolveModelId("qwen2.5:32b:cloud")).toEqual({ provider: "ollama-cloud", model: "qwen2.5:32b" });
	});

	it("passes through provider-prefixed IDs without :cloud", () => {
		expect(resolveModelId("anthropic/claude-sonnet-4-20250514")).toEqual({
			provider: "ollama-cloud",
			model: "anthropic/claude-sonnet-4-20250514",
		});
	});

	it("returns consistent provider for cloud vs non-cloud variants", () => {
		const cloudResult = resolveModelId("gpt-4:cloud");
		const nonCloudResult = resolveModelId("gpt-4");
		expect(cloudResult.provider).toBe("ollama-cloud");
		expect(nonCloudResult.provider).toBe("ollama-cloud");
		expect(cloudResult.model).toBe("gpt-4");
		expect(nonCloudResult.model).toBe("gpt-4");
	});

	// ── @ syntax takes priority over :cloud ──

	it("@ syntax takes priority over :cloud when both are present", () => {
		expect(resolveModelId("model@provider:cloud")).toEqual({ provider: "provider:cloud", model: "model" });
		expect(resolveModelId("model:cloud@provider")).toEqual({ provider: "provider", model: "model:cloud" });
	});
});
