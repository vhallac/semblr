import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		include: ["lib/**/*.test.ts", "scripts/**/*.test.ts"],
		coverage: {
			provider: "v8",
			include: ["lib/**/*.ts", "scripts/**/*.ts"],
			exclude: ["**/*.test.ts"],
			thresholds: {
				branches: 80,
				functions: 80,
				lines: 80,
				statements: 80,
			},
		},
	},
});
