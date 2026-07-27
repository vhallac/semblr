import fs from "node:fs";
import { describe, expect, it } from "vitest";

describe("golden evaluation recipes", () => {
	const justfile = fs.readFileSync(new URL("../justfile", import.meta.url), "utf8");

	it.each([
		["eval-golden", ""],
		["eval-golden-similarity", "--mode similarity"],
		["eval-golden-tool", "--mode tool"],
	])("provides %s with the expected mode filter", (recipe, modeFlag) => {
		const body = justfile.match(new RegExp(`^${recipe} corpus out=.*:\\n((?:    .*\\n)+)`, "m"))?.[1];

		expect(body).toContain("--golden docs/eval/golden-labels.local.json");
		expect(body).toContain(`--sessions {{corpus}}/sessions`);
		if (modeFlag) {
			expect(body).toContain(modeFlag);
		} else {
			expect(body).not.toContain("--mode");
		}
	});
});
