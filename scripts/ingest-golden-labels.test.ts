import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { parseWorksheetSelections, runIngestGoldenLabels } from "./ingest-golden-labels.ts";

describe("ingest-golden-labels script", () => {
	it("reads the .local pool and worksheet and writes committed golden-labels.json with primary", async () => {
		const root = fs.mkdtempSync(path.join(os.tmpdir(), "semblr-ingest-golden-labels-test-"));
		const poolFile = path.join(root, "docs", "eval", "golden-pool.local.json");
		const worksheetFile = path.join(root, "docs", "eval", "golden-worksheet.local.md");
		const outFile = path.join(root, "docs", "eval", "golden-labels.json");
		const warnings: string[] = [];
		fs.mkdirSync(path.dirname(poolFile), { recursive: true });
		fs.writeFileSync(
			poolFile,
			JSON.stringify(
				{
					kind: "golden-pool",
					version: 1,
					corpus: "snapshot-a",
					source_pool: poolFile,
					queries: [
						{
							query: "query-a.json",
							prompt: "prompt a",
							difficulty: "hard",
							candidates: [
								{ file: "older-a.json", excerpt: "older a excerpt" },
								{ file: "older-b.json", excerpt: "older b excerpt" },
							],
						},
					],
				},
				null,
				2,
			),
		);
		fs.writeFileSync(
			worksheetFile,
			"# Golden worksheet\n\n## 1. query-a.json\n\n- [ ] older-a.json (primary)\n- [x] older-b.json\n",
		);

		await expect(
			runIngestGoldenLabels({
				args: ["--pool", poolFile, "--worksheet", worksheetFile, "--out", outFile],
				stdout: { log: () => {}, warn: (line: string) => warnings.push(line) },
			}),
		).resolves.toBe(0);
		const labels = JSON.parse(fs.readFileSync(outFile, "utf-8"));
		expect(labels).toEqual({
			kind: "golden-labels",
			version: 1,
			source_pool: poolFile,
			queries: [
				{
					query: "query-a.json",
					prompt: "prompt a",
					difficulty: "hard",
					primary: "older-a.json",
					labels: ["older-a.json", "older-b.json"],
				},
			],
		});
		expect(warnings).toEqual([
			"Primary label auto-selected for query-a.json: older-a.json was marked (primary) without [x]",
		]);
	});

	it("rejects multiple primary labels for one query", () => {
		expect(() =>
			parseWorksheetSelections(
				"# Golden worksheet\n\n## 1. query-a.json\n\n- [x] older-a.json (primary)\n- [x] older-b.json (primary)\n",
			),
		).toThrow("Worksheet has multiple primary labels for query-a.json");
	});
});
