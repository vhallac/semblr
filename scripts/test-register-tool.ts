/**
 * Quick test: load the extension and verify tool registration.
 * Run with: pi -e scripts/test-register-tool.ts
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";

export default async function (pi: ExtensionAPI) {
	// Register a minimal tool
	pi.registerTool({
		name: "search_interactions",
		label: "Search Interactions",
		description: "Test tool - search interactions",
		parameters: Type.Object({
			query: Type.String({ description: "Search query" }),
		}),
		async execute(toolCallId, params, _signal, _onUpdate, _ctx) {
			const q = (params as { query: string }).query;
			return {
				content: [{ type: "text", text: `Would search for: ${q}` }],
				details: {},
			};
		},
	});

	// Check active and all tools
	const all = pi.getAllTools();
	const active = pi.getActiveTools();

	const searchTool = all.find((t: any) => t.name === "search_interactions");
	const searchActive = active.find((t: any) => t.name === "search_interactions");

	console.log("=== getAllTools() ===");
	console.log(
		JSON.stringify(
			all.map((t: any) => t.name),
			null,
			2,
		),
	);
	console.log("\nsearch_interactions in all:", !!searchTool);
	console.log("\n=== getActiveTools() ===");
	console.log(
		JSON.stringify(
			active.map((t: any) => t.name),
			null,
			2,
		),
	);
	console.log("\nsearch_interactions in active:", !!searchActive);

	if (searchTool) {
		console.log("\nsearch_interactions sourceInfo:", JSON.stringify(searchTool.sourceInfo, null, 2));
	}
}
