export interface TextContentBlock {
	type: string;
	text?: string;
}

export function extractText(content: readonly TextContentBlock[]): string {
	return content
		.filter((c) => c.type === "text" && c.text)
		.map((c) => c.text ?? "")
		.join(" ");
}
