export interface TextContentBlock {
	type: string;
	text?: string;
}

export function extractText(content: readonly TextContentBlock[]): string {
	return content
		.filter((c) => (c.type === "text" || c.type === "thinking") && c.text)
		.map((c) => (c.type === "thinking" ? `[thinking] ${c.text ?? ""} [/thinking]` : (c.text ?? "")))
		.join(" ");
}
