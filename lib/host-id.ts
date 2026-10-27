import * as os from "node:os";

/**
 * Host identity (issue #149, OI-6).
 *
 * The index shard of a host is named after its host id. The id is resolved
 * in this order:
 *
 * 1. a configured id (env var `SEMBLR_HOST_ID`),
 * 2. `os.hostname()`.
 *
 * Nothing is persisted: the id is recomputed on every run, so it always
 * tracks the machine's current identity.
 */

/**
 * A usable host id: non-empty, not a path traversal, and free of path
 * separators and control characters (the id becomes a filename component).
 * Returns `undefined` for an unusable value.
 */
export function sanitizeHostId(value: string | undefined | null): string | undefined {
	if (value === undefined || value === null) return undefined;
	const trimmed = value.trim();
	if (!trimmed || trimmed === "." || trimmed === "..") return undefined;
	// biome-ignore lint/suspicious/noControlCharactersInRegex: control characters are rejected on purpose
	if (/[/\\\u0000-\u001f]/.test(trimmed)) return undefined;
	return trimmed;
}

export interface ResolveHostIdDeps {
	/** Host id configured through the `SEMBLR_HOST_ID` env var. */
	configuredHostId?: string;
	/** Machine-name fallback; defaults to `os.hostname()`. */
	hostname?: string;
	/** Warning sink for a rejected configured id. */
	warn?: (message: string) => void;
}

export interface ResolvedHostId {
	/** The resolved host id — the shard name component. */
	hostId: string;
	/** Where the id came from. */
	source: "configured" | "hostname";
}

/**
 * Resolve the host id: the configured `SEMBLR_HOST_ID` value when usable,
 * otherwise the machine name. Nothing is written to the filesystem.
 */
export function resolveHostId(deps: ResolveHostIdDeps): ResolvedHostId {
	const warn = deps.warn ?? console.warn;

	const configured = sanitizeHostId(deps.configuredHostId);
	if (configured === undefined && deps.configuredHostId !== undefined) {
		warn(`Invalid Semblr host id ${JSON.stringify(deps.configuredHostId)} — using the machine name instead.`);
	}
	if (configured !== undefined) return { hostId: configured, source: "configured" };

	return { hostId: deps.hostname ?? os.hostname(), source: "hostname" };
}
