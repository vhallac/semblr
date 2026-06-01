import * as fs from "node:fs";

interface PositionScore {
	presentedCount: number;
	readCount: number;
	presentedHash: string | null;
}

export interface ChainReadStatsState {
	version: number;
	positionScores: PositionScore[];
	lastUpdated?: string;
	position5?: Pick<PositionScore, "presentedCount" | "readCount">;
}

export interface PresentedChainEntry {
	fileName: string;
}

export function createDefaultStatsState(nowIso = new Date().toISOString(), positions = 5): ChainReadStatsState {
	return {
		version: 2,
		positionScores: Array.from({ length: positions }, () => ({
			presentedCount: 0,
			readCount: 0,
			presentedHash: null,
		})),
		lastUpdated: nowIso,
	};
}

export function normalizeStatsState(raw: ChainReadStatsState, nowIso = new Date().toISOString()): ChainReadStatsState {
	if (raw.version === 1 && raw.position5) {
		const old = raw.position5;
		return {
			...raw,
			version: 2,
			positionScores: [
				{ presentedCount: 0, readCount: 0, presentedHash: null },
				{ presentedCount: 0, readCount: 0, presentedHash: null },
				{ presentedCount: 0, readCount: 0, presentedHash: null },
				{ presentedCount: 0, readCount: 0, presentedHash: null },
				{ presentedCount: old.presentedCount, readCount: old.readCount, presentedHash: null },
			],
			position5: undefined,
		};
	}
	if (!Array.isArray(raw.positionScores)) return createDefaultStatsState(nowIso);
	return raw;
}

export function loadStatsFile(statsPath: string, nowIso = new Date().toISOString()): ChainReadStatsState {
	try {
		if (fs.existsSync(statsPath)) {
			return normalizeStatsState(JSON.parse(fs.readFileSync(statsPath, "utf-8")), nowIso);
		}
	} catch {
		// Corrupt stats files are reset; stats must never break the extension.
	}
	return createDefaultStatsState(nowIso);
}

export function flushStatsFile(
	state: ChainReadStatsState,
	statsPath: string,
	statsDir: string,
	nowIso = new Date().toISOString(),
	pid = process.pid,
): void {
	state.lastUpdated = nowIso;
	const tmp = `${statsPath}.tmp.${pid}`;
	try {
		fs.mkdirSync(statsDir, { recursive: true });
		fs.writeFileSync(tmp, JSON.stringify(state, null, 2));
		fs.renameSync(tmp, statsPath);
	} catch {
		try {
			if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
		} catch {}
	}
}

export function recordRead(
	state: ChainReadStatsState,
	presentedHashes: Array<string | null>,
	hash: string,
	positions = 5,
): void {
	for (let i = 0; i < positions; i++) {
		if (presentedHashes[i] && presentedHashes[i] === hash) {
			state.positionScores[i].readCount++;
		}
	}
}

export function recordPresented(
	state: ChainReadStatsState,
	presentedHashes: Array<string | null>,
	chainEntries: PresentedChainEntry[],
	positions = 5,
): void {
	const reversed = [...chainEntries].reverse();
	for (let i = 0; i < positions; i++) {
		const entry = i < reversed.length ? reversed[i] : null;
		const hash = entry ? entry.fileName : null;
		presentedHashes[i] = hash;
		if (hash) state.positionScores[i].presentedCount++;
	}
}

export function formatChainReadStatsReport(state: ChainReadStatsState, positions = 5): string {
	const lines: string[] = ["Recent lookups to build context:"];
	const labels = ["immediate parent", "2nd", "3rd", "4th", "5th"];
	for (let i = 0; i < positions; i++) {
		const ps = state.positionScores[i];
		const pct = ps.presentedCount > 0 ? `${Math.round((ps.readCount / ps.presentedCount) * 100)}%` : "—";
		lines.push(`- ${labels[i]}: ${pct} (${ps.readCount}/${ps.presentedCount})`);
	}
	return lines.join("\n");
}
