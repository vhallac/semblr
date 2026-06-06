import { cosineSimilarity } from "./vector.ts";

export interface SemanticGroup<T> {
	centroid: number[];
	rounds: T[];
}

export function parseGroupThreshold(value: string | undefined, fallback = 0.77): number {
	if (value !== undefined) {
		const parsed = Number.parseFloat(value);
		if (!Number.isNaN(parsed) && parsed >= 0 && parsed <= 1) return parsed;
	}
	return fallback;
}

export function formatGroupStats<T>(groups: Array<SemanticGroup<T>>, threshold: number): string {
	const thr = Math.round(threshold * 100);
	const groupCount = groups.length;
	const topicCounts = groups.map((g) => g.rounds.length).join(",");
	return `THR: ${thr}%; #groups: ${groupCount}, #topics: {${topicCounts}}`;
}

export function assignToGroup<T>(
	groups: Array<SemanticGroup<T>>,
	roundEntry: T,
	vec: number[] | null,
	threshold: number,
	/** When set, skip semantic matching and force-assign to this group index instead. */
	forceGroupIdx?: number | null,
): number {
	// Auto-group: if a previous round was flagged needsFollowup, this round
	// belongs to the same topic — skip semantic matching.
	if (forceGroupIdx !== undefined && forceGroupIdx !== null && forceGroupIdx < groups.length) {
		const group = groups[forceGroupIdx];
		group.rounds.push(roundEntry);
		if (vec) {
			const n = group.rounds.length;
			group.centroid = group.centroid.map((c, i) => c + (vec[i] - c) / n);
		}
		return forceGroupIdx;
	}

	// Null vector: create a new singleton group (short prompts with sparse info
	// produce noisy embeddings — better to not match against existing groups)
	if (!vec) {
		groups.push({
			centroid: [],
			rounds: [roundEntry],
		});
		return groups.length - 1;
	}

	let bestIdx = -1;
	let bestSim = 0;
	for (let i = 0; i < groups.length; i++) {
		const sim = cosineSimilarity(vec, groups[i].centroid);
		if (sim >= threshold && sim > bestSim) {
			bestSim = sim;
			bestIdx = i;
		}
	}
	if (bestIdx >= 0) {
		const group = groups[bestIdx];
		group.rounds.push(roundEntry);
		const n = group.rounds.length;
		group.centroid = group.centroid.map((c, i) => c + (vec[i] - c) / n);
		return bestIdx;
	}
	groups.push({
		centroid: [...vec],
		rounds: [roundEntry],
	});
	return groups.length - 1;
}
