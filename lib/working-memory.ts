export interface MiniMemSlot {
	id: number;
	summary: string;
	content: string;
	createdAt: number;
	sourceRound?: string;
}

export interface MiniMemStore {
	slots: MiniMemSlot[];
}

export function createMiniMemStore(): MiniMemStore {
	return { slots: [] };
}

/**
 * Add a slot to the store. IDs are monotonic (highest seen + 1, starting from 1).
 * When adding beyond the max capacity (7), the oldest slot (lowest id) is silently evicted.
 * Returns the assigned id.
 */
export function addSlot(store: MiniMemStore, summary: string, content: string, sourceRound?: string): number {
	const maxId = store.slots.reduce((max, s) => Math.max(max, s.id), 0);
	const id = maxId + 1;

	const slot: MiniMemSlot = {
		id,
		summary,
		content,
		createdAt: Date.now(),
		sourceRound,
	};

	store.slots.push(slot);

	// Evict oldest (lowest id) if over capacity
	if (store.slots.length > 7) {
		evictOldest(store);
	}

	return id;
}

/** Evict the slot with the lowest id. Assumes store has at least one slot. */
function evictOldest(store: MiniMemStore): void {
	let minIdx = 0;
	for (let i = 1; i < store.slots.length; i++) {
		if (store.slots[i].id < store.slots[minIdx].id) {
			minIdx = i;
		}
	}
	store.slots.splice(minIdx, 1);
}

export function getSlot(store: MiniMemStore, id: number): MiniMemSlot | null {
	const slot = store.slots.find((s) => s.id === id);
	return slot ?? null;
}

export function updateSlot(
	store: MiniMemStore,
	id: number,
	summary: string,
	content: string,
	sourceRound?: string,
): MiniMemSlot | null {
	const idx = store.slots.findIndex((s) => s.id === id);
	if (idx === -1) return null;
	store.slots[idx].summary = summary;
	store.slots[idx].content = content;
	if (sourceRound !== undefined) {
		store.slots[idx].sourceRound = sourceRound;
	}
	return store.slots[idx];
}

export function deleteSlot(store: MiniMemStore, id: number): boolean {
	const idx = store.slots.findIndex((s) => s.id === id);
	if (idx === -1) return false;
	store.slots.splice(idx, 1);
	return true;
}

export function getAndDeleteSlot(store: MiniMemStore, id: number): MiniMemSlot | null {
	const slot = getSlot(store, id);
	if (!slot) return null;
	deleteSlot(store, id);
	return slot;
}
