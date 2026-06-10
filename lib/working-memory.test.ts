import { describe, expect, it } from "vitest";
import {
	addSlot,
	createMiniMemStore,
	deleteSlot,
	formatMiniMemSlot,
	getAndDeleteSlot,
	getSlot,
	updateSlot,
} from "./working-memory.ts";

describe("createMiniMemStore", () => {
	it("creates an empty store", () => {
		const store = createMiniMemStore();
		expect(store.slots).toEqual([]);
	});
});

describe("addSlot", () => {
	it("assigns monotonic ids starting from 1", () => {
		const store = createMiniMemStore();
		const id1 = addSlot(store, "First", "Content 1");
		const id2 = addSlot(store, "Second", "Content 2");
		const id3 = addSlot(store, "Third", "Content 3");
		expect(id1).toBe(1);
		expect(id2).toBe(2);
		expect(id3).toBe(3);
		expect(store.slots.length).toBe(3);
	});

	it("silently evicts the oldest (lowest id) when adding an 8th slot", () => {
		const store = createMiniMemStore();
		const ids: number[] = [];
		for (let i = 0; i < 8; i++) {
			ids.push(addSlot(store, `Slot ${i + 1}`, `Content ${i + 1}`));
		}
		// 8th slot added, oldest (id:1) should be evicted
		expect(store.slots.length).toBe(7);
		const slotIds = store.slots.map((s) => s.id);
		expect(slotIds).not.toContain(1);
		expect(slotIds).toContain(ids[7]); // the 8th slot is present
	});

	it("keeps all slots when exactly at capacity (7)", () => {
		const store = createMiniMemStore();
		for (let i = 0; i < 7; i++) {
			addSlot(store, `Slot ${i + 1}`, `Content ${i + 1}`);
		}
		expect(store.slots.length).toBe(7);
	});

	it("evicts the true lowest id even when ids are non-sequential after deletions", () => {
		const store = createMiniMemStore();
		addSlot(store, "A", "a"); // id 1
		addSlot(store, "B", "b"); // id 2
		addSlot(store, "C", "c"); // id 3
		addSlot(store, "D", "d"); // id 4
		addSlot(store, "E", "e"); // id 5
		addSlot(store, "F", "f"); // id 6
		addSlot(store, "G", "g"); // id 7
		// Delete id 1
		deleteSlot(store, 1);
		expect(store.slots.length).toBe(6);
		// Add another slot — should NOT evict anything yet (only 7 total)
		addSlot(store, "H", "h"); // id 8
		expect(store.slots.length).toBe(7);
		// Add one more — should evict id 2 (the new lowest)
		addSlot(store, "I", "i"); // id 9
		expect(store.slots.length).toBe(7);
		const slotIds = store.slots.map((s) => s.id).sort((a, b) => a - b);
		expect(slotIds).toEqual([3, 4, 5, 6, 7, 8, 9]);
	});
});

describe("getSlot", () => {
	it("returns the slot when found", () => {
		const store = createMiniMemStore();
		addSlot(store, "Test", "Test content");
		const slot = getSlot(store, 1);
		expect(slot).not.toBeNull();
		expect(slot?.summary).toBe("Test");
		expect(slot?.content).toBe("Test content");
	});

	it("returns null for a non-existent id", () => {
		const store = createMiniMemStore();
		expect(getSlot(store, 99)).toBeNull();
	});

	it("returns null for an evicted slot", () => {
		const store = createMiniMemStore();
		for (let i = 0; i < 8; i++) {
			addSlot(store, `Slot ${i + 1}`, `Content ${i + 1}`);
		}
		// Slot 1 was evicted
		expect(getSlot(store, 1)).toBeNull();
	});
});

describe("updateSlot", () => {
	it("overwrites summary, content, and sourceRound", () => {
		const store = createMiniMemStore();
		addSlot(store, "Original", "Original content", "round1.json");
		const updated = updateSlot(store, 1, "Updated", "Updated content", "round2.json");
		expect(updated).not.toBeNull();
		expect(updated?.summary).toBe("Updated");
		expect(updated?.content).toBe("Updated content");
		expect(updated?.sourceRound).toBe("round2.json");
	});

	it("returns null for non-existent id", () => {
		const store = createMiniMemStore();
		expect(updateSlot(store, 42, "X", "Y")).toBeNull();
	});

	it("does not overwrite sourceRound when not provided", () => {
		const store = createMiniMemStore();
		addSlot(store, "Original", "Content", "round1.json");
		const updated = updateSlot(store, 1, "Updated", "Updated content");
		expect(updated).not.toBeNull();
		expect(updated?.sourceRound).toBe("round1.json");
	});
});

describe("deleteSlot", () => {
	it("returns true and removes the slot when found", () => {
		const store = createMiniMemStore();
		addSlot(store, "To delete", "Content");
		expect(store.slots.length).toBe(1);
		expect(deleteSlot(store, 1)).toBe(true);
		expect(store.slots.length).toBe(0);
	});

	it("returns false when id is not found", () => {
		const store = createMiniMemStore();
		expect(deleteSlot(store, 99)).toBe(false);
		expect(store.slots.length).toBe(0);
	});
});

describe("getAndDeleteSlot", () => {
	it("returns the slot and removes it", () => {
		const store = createMiniMemStore();
		addSlot(store, "Disposable", "Use once");
		const slot = getAndDeleteSlot(store, 1);
		expect(slot).not.toBeNull();
		expect(slot?.summary).toBe("Disposable");
		expect(store.slots.length).toBe(0);
	});

	it("returns null when id is not found", () => {
		const store = createMiniMemStore();
		expect(getAndDeleteSlot(store, 99)).toBeNull();
	});
});

describe("formatMiniMemSlot", () => {
	it("formats a slot with sourceRound", () => {
		const store = createMiniMemStore();
		addSlot(store, "Plan A", "Step 1: do X", "round-1.json");
		const slot = getSlot(store, 1);
		if (!slot) throw new Error("expected slot");
		expect(formatMiniMemSlot(slot)).toBe("[id: 1] Plan A\nSource round: round-1.json\n\nStep 1: do X");
	});

	it("formats a slot without sourceRound", () => {
		const store = createMiniMemStore();
		addSlot(store, "Plan B", "Step 2");
		const slot = getSlot(store, 1);
		if (!slot) throw new Error("expected slot");
		expect(formatMiniMemSlot(slot)).toBe("[id: 1] Plan B\n\nStep 2");
	});
});

describe("preservation after eviction", () => {
	it("preserved slots retain their ids and content after eviction of other slots", () => {
		const store = createMiniMemStore();
		addSlot(store, "First", "Content 1");
		addSlot(store, "Second", "Content 2");
		addSlot(store, "Third", "Content 3");
		addSlot(store, "Fourth", "Content 4");
		addSlot(store, "Fifth", "Content 5");
		addSlot(store, "Sixth", "Content 6");
		addSlot(store, "Seventh", "Content 7");

		// Add 8th to evict id 1
		addSlot(store, "Eighth", "Content 8");

		// Check preserved slots keep their original content
		const slot2 = getSlot(store, 2);
		expect(slot2?.summary).toBe("Second");
		expect(slot2?.content).toBe("Content 2");

		const slot7 = getSlot(store, 7);
		expect(slot7?.summary).toBe("Seventh");
		expect(slot7?.content).toBe("Content 7");

		const slot8 = getSlot(store, 8);
		expect(slot8?.summary).toBe("Eighth");
	});
});
