import * as fs from "node:fs";

/** Minimal structural fs surface used by atomicWriteFileSync. */
export interface AtomicWriteFs {
	writeFileSync(path: string, data: string): void;
	renameSync(from: string, to: string): void;
}

/**
 * Atomic file write (decision-015): write to a sibling tmp file
 * `<target>.tmp.<pid>`, then renameSync it over the target. A crash or
 * throw during write leaves the target byte-identical; only a successful
 * rename publishes the new content. Used for round files — the durable
 * store — where a truncating in-place write could destroy data that has
 * no session JSONL left to recover from.
 *
 * The fs implementation is injectable so tests can fail the write and
 * assert the target is untouched.
 */
export function atomicWriteFileSync(targetPath: string, data: string, fsImpl: AtomicWriteFs = fs): void {
	const tmpPath = `${targetPath}.tmp.${process.pid}`;
	try {
		fsImpl.writeFileSync(tmpPath, data);
		fsImpl.renameSync(tmpPath, targetPath);
	} catch (err) {
		try {
			fs.unlinkSync(tmpPath);
		} catch {
			// tmp may never have been created; nothing to clean up
		}
		throw err;
	}
}
