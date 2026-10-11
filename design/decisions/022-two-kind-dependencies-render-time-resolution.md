# 022 — Two-kind dependencies with render-time resolution

**Status:** accepted

## Context
`dependsOn` is currently `string[]` with no way to express whether a dependency is required. Resolution happens at registration time, so a provider must be registered before its dependents, registration order becomes load-bearing, and any absent dependency is rejected outright. PR !144 finding F3 flagged this: dependents often want to consume a section that may legitimately be absent (the provider did not render), and requiring registration-order coupling is an unnecessary constraint.

## Decision
`dependsOn` becomes `ReadonlyArray<{ name: string; required: boolean }>` — two kinds of edges:

1. **Required edges** (`required: true`): if the named provider is registered but did not render (dropped after throwing), the dependent is statically disabled with a TUI diagnostic (`provider X disabled: required provider Y not registered`) and its dependents are cascade-disabled transitively. If the named provider is not registered at all, the dependent is likewise statically disabled — resolution happens at render time over the fully-registered set, not at registration.
2. **Optional edges** (`required: false`): if the section is absent at render time, the edge is dropped silently and the dependent still renders; it observes `undefined` for that section.

Resolution is deferred to render time:

- Registration-time existence validation of `dependsOn` entries is **dropped**. Registration order stops mattering for dependency resolution.
- Cycle detection moves onto structured edges and is retained as a defensive check (over the structured `dependsOn` graph, including direct registry-map mutation reaches). Cycles are errors for both edge kinds — neither required nor optional cycles can be satisfied.
- The static disable check (required-absent detection) runs once per build, not per render call.

This record supersedes the `dependsOn` validation clause of 019 point 3 (existence checks at registration); 019's fail-by-omission policy and remaining registration validation (duplicate/invalid identities, unsupported `apiVersion`) are unchanged. 021's phase-2 topological ordering is retained but now operates over the **resolved** edge set (after optional-absent drops and static-disable cascades), not the raw declared edges.

## Consequences
- Registration order no longer couples providers; declaration order is free.
- Absent optional dependencies are a normal, silent condition; absent required dependencies are a visible, statically-determined disable with cascade — a broken dependency costs the dependent's section (and its transitive dependents), not the whole context build.
- The rendered order remains deterministic: phase-1 independents first, then phase-2 dependents topologically over resolved edges.
- Provider authors gain explicit control over soft vs hard dependencies without duplicating providers or guarding reads.
