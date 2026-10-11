# 021 — Two-phase provider render with a frozen SectionsView

**Status:** accepted

## Context
Providers may depend on sections produced by other providers (`dependsOn`). Rendering order must be deterministic, and dependents must not be able to mutate what they read. The initially natural choice — a `Map` on the render context — is mutable and order-bearing.

## Decision
`renderProviders()` walks in two phases: independents first (priority order, ties by registration order), then dependents. Within phase 2, providers are ordered **topologically over `dependsOn` edges** (Kahn's algorithm, priority as tie-break among ready nodes, stable sort keeps registration order) — so a dependent always renders after its declared dependencies, even under priority inversion. (As of 022, this ordering operates over the resolved edge set — optional-absent edges dropped, statically-disabled providers and their transitive dependents removed.) Dependent providers receive a frozen `SectionsView` (a plain record, not a `Map` — genuinely read-only) as an additive field on `BuildContext`. The frozen-field-list contract permits additive fields; existing providers are unaffected.

## Consequences
- Deterministic render order independent of provider implementation details.
- Dependents read completed sections only, and cannot mutate them.
- `null`-returning providers still allow their dependents to render (null is not a failure; the dependent observes `undefined` for the absent section); only thrown/dropped providers cascade-skip dependents.
- Phase-1 providers see the sections built so far (empty only for the first phase-1 provider); render order is the priority-ordered two-phase walk, not registration order.
