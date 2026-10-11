# 020 — Provider contract public surface: package-entry named exports

**Status:** accepted

## Context
The #111 contract needed a definition of its "public surface". The assumption was that it means named exports from the package entry (`src/semblr.ts`), not the individual `lib/` modules, matching the repo convention that only the entry lives in `src/`.

## Decision
The contract's public surface is the set of named exports from the package entry (`src/semblr.ts`), which re-export the contract types and functions from `lib/provider-contract.ts`.

## Consequences
- Consumers import the contract only via the package entry; `lib/` internals remain free to move.
- New public API must be added to the entry's export list, which keeps the surface explicit and reviewable.
