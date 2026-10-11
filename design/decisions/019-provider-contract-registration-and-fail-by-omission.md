# 019 — Provider contract: registration validation and fail-by-omission in the base contract

**Status:** accepted

## Context
The context-provider API contract (#111) could have been a bare types module, deferring all validation and error handling to the timeout/oversize ceiling work (#112). Without registration validation, duplicate identities and version mismatches would surface as silent context corruption; without a fail-by-omission policy, a throwing provider would abort the whole context build.

## Decision
Registration validation and minimal fail-by-omission live in the contract module (`lib/provider-contract.ts`), not #112:

1. Registration rejects duplicate identities and invalid identities; providers declaring an unsupported `apiVersion` are skipped (not an error).
2. Fail-by-omission: a provider returning `null` is not a failure — it simply renders nothing and its dependents still render. A provider that throws is dropped with a diagnostic, and its dependents are skipped (cascade), but the build continues.
3. `dependsOn` is validated at registration: unknown deps, self-deps, and cycles are rejected; a defensive cycle guard (`reaches()`) covers direct registry-map mutation.

## Consequences
- The contract is testable and safe in isolation; #112 only adds ceiling/timeout on top.
- Context builds degrade gracefully — a broken provider costs its own section, not the whole context.
