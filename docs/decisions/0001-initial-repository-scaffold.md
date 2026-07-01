# 0001: Initial Repository Scaffold

Status: accepted  
Date: 2026-06-29

> Phase 1 autodiff direction is set by
> [0002: Phase 1 Enzyme-Backed `f64` Core](./0002-phase-1-enzyme-f64-core.md).

## Decision

Start Mercury as a single normal Cargo library crate wrapped in a reproducible
Nix workflow.

Do not start as a Cargo workspace. Do not add AD, linalg, symbolic, or native
dependencies until the first trait and API boundaries are clearer.

## Rationale

The repository standard in `pantheon` says to keep the crate boring first:

- one library crate
- committed `Cargo.lock`
- committed `flake.lock`
- Nix dev shells and checks
- common scripts
- CI templates

Mercury will become a deep math crate, but over-designing before the first
contracts are tested would make the foundational layer harder to reason about.

## Consequences

- Early code is minimal.
- Tooling can be verified before math architecture expands.
- Future AD/linalg/symbolic dependencies must enter through Mercury-owned
  abstractions.
