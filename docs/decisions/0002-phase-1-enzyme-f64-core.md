# 0002: Phase 1 Enzyme-Backed `f64` Core

Status: accepted  
Date: 2026-06-30

## Decision

Mercury Phase 1 uses plain `f64` model kernels differentiated by Rust nightly
`std::autodiff` / Enzyme.

This replaces the early generic scalar / symbolic trace direction as the core
implementation plan. The core does not start with `ad_trait`, a dual-number
scalar, a symbolic expression graph, sparse graph coloring, or solver bindings.

Mercury still owns the public derivative contract. Enzyme activity markers,
shadow buffers, generated derivative functions, toolchain quirks, and validation
tests are internal Mercury responsibilities.

## Rationale

Metis proved the value of one model feeding simulation and derivatives, but its
derivative path was gated through symbolic graph construction. Enzyme changes the
best first move: derivatives can be compiled from the same `f64` code that runs
the simulation path.

This should make Mercury simpler than Metis in the core:

- less generic scalar plumbing in author code
- no symbolic graph required for dense derivatives
- normal Rust control flow in model kernels
- faster reverse-mode gradients on the benchmarked kernels
- a smaller first implementation surface

## Consequences

- The build depends on a pinned nightly Rust toolchain with the matching Enzyme
  plugin.
- Differentiated kernels are isolated so fat-LTO and nightly constraints do not
  dominate the whole workspace.
- Mercury needs an AD-safe kernel subset and tests for the LLVM IR patterns
  Enzyme accepts.
- Value-dependent branches are allowed but remain piecewise operations with
  optimizer-visible derivative policy.
- The final macro or registration syntax is not promised until it composes with
  `std::autodiff` in a compile test.
- Existing scaffold APIs such as `Scalar` and `where_` should not be expanded
  into the Phase 1 core contract.

## Phase 1 Non-Goals

- symbolic expression IR
- generic scalar API for all model code
- sparse Jacobian/Hessian coloring
- nonlinear programming solver integration
- broad `nalgebra`/`faer` facade
- C/C++ FFI in the core

## Exit Criteria

- A plain `f64` model kernel runs directly and produces an Enzyme reverse-mode
  gradient.
- A forward derivative or JVP path is proven.
- Derivatives cross-check against finite differences and analytic derivatives
  where available.
- The chosen model registration syntax hides raw Enzyme activity markers and
  derivative buffers from normal model code.
- Unsupported differentiated-code patterns are documented with tests or examples.
