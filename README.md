# Mercury

[![CI](https://github.com/pantheon-rs/mercury/actions/workflows/ci.yml/badge.svg)](https://github.com/pantheon-rs/mercury/actions/workflows/ci.yml) [![Format](https://github.com/pantheon-rs/mercury/actions/workflows/format.yml/badge.svg)](https://github.com/pantheon-rs/mercury/actions/workflows/format.yml) [![Docs](https://github.com/pantheon-rs/mercury/actions/workflows/docs.yml/badge.svg)](https://github.com/pantheon-rs/mercury/actions/workflows/docs.yml) [![Security](https://github.com/pantheon-rs/mercury/actions/workflows/security.yml/badge.svg)](https://github.com/pantheon-rs/mercury/actions/workflows/security.yml) [![codecov](https://codecov.io/gh/pantheon-rs/mercury/graph/badge.svg)](https://codecov.io/gh/pantheon-rs/mercury)

`mercury` is the differentiable math substrate for `pantheon-rs`.

The Phase 1 direction is an Enzyme-first autodiff crate for plain `f64` model
code. This is the Metis idea reduced to the part that matters first: model code
is written once as ordinary numeric Rust, and Mercury owns the derivative entry
points, shadow-buffer plumbing, and validation surface.

Phase 1 owns:

- plain `f64` model-kernel conventions
- Enzyme-backed dense derivative evaluators
- finite-difference and analytic derivative checks
- a conservative AD-safe kernel subset
- room for sparse derivative callbacks later

It does not start with a generic scalar trait, a symbolic graph engine, a solver
stack, or a full linear algebra facade. Sparsity, graph coloring, and
optimization-facing callbacks are designed when real problem scale demands
them, without changing ordinary model code into a symbolic DSL.

Phase 2 adds the owned core types and the first owned derivative rule:
kernel-safe `SVector`/`SMatrix`/`Quaternion` (proven against Enzyme per
type), host-side `Vector`/`Matrix`, and linear solve where small systems
differentiate through `solve_fixed_unchecked` (the kernel-safe infallible
variant; `solve_fixed` is the `Result`-returning host wrapper) while
problem-scale systems use the LU primitive with the adjoint rule. See
`docs/decisions/0003-differentiable-primitives-identity.md`.

## Source Layout

```text
src/
  lib.rs
  objective.rs     # scalar_objective! macro (Enzyme reverse entry points)
  validation.rs    # finite-difference oracles
  core/            # SVector, SMatrix (kernel-safe) + Vector, Matrix (host-side)
  geometry/        # Quaternion
  linalg/          # solve_fixed, LU solve + adjoint rule (solve_vjp/solve_jvp)
tests/             # one suite per module, three-legged test law
examples/
  solve_gradient.rs  # one gradient, three ways (fd / enzyme / adjoint)
```

The root crate is the Enzyme-backed Mercury library. `src/objective.rs` contains
the initial scalar-objective API, and `tests/objective.rs` proves that API
against Enzyme, finite differences, and analytic gradients.

The first user-facing API is:

```rust
mercury::scalar_objective! {
    pub mod rosenbrock(x) {
        let mut acc = 0.0;
        for i in 0..x.len() - 1 {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            acc += 100.0 * a * a + b * b;
        }
        acc
    }
}

let result = rosenbrock::value_and_gradient(&[0.5; 6]);
```

## Development

```text
nix develop
./scripts/build.sh
./scripts/test.sh
./scripts/ci.sh
```

## Documentation

- [Architecture](docs/architecture.md)
- [Phase 1 Enzyme-backed `f64` decision](docs/decisions/0002-phase-1-enzyme-f64-core.md)
- [Phase 1 gradient validation implementation plan](docs/implementation-plans/phase-1-gradient-validation.md)
- [Phase 2 differentiable primitives identity decision](docs/decisions/0003-differentiable-primitives-identity.md)
- [Phase 2 core types + linalg implementation plan](docs/implementation-plans/phase-2-core-types-and-linalg.md)
- [Decisions](docs/decisions/)
