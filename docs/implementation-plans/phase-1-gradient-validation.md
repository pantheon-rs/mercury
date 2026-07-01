# Phase 1 Gradient Validation Slice

Status: planned  
Date: 2026-06-30

## Goal

Build the smallest useful Mercury implementation slice:

```text
plain f64 model kernel
  -> Enzyme reverse-mode gradient
  -> finite-difference and analytic validation
  -> clear diagnostics
```

This is the first production step toward the Phase 1 Enzyme-backed `f64` core.
It should prove the toolchain, kernel shape, derivative ABI, and validation
surface before Mercury grows math, linear algebra, sparsity, or optimization
layers.

## Source References

- Metis differentiability harness:
  `/home/tanged/sources/metis/include/metis/utils/DiffTestHarness.hpp`
- Metis finite-difference utilities:
  `/home/tanged/sources/metis/include/metis/math/FiniteDifference.hpp`
- Enzyme toolchain and smoke test:
  `/home/tanged/sources/metis-ad-spike/std_autodiff/`
- Enzyme vs `ad_trait` benchmark:
  `/home/tanged/sources/metis-ad-spike/bench_enzyme_vs_adtrait/`

Use these as behavior and validation references. Do not port Metis's symbolic
type system, CasADi function wrapper, generic scalar dispatch, or solver stack.

## Non-Goals

- no symbolic expression graph
- no generic AD scalar
- no broad math facade
- no `nalgebra` or `faer` dependency yet
- no optimization problem interface
- no sparse derivative coloring
- no public promise for final macro syntax

## Milestone 1: Toolchain Lands In Mercury

Bring the working Enzyme setup from the spike into Mercury's Nix workflow.

Tasks:

- Add a dedicated Enzyme dev shell or feature shell using the pinned nightly and
  matching `libEnzyme-22.so`.
- Set `RUSTFLAGS="-Zautodiff=Enable"` only in that Enzyme shell.
- Add a release profile with `lto = "fat"` for the Enzyme test target.
- Document the command that runs Enzyme tests.
- Keep the normal stable/dev workflow working for ordinary tests.

Acceptance:

- `./scripts/test.sh` still runs the normal scaffold tests.
- An explicit Enzyme command can build one `#![feature(autodiff)]` target in
  release mode.
- The command fails clearly outside the Enzyme shell instead of silently running
  the wrong compiler.

## Milestone 2: First Differentiated Kernel

Add a generalized Rosenbrock scalar-output kernel as the first AD-safe model.

Target shape:

```rust
fn rosenbrock(x: &[f64], out: &mut f64) {
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        acc += 100.0 * a * a + b * b;
    }
    *out = acc;
}
```

Tasks:

- Put the kernel in an Enzyme-only test/example target, not the stable core path.
- Generate the reverse-mode entry point with raw `#[autodiff_reverse]` first.
- Keep the raw Enzyme ABI local to this test target.
- Add the analytic Rosenbrock gradient as the correctness oracle.

Acceptance:

- The direct `f64` kernel returns the expected scalar value at `x_i = 0.5`.
- Enzyme returns a gradient that matches the analytic gradient to tight
  tolerance.
- The differentiated call graph uses only slice reads, local scalar mutation,
  fixed loops over input length, and output buffers.

## Milestone 3: Finite-Difference Validation Utilities

Implement the tiny validation support Mercury needs before adding more kernels.

Proposed module shape:

```text
src/
  validation.rs
```

Initial API:

```rust
pub struct GradientCheck {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub worst_index: usize,
}

pub fn central_difference_gradient<F>(f: F, x: &[f64], step: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64;
```

Tasks:

- Use central differences.
- Scale perturbations by `max(1, abs(x[i]))`.
- Return diagnostics instead of only booleans.
- Add validation helpers for max absolute and relative gradient error.
- Keep the API independent of Enzyme so it runs on stable Rust.

Acceptance:

- Unit tests cover a quadratic, a small Rosenbrock input, and invalid step
  handling.
- The helper catches an intentionally wrong gradient with a useful worst-index
  diagnostic.
- Normal `./scripts/test.sh` covers these stable utilities.

## Milestone 4: Metis-Style Gradient Check Harness

Turn the Enzyme smoke test into a reusable check pattern.

Proposed result type:

```rust
pub struct EnzymeGradientCheck {
    pub value: f64,
    pub enzyme_gradient: Vec<f64>,
    pub finite_difference_gradient: Vec<f64>,
    pub analytic_gradient: Option<Vec<f64>>,
    pub max_fd_abs_error: f64,
    pub max_analytic_abs_error: Option<f64>,
}
```

Tasks:

- Compare direct model value to the differentiated call's primal output.
- Compare Enzyme gradient to finite differences.
- Compare Enzyme gradient to analytic derivatives where supplied.
- Format diagnostics with input point, worst index, expected value, actual value,
  absolute error, and relative error.

Acceptance:

- Rosenbrock passes Enzyme-vs-analytic and Enzyme-vs-finite-difference checks.
- Failure messages are good enough to identify the broken component without a
  debugger.
- The harness remains scalar-output and dense-gradient only.

## Milestone 5: Registration Syntax Spike

Do not build the final macro first. Prove the viable syntax after the raw ABI
works in Mercury.

Test these options in order:

1. Attribute macro on the model function.
2. Wrapper macro that contains the full function definition.
3. Explicit registration macro that repeats signature and activity information.

Acceptance:

- At least one syntax generates an Enzyme entry point that compiles in Mercury.
- The syntax hides raw activity markers and shadow buffers from normal model
  call sites.
- The architecture doc is updated with the syntax that actually works.

## Milestone 6: Second Kernel For Enzyme-Safety

Add one small matrix-heavy kernel after Rosenbrock passes. This should be
RBD-shaped, not a full linear algebra facade.

Good candidate:

- fixed-size 3D rotation-chain energy or residual
- inputs are a slice of angles
- output is one scalar
- analytic or finite-difference checks are available

Tasks:

- Avoid array zero-initialization patterns that lower to problematic `memset`.
- Build fixed arrays by direct element stores or literals.
- Check Enzyme gradient against finite differences.

Acceptance:

- The kernel compiles under Enzyme.
- The gradient matches finite differences.
- Any IR pattern that fails Enzyme is recorded as an AD-safe-kernel rule.

## File-Level Starting Point

Expected first files:

- `src/validation.rs`
- `tests/validation.rs`
- `examples/` or `tests/` target for the Enzyme Rosenbrock smoke
- `nix/dev-shells.nix`
- `Cargo.toml`
- `docs/architecture.md`

Do not split Mercury into a workspace until the raw Enzyme path and validation
harness prove they need isolation. If fat-LTO or nightly configuration starts to
infect normal development, then introduce a leaf crate for differentiated
kernels.

## Phase 1 Slice Exit Criteria

- Normal Mercury tests still pass on the standard dev path.
- Enzyme tests pass in the pinned Enzyme shell.
- Rosenbrock direct value, Enzyme gradient, finite differences, and analytic
  gradient agree.
- One matrix-heavy kernel compiles and validates.
- The repo documents the first working registration syntax.
- No Metis symbolic/CasADi/generic-scalar architecture is reintroduced.
