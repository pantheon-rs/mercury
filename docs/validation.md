# Validation

## Current scaffold

`tests/enzyme.rs` applies forward and reverse autodiff to the same slice-based
function from two inputs to three outputs. It checks primal values, non-basis
JVP/VJP seeds against analytic derivatives, a central directional difference,
and the adjoint identity. These tests exercise the compiler; they do not
establish a public operator API or graph composition.

The environment in `nix/rust-toolchain.nix` pins Rust nightly `2026-06-23`
with its matching distributed `enzyme` component for `x86_64-linux`.
Builds use `-Zautodiff=Enable` and release fat LTO. This replaces the March
compiler's CI artifact, whose download returned HTTP 404 during the reset.
The existing Nix input locks are unchanged. See Rust's
[Enzyme installation guide](https://rustc-dev-guide.rust-lang.org/autodiff/installation.html).

```sh
nix develop path:.
cargo test --release --locked --test enzyme
./scripts/ci.sh
nix flake check
```

The CI script checks formatting, clippy, release tests, documentation, and
dependencies. Flake checks also build the package in the Nix sandbox. New files
must be tracked for Git-backed flake checks to include them. Coverage is deferred
until the library has executable behavior; instrumented Enzyme paths previously
failed on injected atomic counters.

## Isolated probes: 2026-09-12

Scratch probes used the pinned compiler `4429659e4` (LLVM 22.1.7), release
optimization, fat LTO, and `-Zautodiff=Enable`. They are not committed regression
tests or performance measurements.

| Pattern | Observed result |
| --- | --- |
| Width-4 forward `Dual`, slices | Four analytic JVPs passed |
| Width-4 reverse `Duplicated`, slices | Four analytic VJPs passed |
| Width-4 packed `Dualv`, slices | Four analytic JVPs passed |
| Scalar-reference `Dualv`; slice-output `DualvOnly` | Compiler internal errors |
| Proc macro emitting both AD attributes | Primal, JVP, and VJP passed |

Width follows the derivative name, for example
`#[autodiff_reverse(df, 4, Duplicated, Duplicated)]`. Reverse batching works
without a `Duplicatedv` activity. These results do not establish reusable reverse
tapes, allocation bounds, or speedups. See the pinned compiler's
[argument lowering](https://github.com/rust-lang/rust/blob/4429659e4745016bd3f26a4a421843edc7fbc422/compiler/rustc_codegen_llvm/src/builder/autodiff.rs)
and [batching test](https://github.com/rust-lang/rust/blob/4429659e4745016bd3f26a4a421843edc7fbc422/tests/codegen-llvm/autodiff/batched.rs).

## Earlier compiler findings

The July 2026 experiments used nightly `2026-03-03` (`ec818fda3`, LLVM 22)
with nalgebra 0.34.2 and faer 0.23.2.
They reported copy-type analysis failures through nalgebra constructors and
solves, and SIMD-dispatch or allocator failures through faer operations. Direct
element-wise array kernels passed selected checks. These are historical results,
not blanket statements about either library or current compiler releases.

The original reports and numerical reference tests remain in Git at `58d2f49`,
including `docs/decisions/0003-differentiable-primitives-identity.md` and `tests/`.
Revalidate a specific kernel before expanding the supported subset. The scaffold
does not prove automatic custom-rule substitution, nested AD, or allocation-free
generated derivatives.

## Checks for the next implementation

- Operators: shape errors, complete writes, unchanged input seeds, repeated calls,
  analytic checks, and directional finite differences.
- Batching: seed independence, packed output layout, partial final batches, and
  empty batches.
- Composition: fan-out, repeated inputs, shared parameters, adjoint identity,
  and rejection of stale plans or linearizations; failed evaluation must
  invalidate the linearization.
- Solves: residual accuracy and sensitivities compared with perturb-and-resolve;
  singularity or nonconvergence must invalidate results.
- Simulation example: compare the numerical and differentiated evaluation paths,
  with explicit held state and the same integration stages.
