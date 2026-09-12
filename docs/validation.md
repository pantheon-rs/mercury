# Validation

## Current checks

`tests/enzyme.rs` applies forward and reverse autodiff to the same slice-based
function from two inputs to three outputs. It checks primal values, non-basis
JVP/VJP seeds against analytic derivatives, a central directional difference,
and the adjoint identity. The implementation adds:

| Boundary | Evidence |
| --- | --- |
| Kernel macro/adapter | Analytic derivatives, inactive configuration, runtime dimensions, batches, preserved seeds, domain and buffer failures |
| Runtime plan | Fan-out, repeated inputs/outputs, finite differences, adjoint identity, Jacobian assembly, cycle/foreign-handle rejection, failure recovery |
| Solves | Pivoted nonsymmetric systems, perturb-and-resolve, cached products, final-root Jacobian, composed residual plans, singularity/nonconvergence |
| Flight trajectory | RK4 analytic cases, ten sensitivities by finite differences, adjoint identity, exact fixed-schedule checkpoint replay |

A compile-fail doctest checks that a borrowed linearization prevents mutation
of its point. These checks establish the tested numerical paths, not general
Enzyme compatibility or performance bounds.

The environment in `nix/rust-toolchain.nix` pins Rust nightly `2026-06-23`
with its matching distributed `enzyme` component for `x86_64-linux`.
Builds use `-Zautodiff=Enable` and release fat LTO. This replaces the March
compiler's CI artifact, whose download returned HTTP 404 during the reset.
The existing Nix input locks are unchanged. See Rust's
[Enzyme installation guide](https://rustc-dev-guide.rust-lang.org/autodiff/installation.html).

```sh
nix develop
cargo test --release --workspace --all-features --locked
./scripts/ci.sh
nix flake check
```

The CI script checks formatting, workspace clippy/release tests, documentation, and
dependencies. Flake checks also build the package in the Nix sandbox. New files
must be tracked for Git-backed flake checks to include them. Coverage is deferred
until the library has executable behavior; instrumented Enzyme paths previously
failed on injected atomic counters.

The flight kernel exposed another compiler limitation: zero-initialized
temporary RK arrays followed by overwrite loops failed Enzyme's `memset` type
inference. Explicit element construction passes. This is a finding for that
kernel and compiler, not a ban on mutable arrays.

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
Revalidate a specific kernel before expanding the supported subset. The implementation
does not prove automatic custom-rule substitution, nested AD, or allocation-free
generated derivatives.

## Limits

Runtime batches currently loop over scalar derivatives. The native-width probes
above do not establish a faster adapter. Workspaces reuse graph buffers; batch
growth, dense Jacobian assembly, and faer's high-level factor/solve paths may
allocate. LU rejects zero/non-finite pivots without providing a condition estimate.
Newton requires a suitable initial guess. Replay evidence covers the example's
fixed plan and held-input schedule on the pinned build/platform.

`deny.toml` acknowledges [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436):
faer transitively uses the unmaintained `paste` macro crate. An isolated check of
faer 0.24.4 found the same dependency. The advisory reports no vulnerability or
patched version; the exception stays specific to this maintenance notice.
