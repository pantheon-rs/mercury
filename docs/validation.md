# Validation

## Current checks

`tests/enzyme.rs` applies forward and reverse autodiff to the same slice-based
function from two inputs to three outputs. It checks primal values, non-basis
JVP/VJP seeds against analytic derivatives, a central directional difference,
and the adjoint identity. The implementation adds:

| Boundary | Evidence |
| --- | --- |
| Typed function API | Rosenbrock gradients and combined calls, rectangular Jacobians against analytic and finite-difference oracles, graph composition, nonfinite results and independent calls after failure |
| Foundation | Typed vector/matrix arguments and named partials; CSC structure and one JVP per color; analytic Hessians; derivative operators; weighted curvature against perturbed gradients, including linear and implicit solves; missing-rule failure and recovery |
| Simple plan API | Owned values and gradients, faer Jacobian layout, scalar-output validation, solves, error recovery and empty outputs |
| Kernel macro/adapter | Analytic derivatives, inactive configuration, runtime dimensions, batches, preserved seeds, domain and buffer failures |
| Runtime plan | Fan-out, repeated inputs/outputs, finite differences, adjoint identity, Jacobian assembly, cycle/foreign-handle rejection, failure recovery |
| Solves | Pivoted nonsymmetric systems, perturb-and-resolve, cached products, final-root Jacobian, composed residual plans, singularity/nonconvergence |
| Flight trajectory | RK4 analytic cases, ten sensitivities by finite differences, adjoint identity, exact fixed-schedule checkpoint replay |

A compile-fail doctest checks that a borrowed linearization prevents mutation
of its point. These checks establish the tested numerical paths, not general
Enzyme compatibility or performance bounds.

The examples in `docs/api.md` and `docs/advanced.md` are included in rustdoc and
executed by the normal test command. `RUSTDOCFLAGS` supplies Enzyme and release
code-generation settings because doctest compilation does not inherit Cargo's
release profile. The checkpointed flight fixture is in `tests/support/flight.rs`.

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
must be tracked for Git-backed flake checks to include them. Coverage
instrumentation remains deferred; instrumented Enzyme paths previously
failed on injected atomic counters.

The flight kernel exposed another compiler limitation: zero-initialized
temporary RK arrays followed by overwrite loops failed Enzyme's `memset` type
inference. Explicit element construction passes. This is a finding for that
kernel and compiler, not a ban on mutable arrays.

The typed API checks also exposed two numerical limitations on this pin:
reverse differentiation of `sqrt` at zero returned zero (the function is not
differentiable there), and a forward derivative of `(x * f64::MAX) * 2.0`
at `x = 0.1` returned zero instead of overflowing. The latter reproduces with
Enzyme alone in `tests/enzyme.rs`, without Mercury's macro. That regression is
explicitly ignored in CI; run it with
`cargo test --release --test enzyme constant_scaling_reports_derivative_overflow -- --ignored`
inside `nix develop`. Finite checks cannot detect an incorrect finite derivative.
Ordinary-domain analytic checks and explicit nonfinite-result rejection pass;
they do not establish correctness for every floating-point extreme.

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
does not prove automatic custom-rule substitution or allocation-free generated
derivatives. The current forward-over-reverse path is checked separately below.

## Second-order foundation

The pinned compiler passed a direct forward-over-reverse probe and the committed
foundation tests. Typed scalar and vector kernels supply weighted curvature;
tests cover structured arguments, analytic Hessians, derivative operators, and
weighted products through composed plans. A nonsymmetric linear solve and a
coupled implicit root are checked against gradients at perturbed, re-solved
points. Missing curvature rules invalidate the linearization explicitly.

A slice copy into reverse weights failed nested Enzyme type inference. Explicit
array element construction passes and is used in the typed macro. This evidence
covers the tested kernels, not arbitrary nested differentiation. Third-order
rules and automatic curvature generation for dynamic slice kernels are absent.

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
