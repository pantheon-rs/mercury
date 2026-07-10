# Fable Implementation Brief: Differentiable Function Contract

Status: proposed
Date: 2026-07-10
Audience: Fable / implementation agent

## Executive Instruction

Implement Mercury's missing end-to-end differentiable function contract before
starting SVD, EVD, GEVD, sparsity, optimization, or additional decomposition
work.

The deliverable is not another isolated numerical algorithm. It is one complete
path from an ordinary vector-valued Rust kernel to reusable primal, JVP, VJP,
and dense-Jacobian callbacks, followed by one aerospace-shaped example that
uses those callbacks for linearization.

The intended user promise is:

> Write a pure numerical function in ordinary `f64` Rust. Mercury produces a
> validated differentiable operator with primal, JVP, and VJP entry points.

This brief is intentionally narrower than the long-term architecture. Complete
this vertical slice before broadening the math surface.

## Why This Is The Next Slice

Mercury already has strong building blocks:

- a working pinned Rust/Enzyme toolchain
- `SVector`, `SMatrix`, and `Quaternion` kernel types
- Enzyme tests through mutation, loops, branches, and fixed-size solves
- host-side LU, LLT, LDLT, QR, and least-squares implementations
- validated manual JVP/VJP formulas for solves
- finite-difference gradient validation

The missing product boundary is composition:

- `scalar_objective!` only handles a single `&[f64] -> f64` shape and only
  exposes a reverse gradient;
- Mercury has no general `R^n -> R^m` operator;
- the existing solve JVP/VJP functions are manually callable utilities, not
  rules proven to compose automatically inside an outer differentiated model;
- no downstream-shaped model demonstrates how Vulcan or Icarus would consume
  Mercury.

Do not infer readiness from isolated derivative tests. The phase is complete
only when one vector-valued model exposes callbacks that a future Icarus
linearizer can call without knowing Enzyme's activity markers or shadow ABI.

## Identity And Ownership Boundaries

Use this boundary while implementing:

### Mercury owns

- plain-`f64` kernel conventions
- POD-transparent kernel math types
- generation of primal/JVP/VJP entry points
- dense Jacobian and gradient conveniences derived from JVP/VJP
- derivative-aware numerical primitives
- derivative validation and diagnostics
- a host-side operator interface for invoking already-generated derivatives

### Vulcan will own

- stateless aerospace equations
- atmosphere, gravity, aerodynamics, propulsion, frames, and mass properties
- rigid-body right-hand-side functions
- table data, units, and aerospace conventions

### Icarus will own

- state and signal ownership
- component lifecycle, topology, and scheduling
- integrator selection and event policy
- composition of component-local JVPs/VJPs
- global linearization, trim problem construction, logging, and replay

Do not put aerospace physics or simulation lifecycle code into Mercury. The
aerospace code requested below is an example only, used to prove the downstream
contract.

## Non-Goals

Do not implement any of the following in this slice:

- SVD, EVD, or GEVD
- sparse matrices, sparsity tracing, or graph coloring
- an NLP solver or optimization DSL
- root finding or implicit-function derivatives
- adaptive ODE integration or event sensitivities
- symbolic tracing or a generic scalar trait
- a broad replacement for `std`, `nalgebra`, or `faer`
- a workspace split unless implementation evidence makes a proc-macro crate
  unavoidable
- changes in the C++ Metis, Vulcan, or Icarus repositories

Preserve the existing public APIs unless a narrowly justified correction is
required. In particular, keep `scalar_objective!` working.

## Global Constraints

1. Model code remains ordinary `f64` Rust. Do not introduce `Scalar<T>`, dual
   numbers, or symbolic values.
2. Raw `#[autodiff_*]` attributes, activity markers, derivative symbols, and
   shadow-buffer initialization remain internal to generated Mercury code.
3. No heap allocation, `dyn Trait`, I/O, global mutation, or opaque FFI is
   permitted inside differentiated kernels.
4. Host-side wrappers may allocate output vectors and matrices for ergonomics.
5. In-place kernel entry points must exist so downstream hot loops can reuse
   buffers.
6. Every new derivative path must be checked against finite differences and an
   analytic derivative where practical.
7. Do not claim that a derivative rule composes automatically until an outer
   Enzyme kernel proves it.
8. Keep differentiated error handling separate from host validation. Validate
   dimensions before entering a generated derivative function.
9. Document pathwise derivative behavior for branches and nonsmooth points.
10. Run `./scripts/ci.sh` before considering the work complete.

## Work Package 1: Enzyme Capability Matrix

Before selecting final public syntax, add focused compile-and-run tests for the
pinned toolchain.

Test these kernel shapes:

1. `&[f64] -> &mut [f64]` with multiple outputs.
2. A fixed-size Mercury struct or aggregate as input and output.
3. Forward JVP with input tangents and output tangents.
4. Reverse VJP with an arbitrary output cotangent seed.
5. Local mutation and fixed-structure `for` loops.
6. A value-dependent `if` away from its switching boundary.
7. A small `match` over inactive structural data.
8. A bounded `while` loop, including one test whose iteration count depends on
   an active value.

The `while` test proves compiler capability only. Its rustdoc must state that
Enzyme differentiates the executed iterations, not an abstract converged
solution.

Also preserve or add documented negative cases for:

- `dyn Trait` in the differentiated call graph
- a `Result<aggregate, Error>` return if it still triggers the known copy-type
  failure
- heap-backed `Vector`/`Matrix` use in the differentiated path

Record the result in a short table under `docs/architecture/`, with columns for
pattern, compile status, derivative validation status, and author guidance.
Tests, not prose alone, are the source of truth for supported positive cases.

### Acceptance

- All positive cases compile and their derivatives match finite differences.
- Unsupported cases have explicit guidance and do not silently become part of
  the public contract.
- No public macro design is committed before the input/output ABI is proven.

## Work Package 2: General Differentiable Operator

Add a general vector-valued operator facility. Exact macro spelling is open to
implementation evidence, but the generated behavior is fixed.

Conceptual kernel:

```rust
fn eval(x: &[f64], y: &mut [f64]) {
    // Ordinary Rust numerical code.
}
```

Required generated operations:

```rust
fn eval(x: &[f64], y: &mut [f64]);

fn jvp(
    x: &[f64],
    x_dot: &[f64],
    y: &mut [f64],
    y_dot: &mut [f64],
);

fn vjp(
    x: &[f64],
    y_bar: &[f64],
    y: &mut [f64],
    x_bar: &mut [f64],
);
```

The exact argument order may follow the generated Enzyme ABI internally, but
the public wrapper should use mathematical names and validate all dimensions
before dispatch.

Also provide host-side conveniences:

- allocate-and-return `evaluate`
- allocate-and-return directional JVP
- allocate-and-return VJP
- dense Jacobian materialization
- scalar-output gradient as a thin special case

Dense Jacobian should be derived from JVP columns or VJP rows. It should choose
the cheaper direction using input and output dimensions when both modes are
available. Do not implement a separate finite-difference Jacobian as the
production derivative.

### Host-Side Operator Contract

Expose a Mercury-owned host interface that Icarus could store behind runtime
dispatch without placing that dispatch inside an Enzyme call graph. A trait or
callback struct is acceptable.

Conceptually:

```rust
pub trait DifferentiableOperator {
    fn input_dimension(&self) -> usize;
    fn output_dimension(&self) -> usize;
    fn eval_into(&self, x: &[f64], y: &mut [f64]);
    fn jvp_into(&self, x: &[f64], x_dot: &[f64], y: &mut [f64], y_dot: &mut [f64]);
    fn vjp_into(&self, x: &[f64], y_bar: &[f64], y: &mut [f64], x_bar: &mut [f64]);
}
```

This trait is host-side only. Enzyme differentiates the concrete generated
kernel, never a `dyn DifferentiableOperator` call.

If a trait creates unnecessary complexity, use a struct of generated function
pointers with dimensions. Preserve the same separation.

### Validation Support

Extend `validation` with vector-output support:

- central-difference directional derivative
- central-difference dense Jacobian for tests
- matrix/Jacobian comparison diagnostics
- maximum absolute and relative error with row/column location

### Acceptance

- A nonlinear `R^2 -> R^3` test matches an analytic Jacobian.
- JVP equals `J * x_dot`.
- VJP equals `J^T * y_bar` for a nontrivial cotangent seed.
- Dense Jacobian agrees with central differences.
- Existing `scalar_objective!` tests remain green.
- Callers do not import anything from `std::autodiff`.
- In-place calls perform no host allocation after buffers are prepared.

## Work Package 3: Rule-Composition Go/No-Go Spike

Resolve the largest open architectural claim: can Mercury attach an owned
derivative rule to a primitive so that an outer Enzyme-generated derivative
uses it automatically?

Use the smallest possible experimental primitive. Do not start with dynamic QR
or a large factorization. A fixed 2x2 solve or similarly small opaque operation
is enough.

Required experiment:

```text
outer scalar/vector model
  -> calls one opaque Mercury primitive
  -> outer derivative generated by Enzyme
  -> derivative must use the Mercury-owned rule rather than differentiating
     the primitive implementation
```

Validate the result against:

- finite differences
- the existing analytic solve JVP/VJP formula
- differentiation through the transparent fixed-size implementation

### Go Result

If automatic rule composition works on the pinned Rust toolchain:

- isolate the minimal reusable mechanism;
- add a regression test that fails if Enzyme stops selecting the custom rule;
- document how primal state/tape and cotangents cross the boundary;
- then apply the mechanism to exactly one solve primitive.

### No-Go Result

If Rust's current `std::autodiff` surface cannot express or select the rule:

- do not use unsafe LLVM metadata workarounds in this slice;
- record the limitation in an ADR;
- update architecture language so `solve_jvp`/`solve_vjp` are described as
  explicit derivative operators, not transparent custom rules;
- keep differentiating through small fixed-size solves;
- defer backend-opaque dynamic solves from monolithic Enzyme kernels;
- define the future Icarus composition model around explicit local callbacks.

The no-go outcome is useful and acceptable. An undocumented or untested claim
is not.

## Work Package 4: Aerospace-Shaped Vertical Slice

Add an example, not a Mercury domain module, that resembles the first future
Vulcan/Icarus consumer.

Preferred example: a 13-state rigid-body RHS with:

- position: 3
- body velocity: 3
- quaternion attitude: 4
- body angular velocity: 3
- applied body force: 3 controls
- applied body moment: 3 controls
- mass and a full symmetric 3x3 inertia tensor as parameters

The RHS should exercise:

- `SVector<3>` cross products
- quaternion rotation or kinematics
- `SMatrix<3, 3>` multiplication
- the fixed-size inertia solve
- vector-valued outputs
- ordinary local mutation and loops

Keep state packing explicit in the example. Do not introduce an aerospace state
type into Mercury's public API.

Demonstrate:

1. ordinary primal RHS evaluation;
2. one state-direction JVP;
3. one arbitrary-output-cotangent VJP;
4. dense linearization;
5. extraction of `A = df/dx` and `B = df/du` blocks;
6. JVP agreement with a central-difference directional derivative.

If the full 13-state example exposes an unrelated Enzyme compiler bug, first
land a smaller J2 gravity `R^3 -> R^3` example, document the blocker, and keep
the 6DOF example as the required exit criterion rather than weakening it
silently.

### Acceptance

- The example runs under `./scripts/run.sh` or the repository's established
  example check.
- No raw Enzyme types appear in the example.
- A future Vulcan author can copy the kernel shape without adopting symbolic or
  generic-scalar syntax.
- A future Icarus author can consume the generated operator without
  differentiating its scheduler or signal registry.

## Work Package 5: Architecture Reconciliation

After implementation results are known, update the architecture documents to
match demonstrated behavior.

Required corrections:

- distinguish transparent kernels, rule-owned primitives, and host-only
  operations;
- mark JVP, VJP, and dense Jacobian status from actual implementation;
- state the custom-rule composition result honestly;
- replace SVD/EVD as the immediate next phase with this completed vertical
  slice;
- document that normal control flow is supported pathwise, while branch
  boundaries and iteration-count changes remain nonsmooth;
- explain that Icarus should compose generated component callbacks rather than
  place dynamic component dispatch inside an Enzyme kernel;
- identify interpolation as the next consumer-driven primitive after this
  slice.

Do not rewrite historical ADRs. Add a superseding ADR where a previous accepted
claim changes.

## Recommended Implementation Order

1. Capability tests and result table.
2. Vector-output finite-difference validation helpers.
3. Raw internal vector JVP/VJP proof.
4. Public generated operator and host interface.
5. `R^2 -> R^3` analytic-law tests.
6. Rule-composition spike and ADR.
7. Aerospace-shaped 6DOF example and linearization test.
8. Documentation reconciliation.
9. Full CI.

Keep commits aligned to these boundaries where practical. Do not mix broad
container refactors or unrelated formatting into the implementation.

## Definition Of Done

This work is complete only when all of the following are true:

- Mercury exposes a tested vector-valued primal/JVP/VJP contract.
- Gradient and dense Jacobian are conveniences over that contract.
- The operator can be invoked through a host-side interface suitable for a
  future runtime graph.
- Branching and loop capability is captured by executable tests and documented
  semantics.
- The solve custom-rule composition question has a tested go/no-go answer.
- A 13-state rigid-body example produces `A` and `B` linearization blocks.
- Existing APIs and tests continue to pass.
- `./scripts/ci.sh` passes.
- Architecture prose no longer claims behavior that has not been demonstrated.

## Expected Final Report From Fable

Report:

1. the public API added;
2. supported and unsupported Enzyme patterns discovered;
3. custom-rule composition go/no-go result;
4. 6DOF example derivative errors versus finite differences;
5. allocation behavior of the in-place APIs;
6. tests and CI commands run;
7. deferred work and concrete blockers.

Do not report the phase complete if the implementation stops at a macro spike
without the aerospace-shaped consumer.
