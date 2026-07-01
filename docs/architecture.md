# Mercury Architecture

Mercury is the math substrate for Pantheon.

The crate should remain useful outside aerospace. Aerospace-specific physics
belongs in `vulcan`; plant simulation belongs in `icarus`.

## Phase 1: Enzyme-Backed `f64` Core

Mercury Phase 1 is the practical evolution of Metis:

- keep the "write the model once" workflow
- hide derivative plumbing behind Mercury APIs
- drop the symbolic graph as the core derivative path
- prove fast dense derivatives before building solver and sparsity layers

The decision record is
[`0002: Phase 1 Enzyme-Backed f64 Core`](decisions/0002-phase-1-enzyme-f64-core.md).
The first implementation slice is
[`Phase 1 Gradient Validation Slice`](implementation-plans/phase-1-gradient-validation.md).

The core model language is ordinary Rust over `f64`. Derivatives are generated
from that code with Rust nightly `std::autodiff` / Enzyme. Phase 1 does not use a
generic scalar trait, a dual-number scalar, or a symbolic expression engine as
the central contract.

## Core Commitments

- Model kernels are plain `f64` functions.
- Enzyme activity markers, shadow buffers, and generated derivative entry points
  stay behind Mercury-owned APIs.
- Differentiated kernels live in leaf crates or isolated modules so nightly,
  Enzyme, and fat-LTO costs do not spread through the whole workspace.
- Dense derivatives come first: scalar-output gradients, Jacobian columns or
  JVPs, and finite-difference validation.
- Hessian-vector products wait until nested autodiff is proven in Rust.
- Sparse tracing, graph coloring, nonlinear programming interfaces, and solver
  bindings are not Phase 1 implementation work.

## Author-Facing Contract

Model authors should be able to write normal numeric Rust:

```rust
fn drag(rho: f64, v: f64, area: f64, cd: f64) -> f64 {
    0.5 * rho * v.powi(2) * area * cd
}
```

Control flow is allowed, including `if`, `match`, and fixed-structure loops.
That does not make value-dependent branches smooth. Branches over model values,
`abs`, `min`, `max`, clamps, table lookups, and mode switches are piecewise
operations; Mercury should document their derivative policy and provide smoothed
helpers where optimizers need them.

Phase 1 kernels should stay within a conservative AD-safe subset:

- deterministic numeric code
- scalar, slice, fixed-array, and fixed-size matrix inputs
- local mutation and output buffers only
- no `dyn Trait` in the differentiated call graph
- no I/O, global mutation, threading side effects, or FFI on the differentiated
  path
- no reliance on allocator-heavy or opaque library internals until a compile
  test proves them safe

## Public Derivative Shape

The ergonomic goal is a small Mercury API such as:

```rust
let value = drag(rho, v, area, cd);
let gradient = mercury::grad(drag, &[rho, v, area, cd]);
```

The exact registration syntax is not fixed until it is proven against
`std::autodiff`. A function-like macro that receives only a function name cannot
inspect an already-defined Rust function's signature or body. The first
implementation should use whichever shape compiles reliably:

- an attribute macro on the model function
- a wrapper macro that contains the full function definition
- or an explicit registration macro that repeats the needed signature and
  activity information

The one hard requirement is that users do not hand-write Enzyme activity markers
or derivative shadow-buffer plumbing in normal model code.

## Enzyme-Safe Builders And Tests

Mercury should centralize patterns that produce Enzyme-friendly LLVM IR. Known
early checks include:

- zero-initialization patterns that lower to `llvm.memset`
- fixed arrays and small matrix builders
- slice gradients
- loops over fixed structure
- value-dependent branches
- calls through `nalgebra` and `faer` candidates

Every helper that is intended for differentiated code should have a compile/run
test under the pinned Enzyme toolchain, plus numeric checks against finite
differences or closed-form derivatives.

## Linear Algebra

Phase 1 should not build a full linear algebra facade. It may use simple
`f64`-only helpers and small fixed-size matrix kernels to prove the AD path.

When larger linear algebra enters, `nalgebra` and `faer` are backend candidates,
not the public contract. Solves and factorizations should become Mercury
primitives with validated derivative behavior instead of assuming Enzyme will
always differentiate through arbitrary library internals cleanly.

## Deferred Layers

Sparsity is deferred, not discarded. If real targets become large
direct-collocation or trajectory-optimization problems, Mercury should add a
narrow dependency tracer and coloring layer that feeds compressed Enzyme passes.
That tracer is not a revival of the Metis symbolic graph contract.

Optimization is also deferred. The future shape is one Mercury-owned NLP problem
interface whose callbacks can be filled by Enzyme derivatives. The first solver
backend can be chosen later, after dense derivative callbacks are stable.

## Phase 1 Exit Criteria

- A plain `f64` physics kernel runs natively and produces a reverse-mode gradient
  through the pinned Enzyme toolchain.
- The same kernel has a forward derivative or JVP path.
- Derivatives match finite differences and any available analytic derivatives.
- The chosen differentiable registration syntax has a compile test.
- Unsupported kernel patterns fail with documented guidance.
- The crate still has no symbolic-math dependency, no AD scalar dependency, and
  no C/C++ FFI in the core.
