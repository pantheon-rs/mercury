# Mercury Architecture

> **Mercury is a differentiable-by-construction math library for engineering
> simulation and optimization. Every primitive is plain-`f64` Rust with a
> validated, Mercury-owned derivative rule. Enzyme is the derivative engine;
> Mercury owns the mathematical joints where brute-force AD is wrong.**

Mercury is the math substrate for `pantheon-rs`. It should remain useful
outside aerospace: aerospace-specific physics belongs in `vulcan`; plant
simulation belongs in `icarus`.

The identity decision is
[`0003: Differentiable Primitives Identity`](decisions/0003-differentiable-primitives-identity.md);
the engine decision is
[`0002: Phase 1 Enzyme-Backed f64 Core`](decisions/0002-phase-1-enzyme-f64-core.md).

## The Thesis

Metis's identity was a bridge between two worlds — the same templated model
code compiled to Eigen arithmetic or a CasADi graph — with a large math
toolbox on top. Enzyme dissolves that bridge: plain `f64` code is
differentiated directly by the compiler.

Mercury's bridge is a different one: **primal code to correct derivative.**
Enzyme handles arbitrary user kernel code; Mercury supplies the
mathematically correct derivative rule at every primitive where
differentiating the algorithm is wrong or wasteful. That rule-ownership is
what a bare `#[autodiff]` attribute does not give you, and it is the
affirmative reason each Mercury subsystem exists.

## Core Types: POD-Transparency

Mercury owns its math types. The public contract never exposes `nalgebra` or
`faer` types; backends may appear later *behind* primitives.

One design law governs every type — **POD-transparency**:

- plain contiguous `f64` storage
- no hidden allocation on differentiated paths
- no `dyn`, no generic scalar
- a `Duplicated` shadow of any Mercury type is the same type zeroed
- every type has an Enzyme compile test proving it passes through
  `#[autodiff]` cleanly

The types:

- `SVector<const N: usize>` / `SMatrix<const R: usize, const C: usize>` —
  stack `[f64; N]`-backed fixed-size types. The aerospace hot path
  (3-vectors, 3×3 DCMs, 6×6 spatial matrices, quaternion storage) and
  Enzyme's happiest input shape.
- `Vector` / `Matrix` — heap `Vec<f64>`-backed dynamic dense types for
  problem-scale data (trajectories, Jacobians, collocation grids).
- `Quaternion` plus rotation constructors and conversions
  (DCM ↔ quaternion ↔ Euler) with analytic derivatives.

Operator overloading covers the ring operations (`+`, `-`, `*` with scalars
and matrices). **Solves and factorizations are primitives, not methods** —
`mercury::solve(&A, &b)` carries the adjoint rule, so a `faer` backend can
slot in later without the public contract changing.

The `f64`-only commitment is the point, not a limitation: Metis needed
generic scalars because the scalar *was* the backend switch. Mercury's
backend switch is a compiler pass, so the types stay concrete and simple.

## Derivative Rules at the Joints

Enzyme differentiates user kernels and residual/RHS functions. At the
primitives, Mercury owns the rule, built by composing Enzyme calls on the
pieces:

| Primitive          | Rule                                                        |
| ------------------ | ----------------------------------------------------------- |
| Root finding       | Implicit function theorem on the residual — never the loop  |
| Linear solve       | Adjoint rule (two solves) — never the factorization         |
| Interpolation      | Closed-form basis derivatives; documented breakpoint policy |
| ODE integration    | Differentiate-through fixed-step RK; sensitivity later      |
| Quaternions        | Analytic                                                    |

A caller cannot tell whether Enzyme or a hand-derived rule produced the
numbers; both are held to identical validation.

## The Derivative Contract

Everything differentiable exports the same shapes — user kernels via the
`scalar_objective!` macro, primitives via their owned rules. These are chosen
because they are exactly the callbacks the end-game optimization layer
consumes:

| Shape                | Mode                 | Status                   |
| -------------------- | -------------------- | ------------------------ |
| `value_and_gradient` | reverse              | exists                   |
| `jvp`                | forward              | Phase 1 exit criterion   |
| `vjp`                | reverse              | with `jvp`               |
| `jacobian` (dense)   | batched jvp/vjp      | foundations phase        |
| `hvp`                | forward-over-reverse | deferred until nested AD |

## Module Map

Single crate; modules in dependency order. No workspace until something
forces one.

```text
src/
  lib.rs
  core/        # SVector, SMatrix, Vector, Matrix, ops — POD-transparency law
  ad/          # scalar_objective macro, jvp/vjp/jacobian, finite-diff validation
  linalg/      # solve() + factorizations (LLT, LU) as adjoint-rule primitives
  geometry/    # Quaternion, DCM/Euler conversions, analytic derivatives
  interp/      # gridded 1D → N-D tables, documented breakpoint policy
  roots/       # Newton/bracketing + IFT derivative rule
  integrate/   # fixed-step RK4/RK45 differentiate-through; sensitivity later
```

Today's `objective.rs` / `validation.rs` are the `ad/` module — Phase 1 was
building it all along; the reframe names it.

### Phase 3: decomposition suite

Phase 3 mines faer's layout (vendored at `ref/faer`) for an Enzyme-compatible
decomposition suite: Cholesky (`llt_factor`, unpivoted `ldlt_factor`) and
Householder QR (`qr_factor` with square `solve` and full-rank `solve_lstsq`),
built on a shared triangular-substitution substrate and a `Perm` type. All
square-solve factorizations (LU, LLT, LDLT) implement one `Factorization`
trait, so a single adjoint rule (`solve_vjp`/`solve_jvp`) serves them all;
least squares carries its own rule (`lstsq_vjp`/`lstsq_jvp`), validated by
the same three-way agreement law. The one kernel-side addition is
`solve_spd_fixed_unchecked` (unpivoted LLT, NaN-propagating). Matrix views
and SVD/EVD are Phase 4 candidates per the Phase 3 spec.

## Phasing

Foundations first, in dependency order. NLP / IPOPT / collocation /
transcriptions are the end-game the foundations are designed for — the same
strategy Metis followed.

| Phase       | Contents                                        | Why this order                                                             |
| ----------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| 1           | `ad/`: macro, gradient, jvp/vjp, FD validation  | The engine room — everything else validates against it                     |
| 2           | `core/` types + `linalg/` solve; `geometry/`    | Types are the vocabulary; quaternions are cheap analytic wins              |
| 3 (current) | decomposition suite: Cholesky (LLT/LDLT), QR    | Mines faer's layout for an Enzyme-compatible factorization substrate       |
| 4 (planned) | SVD / EVD / GEVD                                | faer's heaviest machinery; derivative rules complicated by degeneracy      |
| 5           | `interp/` 1D → N-D gridded                      | The aerospace workhorse; first real test of the kink-policy discipline     |
| 6           | `roots/` (IFT) + `integrate/` (RK)              | First *composed* rules — consume Enzyme-on-residuals plus linalg solves    |
| 7           | NLP interface, solver backend, transcriptions   | Consumes the derivative-contract callback shapes                           |

Each phase gets its own decision record and implementation plan under
`docs/decisions/` and `docs/implementation-plans/`.

## Validation: The Three-Legged Test Law

Every primitive ships all three or it does not merge:

1. **Enzyme compile test** — the pattern passes `#[autodiff]` under the
   pinned toolchain. This is nightly; it *will* move, and these tests are
   the regression tripwire.
2. **Finite-difference cross-check** — via the `validation` module.
3. **Analytic check** — wherever closed-form derivatives exist
   (quaternions, interpolation bases, linear solve).

Unsupported patterns (e.g., `dyn Trait` in the differentiated call graph)
get **negative tests** with documented guidance — the AD-safe subset is an
enforced contract, not prose.

## Error Handling

Fallible primitives (`solve` on a singular matrix, interpolation
out-of-bounds) return `Result` at the API boundary, *before* entering
differentiated code. Differentiated paths stay panic-free and branch-light.

Value-dependent branches, `abs`, `min`/`max`, clamps, table lookups, and
mode switches remain piecewise operations: every primitive with a kink or
policy choice documents its derivative behavior in its rustdoc. This
per-primitive derivative-policy documentation is Mercury's equivalent of
Metis's user-guide discipline.

## Author-Facing Contract

Model authors write normal numeric Rust:

```rust
fn drag(rho: f64, v: f64, area: f64, cd: f64) -> f64 {
    0.5 * rho * v.powi(2) * area * cd
}
```

Control flow is allowed, including `if`, `match`, and fixed-structure loops.
Phase 1 kernels stay within the conservative AD-safe subset:

- deterministic numeric code
- scalar, slice, fixed-array, and Mercury-type inputs
- local mutation and output buffers only
- no `dyn Trait` in the differentiated call graph
- no I/O, global mutation, threading side effects, or FFI on the
  differentiated path
- no allocator-heavy or opaque library internals until a compile test proves
  them safe

Raw Enzyme activity markers, shadow buffers, and generated derivative entry
points stay behind Mercury-owned APIs.
