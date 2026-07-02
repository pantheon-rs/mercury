# 0003: Differentiable Primitives Identity

Status: accepted
Date: 2026-07-02

## Decision

Mercury is a differentiable-by-construction math library for engineering
simulation and optimization. Every primitive is plain-`f64` Rust with a
validated, Mercury-owned derivative rule. Enzyme is the derivative engine;
Mercury owns the mathematical joints where brute-force AD is wrong or
wasteful.

Decision 0002 chose the engine (Enzyme over dual-mode scalar types). This
record chooses the library's job. Metis's identity was a bridge between the
numeric and symbolic worlds with a math toolbox on top. Enzyme dissolves that
bridge, so Mercury's bridge is a different one: **primal code to correct
derivative**, with Mercury as the rule-owner at every primitive.

The one-line answer to "what does Mercury do that a bare `#[autodiff]`
attribute does not":

> Write ordinary `f64` Rust; get correct derivatives — because every Mercury
> primitive owns its derivative rule.

## The Three Pillars

### 1. Mercury-owned, POD-transparent types

Mercury defines its own math types rather than adopting `nalgebra` or `faer`
types as the public contract. All types obey one design law,
**POD-transparency**: plain contiguous `f64` storage, no hidden allocation on
differentiated paths, no `dyn`, no generic scalar. A `Duplicated` shadow of
any Mercury type is the same type zeroed. Every type carries an Enzyme
compile test proving it passes through `#[autodiff]` cleanly.

The `f64`-only commitment is the point, not a limitation: Metis needed
generic scalars because the scalar *was* the backend switch. Mercury's
backend switch is a compiler pass, so the types stay concrete.

### 2. Mercury-owned derivative rules at the joints

Enzyme differentiates user kernels and residual/RHS functions. At primitives
where differentiating the algorithm is mathematically wrong or wasteful,
Mercury supplies the correct rule by composing Enzyme calls on the pieces:

- **Root finding** — implicit function theorem on the residual, never
  differentiate the Newton loop.
- **Linear solve / factorization** — adjoint rule (two solves), never
  differentiate the factorization; this also frees the backend to be `faer`
  later without contract change.
- **Interpolation** — closed-form basis derivatives with a documented policy
  at breakpoints.
- **ODE integration** — differentiate-through for fixed-step RK first;
  sensitivity/adjoint methods later.
- **Quaternions / rotations** — analytic.

A caller cannot tell whether Enzyme or a hand-derived rule produced the
numbers; both are held to the same validation.

### 3. One derivative contract

Everything differentiable — user kernels via the macro, primitives via owned
rules — exports the same shapes, chosen because they are exactly the
callbacks the end-game optimization layer consumes:

| Shape                | Mode                 | Status                   |
| -------------------- | -------------------- | ------------------------ |
| `value_and_gradient` | reverse              | exists                   |
| `jvp`                | forward              | Phase 1 exit criterion   |
| `vjp`                | reverse              | with `jvp`               |
| `jacobian` (dense)   | batched jvp/vjp      | foundations phase        |
| `hvp`                | forward-over-reverse | deferred until nested AD |

## Scope and End-Game

Foundations first, in dependency order; NLP / IPOPT / collocation /
transcriptions are the end-game the foundations are designed for — the same
strategy Metis followed.

| Phase         | Contents                                                       |
| ------------- | -------------------------------------------------------------- |
| 1 (current)   | `ad/`: macro, gradient, jvp/vjp, finite-difference validation  |
| 2             | `core/` types + `linalg/` solve with adjoint rule; `geometry/` |
| 3             | `interp/` gridded 1D → N-D tables                              |
| 4             | `roots/` (IFT) + `integrate/` (fixed-step RK)                  |
| 5             | NLP interface, solver backend, transcriptions                  |

Each phase gets its own decision record and implementation plan.

## Alternatives Rejected

- **Enzyme end-to-end** (differentiate through Newton loops, integrator
  loops, and factorizations): one mechanism, but yields derivatives of the
  *approximation* (noisy near solver tolerance, cost scales with iteration
  count), wastes factorization structure, and blocks external backends whose
  internals Enzyme cannot see.
- **Facade over ecosystem crates** (`nalgebra`/`faer` as the public
  contract): least code, but derivative correctness becomes hostage to
  third-party internals that mostly do not pass Enzyme, and Mercury has no
  identity of its own.

## Consequences

- Every primitive ships with the three-legged test law (Enzyme compile test,
  finite-difference cross-check, analytic check where closed form exists) or
  it does not merge.
- Unsupported patterns become *negative tests* with documented guidance,
  turning the AD-safe subset from prose into enforced contract.
- Fallible primitives return `Result` at the API boundary before entering
  differentiated code; differentiated paths stay panic-free and
  branch-light.
- Solves and factorizations are free functions/primitives, not methods on
  types, so backends can change without touching the public contract.
- The derivative-shape table is a stable public commitment; the future
  optimization layer is written against it.
