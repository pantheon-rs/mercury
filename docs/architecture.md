# Mercury architecture

Mercury is the numerical and differentiation foundation for `pantheon-rs`.
The implementation currently contains an Enzyme compiler scaffold and
derivative checks. The contracts below specify future work.

## Numerical boundary

A kernel computes `y = f(q, c)`. `q` contains active numerical values,
including state, controls and any parameters whose derivatives are requested.
`c` contains inactive configuration: dimensions, indices, modes and immutable
data that are deliberately held constant. A parameter must not be hidden in
`c` merely because it changes infrequently.

Kernels use ordinary `f64` arithmetic, arrays and slices. They have explicit
inputs and outputs, with no I/O, global mutation or graph traversal. Fixed
arrays cover small vectors and matrices. Do not introduce generic scalars or
a new matrix library. Add operations for real consumers and validate them
under the pinned Enzyme toolchain.

Rust compiles each named kernel, and Enzyme generates its derivative bodies.
Mercury's operator contract exposes evaluation, a Jacobian-vector product
`Jv`, and a vector-Jacobian product represented as `Jᵀw`. Macros remove
activity and buffer boilerplate without introducing a symbolic DSL.
Compilation success alone does not establish derivative correctness or an
allocation bound.

## Operators, plans and workspaces

An **Operator** describes a numerical map: its evaluation and derivative
entry points, dimensions and conservative dependency contract. Enzyme kernels
and explicitly differentiated solve operators use the same contract.

A **Plan** owns operator instances, inactive configuration, connections,
execution order, buffer layout and structural derivative information. One
plan drives both numerical evaluation and derivative propagation. Runtime
composition calls already-compiled operators; the graph scheduler remains
outside the Enzyme call graph.

A **Workspace** owns mutable evaluation buffers, saved primal values,
derivative scratch and numerical factors. Each concurrent evaluation needs
its own workspace. A linearization refers to a particular primal point and
plan epoch; its saved values cannot be reused after inputs or configuration
change.

Validate dimensions and buffer aliasing before dispatch. A numerical failure
invalidates outputs and cached derivatives; it must not expose stale results.
The operator boundary preserves caller seeds and overwrites destination
buffers. Adapters initialize Enzyme shadow storage and isolate any seed
mutation. The plan explicitly accumulates contributions from fan-out,
repeated inputs and shared parameters, in a defined order.

## Graph and time semantics

Graph edits produce a new immutable plan epoch. Validate dimensions,
connections and cycles before publishing it between evaluations or accepted
simulation steps. Rebuild affected layouts and structural caches; never edit
topology during a derivative sweep or nonlinear solve.

Icarus owns time, state commitment and connection semantics. A same-time
connection participates in an acyclic dependency schedule. A delay reads an
explicit snapshot of previously committed state. Its primal value remains
unchanged during evaluation, while its tangent can propagate across steps.
A declared implicit group defines a residual equation and a solve operator.
Condensing those groups must leave an acyclic schedule;
an unexplained cycle is an error, not an implicit solver request.

## Jacobians and linear algebra

Begin with dense local Jacobians obtained from seeded JVPs or VJPs. Global
Jacobians follow the chain rule through the plan; graph adjacency alone is
not their sparsity pattern. Structural dependencies must conservatively cover
all branches allowed by an epoch. A numerical zero does not remove an entry.

When scale requires sparse assembly, the plan owns stable row/column order,
CSC structure and scatter maps. Numerical values change per linearization.
Symbolic factorization can be reused for an unchanged pattern; numerical
factors can be reused only for an unchanged matrix.

Add faer as the sole general linear-algebra dependency when the first solve
consumer needs it. Use its matrices, borrowed views, factorizations and
workspace APIs; do not reproduce its decomposition algorithms. No nalgebra
dependency or parallel Mercury matrix hierarchy is planned. Arrays can be
borrowed through faer views, while owned faer matrices may include column
padding. Packing, shape and stride handling belong at that boundary.
Faer operations remain outside differentiated kernels unless individually
validated; changing matrix libraries does not establish Enzyme compatibility.

## Differentiating solves

For a nonsingular real system `A x = b`, implement the first-order rules:

```text
JVP:  A dx = db - dA x
VJP:  Aᵀ λ = x_bar;  b_bar = λ;  A_bar = -λ xᵀ
```

Reuse the primal factors for ordinary or transpose solves. For sparse inputs,
compute cotangents only for represented matrix entries. The plan accumulates
these local results.

For a converged root `R(z, q) = 0`, differentiate its residual:

```text
JVP:  R_z dz = -R_q dq
VJP:  R_zᵀ λ = z_bar;  q_bar = -R_qᵀ λ
```

These rules require a differentiable local solution and invertible relevant
Jacobian. Solve errors affect accuracy; failure must remain explicit.
Differentiating a finite iteration sequence is a different contract. First
implement solve operators in host composition: custom rules are not
automatically substituted inside arbitrary Enzyme kernels. Higher derivatives
require additional rules and validation.

## Derivative meaning and validation

A fixed-graph derivative holds topology and discrete modes constant. A step
derivative must cover the actual numerical update, including its stages and
state mapping. A trajectory derivative additionally requires composition
across committed steps. Hybrid derivatives may require event-time and reset
sensitivities; branchwise AD alone does not supply them.

Validate changed boundaries with analytic examples, directional finite
differences, the identity `wᵀ(Jv) = vᵀ(Jᵀw)`, and composition cases involving
shared inputs. Check implicit derivatives by perturbing and resolving.
Deterministic replay initially means a pinned build and platform with stable
execution and reduction order. Allocation bounds, parallel reproducibility
and cross-platform bitwise equality need separate evidence.
