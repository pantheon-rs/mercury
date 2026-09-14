# Advanced execution

Use `mercury::advanced` when implementing a solver, reusing storage, applying
weighted derivatives, or authoring a configured slice kernel. Ordinary examples
use `eval`, `gradient`, and `jacobian` instead.

## Plan execution

Import `PlanExecution` to access `epoch`, `dependencies`, `evaluate`, and `linearize`.
`Workspace::new` creates storage for one plan. `evaluate` writes values into a
caller buffer. `linearize` evaluates a point and returns a borrowed derivative
handle; it neither compiles code nor constructs the full Jacobian.

```rust
use mercury::{Plan, Source};
use mercury::advanced::{PlanExecution, Workspace};

let plan = Plan::builder(1).build([Source::Input(0)])?; // f(x)=x
let epoch = plan.epoch();
assert_eq!(plan.epoch(), epoch);
assert_eq!(plan.dependencies(), &[vec![0]]);
let mut workspace = Workspace::new(&plan);
let mut value = [0.0];
plan.evaluate(&[3.0], &mut workspace, &mut value)?;
assert_eq!(value, [3.0]);
let point = [3.0];
let linearization = plan.linearize(&point, &mut workspace)?;
assert_eq!(linearization.value()?, &[3.0]);
# Ok::<(), mercury::Error>(())
```

## Linearization products

`jvp` computes `Jv`; `vjp` computes `Jᵀw`. Both overwrite the destination and
preserve the seed. Batches store each direction contiguously. `jacobian` writes
all entries in output-by-input row-major order.

```rust
use mercury::{Plan, Source};
use mercury::advanced::{PlanExecution, Workspace};

let plan = Plan::builder(2).build([Source::Input(1), Source::Input(0)])?;
let point = [2.0, 3.0]; // f(x,y)=[y,x], J=[[0,1],[1,0]]
let mut workspace = Workspace::new(&plan);
let mut linearization = plan.linearize(&point, &mut workspace)?;
let mut product = [0.0; 2];
linearization.jvp(&[1.0, 2.0], &mut product)?;
assert_eq!(product, [2.0, 1.0]);
linearization.vjp(&[3.0, 4.0], &mut product)?;
assert_eq!(product, [4.0, 3.0]);
let seeds = [1.0, 0.0, 0.0, 1.0]; // Two directions.
let mut batch = [0.0; 4];
linearization.jvp_batch(2, &seeds, &mut batch)?;
assert_eq!(batch, [0.0, 1.0, 1.0, 0.0]);
linearization.vjp_batch(2, &seeds, &mut batch)?;
assert_eq!(batch, [0.0, 1.0, 1.0, 0.0]);
linearization.jacobian(&mut batch)?;
assert_eq!(batch, [0.0, 1.0, 1.0, 0.0]);
# Ok::<(), mercury::Error>(())
```

The handle borrows its point and exclusively borrows the workspace. Numerical
failure invalidates it; prepare again before requesting further products.
Kernels may replay their primal. There is no retained reusable Enzyme tape.

## Configured kernels

`differentiable(inputs=n, outputs=m)` accepts inactive configuration, an input
slice, and an output slice. It creates `<name>_operator(config)`. Dimension
expressions may refer to configuration. `Kernel::with_domain` installs a check
outside differentiated code.

```rust
#![feature(autodiff)]
use mercury::{Error, Plan, Result};
use mercury::advanced::differentiable;

#[differentiable(inputs = 1, outputs = 1)]
fn scaled(scale: &f64, input: &[f64], output: &mut [f64]) {
    output[0] = *scale * input[0];
}

fn positive_scale(scale: &f64, _input: &[f64]) -> Result<()> {
    if *scale > 0.0 { Ok(()) } else { Err(Error::Domain("scale must be positive")) }
}

fn main() -> Result<()> {
    let kernel = scaled_operator(2.0).with_domain(positive_scale);
    let function = Plan::from_operator(kernel)?;
    assert_eq!(function.value_and_gradient(&[3.0])?, (6.0, vec![2.0]));
    Ok(())
}
```

## Operator contract

`Operator` supplies dimensions, conservative dependencies, and a local workspace.
`Shape` counts active inputs and outputs. A dependency may conservatively be
`true`; reporting `false` promises the output never depends on that input for
this instance. Numerical zeros at one point do not establish independence.

```rust
use mercury::DenseSolve;
use mercury::advanced::{Operator, Shape};

let operator = DenseSolve::new(1)?;
assert_eq!(operator.shape(), Shape { inputs: 2, outputs: 1 });
assert!(operator.depends_on(0, 1));
let mut workspace = operator.workspace();
let mut value = [0.0];
workspace.evaluate(&[2.0, 6.0], &mut value)?;
assert_eq!(value, [3.0]);
# Ok::<(), mercury::Error>(())
```

`OperatorWorkspace` is the implementer's boundary. Call `linearize` before
products, keep the point unchanged, and call `invalidate` after numerical failure.
The plan normally enforces this lifecycle. Custom implementations must obey the
buffer and failure contracts documented on the trait.

```rust
use mercury::DenseSolve;
use mercury::advanced::Operator;

let operator = DenseSolve::new(1)?;
let mut workspace = operator.workspace();
let point = [2.0, 6.0]; // f(a,b)=b/a
let mut value = [0.0];
workspace.linearize(&point, &mut value)?;
assert_eq!(value, [3.0]);
let mut tangent = [0.0];
workspace.jvp(&point, &[0.0, 1.0], &mut tangent)?;
assert_eq!(tangent, [0.5]);
let mut gradient = [0.0; 2];
workspace.vjp(&point, &[1.0], &mut gradient)?;
assert_eq!(gradient, [-1.5, 0.5]);
let mut tangents = [0.0; 2];
workspace.jvp_batch(&point, 2, &[1.0, 0.0, 0.0, 1.0], &mut tangents)?;
assert_eq!(tangents, [-1.5, 0.5]);
let mut gradients = [0.0; 4];
workspace.vjp_batch(&point, 2, &[1.0, 2.0], &mut gradients)?;
assert_eq!(gradients, [-1.5, 0.5, -3.0, 1.0]);
workspace.invalidate();
workspace.linearize(&point, &mut value)?; // Fresh preparation permits reuse.
# Ok::<(), mercury::Error>(())
```

## Manual kernel callbacks

`Kernel::new` accepts primal, forward and reverse callbacks. Most authors use
`differentiable`; this constructor exists for explicit derivative implementations.
Reverse callbacks accumulate into the input shadow. This complete identity
kernel shows the argument order without requiring Enzyme:

```rust
use mercury::Plan;
use mercury::advanced::{Kernel, Shape};

fn value(_: &(), x: &[f64], y: &mut [f64]) { y[0] = x[0]; }
fn forward(c: &(), x: &[f64], dx: &[f64], y: &mut [f64], dy: &mut [f64]) {
    value(c, x, y);
    dy[0] = dx[0];
}
fn reverse(c: &(), x: &[f64], dx: &mut [f64], y: &mut [f64], dy: &mut [f64]) {
    value(c, x, y);
    dx[0] += dy[0];
}
let kernel = Kernel::new((), Shape { inputs: 1, outputs: 1 }, value, forward, reverse);
let function = Plan::from_operator(kernel)?;
assert_eq!(function.value_and_gradient(&[3.0])?, (3.0, vec![1.0]));
# Ok::<(), mercury::Error>(())
```

## Migration

Backend types and the slice macro moved from `mercury::*` to
`mercury::advanced::*`. Replace `plan.workspace()` with `Workspace::new(&plan)`;
import `PlanExecution` for `evaluate`, `linearize`, `epoch`, and `dependencies`.
`Operator::shape()` remains available through the advanced trait.

The former JVP, VJP, batch and configuration tutorial targets were removed. Their operations
remain tested. The checkpointed planar-flight model now lives in
`tests/support/flight.rs`; the flight example demonstrates ordinary graph calls.

## Sparse storage

Reuse your numerical buffer with the plan's fixed CSC pattern:

```rust
use mercury::{Plan, Source};
use mercury::advanced::{PlanExecution, Workspace};
let plan = Plan::builder(2).build([Source::Input(1), Source::Input(0)])?;
let jacobian = plan.jacobian();
let mut values = vec![0.0; jacobian.sparsity().row_idx().len()];
let mut workspace = Workspace::new(&plan);
plan.linearize(&[2.0, 3.0], &mut workspace)?.sparse_jacobian(&mut values)?;
let matrix = faer::sparse::SparseColMatRef::new(jacobian.sparsity(), &values);
assert_eq!(matrix.val(), &[1.0, 1.0]);
# Ok::<(), mercury::Error>(())
```

This reuses the destination and structure. Product scratch may still allocate.
Rebuild the plan when topology or a dependency contract changes.

## Structural dependencies

A kernel may declare that particular output/input pairs are always independent:

```rust
#![feature(autodiff)]
use mercury::{Plan, advanced::differentiable};
#[differentiable(inputs = 2, outputs = 2)]
fn squares(_config: &(), input: &[f64], output: &mut [f64]) {
    output[0] = input[0] * input[0];
    output[1] = input[1] * input[1];
}
# fn main() -> mercury::Result<()> {
let kernel = squares_operator(()).with_dependencies(|(), row, column| row == column);
let plan = Plan::from_operator(kernel)?;
assert_eq!(plan.jacobian().eval_sparse(&[2.0, 3.0])?.val(), &[4.0, 6.0]);
# Ok(())
# }
```

`Operator::depends_on` carries the same contract. A false result promises
independence throughout the supported domain, including all branches. The
caller supplies this mathematical fact; Mercury cannot verify it. An incorrect
declaration can corrupt both dense and sparse assembled Jacobians. Keep the
default `true` when unsure.

## Second-order products

`curvature(weights, direction, output)` applies the Hessian of the fixed weighted
sum of outputs. The low-level workspace form also takes the prepared input.

```rust
#![feature(autodiff)]
use mercury::{Plan, advanced::{differentiable, Operator, PlanExecution, Workspace}};
#[differentiable(inputs = 1, outputs = 1)]
fn square(_config: &(), input: &[f64], output: &mut [f64]) {
    output[0] = input[0] * input[0];
}
# fn main() -> mercury::Result<()> {
let kernel = square_operator(()).with_curvature(|(), _input, weights, direction, output| {
    output[0] = 2.0 * weights[0] * direction[0];
});
{
    let mut workspace = kernel.workspace();
    workspace.linearize(&[3.0], &mut [0.0])?;
    let mut result = [0.0];
    workspace.curvature(&[3.0], &[2.0], &[4.0], &mut result)?;
    assert_eq!(result, [16.0]);
}
let plan = Plan::from_operator(kernel)?;
let mut workspace = Workspace::new(&plan);
let mut result = [0.0];
plan.linearize(&[3.0], &mut workspace)?.curvature(&[2.0], &[4.0], &mut result)?;
assert_eq!(result, [16.0]);
# Ok(())
# }
```

`#[function(Name)]` supplies this callback automatically. Manual callbacks must
overwrite every result and preserve inputs, weights and directions. Hessian
symmetry requires a twice differentiable function on the evaluated domain.
Custom workspaces without a curvature rule return `UnsupportedDerivative`.

## Derivative capabilities

`Operator::derivative_order()` declares the highest implemented order: 0 for
values, 1 for JVP/VJP, and 2 for weighted curvature. Custom operators default to
1; override it when providing curvature. A plan reports the minimum across its
reachable nodes (2 for an identity/empty graph). A derivative handle subtracts
one. This is structural metadata for preflight inspection, not a pointwise
smoothness check or a promise that numerical evaluation will succeed.

```rust
use mercury::{DenseSolve, Plan};
use mercury::advanced::Operator;
let plan = Plan::from_operator(DenseSolve::new(2)?)?;
assert_eq!(plan.derivative_order(), 2);
assert_eq!(plan.jacobian().derivative_order(), 1);
# Ok::<(), mercury::Error>(())
```

## Preparation and scratch ownership

Building a plan validates all wiring and cycles, prunes unreachable nodes and
assigns compact value slots. It does not call `depends_on`, propagate global
input dependencies, or color a Jacobian. `dependencies()` prepares the conservative
global dependency lists on first request. `jacobian().sparsity()` and Jacobian
assembly additionally prepare CSC structure and deterministic column coloring.
Each immutable cache is shared by cloned plan handles and initialized once;
these calls can allocate and should run before a latency-sensitive loop if needed.
Nested plans prepare dependency metadata when a caller requests their dependency
contract. Ordinary value, JVP, VJP and curvature execution do not request it.

A row depending on every input uses one color per column without constructing
all pairwise conflicts. Assembly chooses uncolored reverse rows when the number
of outputs is smaller than the forward color count; ties use colored forward
products. Dense and sparse plan assembly use the same strategy and preserve
structural zeros. Typed functions and derivative operators choose the smaller of
input-column and output-row counts. No runtime timing heuristic is involved.

Workspace-owned buffers retain capacity for Jacobian assembly and curvature.
Curvature scratch grows on first use; batching and differently shaped nested
operations can also grow buffers. Derivative and solve workspaces own their local
scratch. Numerical errors poison public outputs and invalidate numerical caches;
structural caches and reusable scratch capacity survive for fresh preparation.
Operator callbacks still must overwrite all outputs and preserve points/seeds.

This is not an allocation-free guarantee: Enzyme callbacks, faer factor/solve
operations, workspace creation, first-use preparation, and returned owned results
can allocate. The [preparation example](https://github.com/pantheon-rs/mercury/blob/main/examples/preparation.rs)
shows explicit reuse. `scripts/bench.sh` measures plan construction, reused
execution/products and checkpointed flight replay. Call-count regressions live
in `tests/preparation.rs` and `tests/solve.rs`; allocator-wide totals require a
separate profiler and are not reported by the timing harness.

## Aerospace boundary

Mercury differentiates explicit Euclidean coordinates. The host owns physical
units, frames, storage layouts, manifold charts, mode selection and event history.
Keep named physical values until a single documented packing boundary; do not
let independent consumers invent different index orders for the same state.

The [attitude example](https://github.com/pantheon-rs/mercury/blob/main/examples/attitude.rs)
rotates body-frame force in newtons into world coordinates. It explicitly
normalizes scalar-first quaternion storage and separately defines a three-angle
local chart, `normalize([1, delta/2])`, about identity. Its local Jacobian is with
respect to three perturbations in radians, not four stored quaternion entries.
This example is a consumer fixture; Mercury does not introduce a units, frames,
or manifold type system.

The [impact example](https://github.com/pantheon-rs/mercury/blob/main/examples/impact.rs)
uses an existing `ImplicitSolve` for a selected ground-crossing time and composes
it with a velocity reset. The result is velocity immediately after that event,
not a state sampled at a fixed terminal time. Differentiating event time matters:
freezing it would lose the altitude sensitivity. The host must select and validate
the branch (positive altitude/gravity, a suitable positive-time Newton basin,
and a transverse crossing). Grazing, changing event order, and general event
scheduling are outside the example's contract. A fixed-terminal-time trajectory
would also need post-event flow and its dependence on the remaining duration.
