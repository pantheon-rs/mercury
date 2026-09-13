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
