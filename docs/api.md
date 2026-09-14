# Mercury API

Define mathematics, evaluate it, and request derivatives. These are the four
calculation operations for ordinary users:

| Operation | Meaning |
| --- | --- |
| `eval(...)` | Compute values |
| `gradient().eval(...)` | Differentiate one scalar output |
| `jacobian().eval(...)` | Differentiate every output against every input |
| `value_and_gradient(...)` | Compute one scalar value and its gradient together |

## Scalar functions

`function(Name)` annotates a module-level function and names the generated type. `new()` constructs a stateless handle;
compilation happens during the build. The original Rust function stays callable.

```rust
#![feature(autodiff)]

#[mercury::function(Square)]
fn square(x: f64) -> f64 { x * x }

fn main() -> mercury::Result<()> {
    let function = Square::new();
    assert_eq!(function.eval(3.0)?, 9.0);
    let gradient = function.gradient();
    assert_eq!(gradient.eval(3.0)?, [6.0]);
    assert_eq!(function.value_and_gradient(3.0)?, (9.0, [6.0]));
    Ok(())
}
```

Scalar arguments are active `f64` values; array arguments are described below.
A scalar gradient follows argument order.

## Vector functions

A function returning `[f64; N]` exposes `jacobian()`. Rows correspond to outputs;
columns correspond to arguments. Array length must be a positive integer literal.

```rust
#![feature(autodiff)]

#[mercury::function(Pair)]
fn pair(x: f64, y: f64) -> [f64; 2] { [x * x, x * y] }

fn main() -> mercury::Result<()> {
    let function = Pair::new();
    assert_eq!(function.eval(2.0, 3.0)?, [4.0, 6.0]);
    let jacobian = function.jacobian();
    assert_eq!(jacobian.eval(2.0, 3.0)?, [[4.0, 0.0], [3.0, 2.0]]);
    Ok(())
}
```

## One operator as a function

`Plan::from_operator` makes a solve or configured kernel directly callable.
`DenseSolve::new(n)` solves `Ax=b`; its inputs contain row-major `A`, then `b`.
This example solves `2x=6`.

```rust
use mercury::{DenseSolve, Plan};

let function = Plan::from_operator(DenseSolve::new(1)?)?;
assert_eq!(function.eval(&[2.0, 6.0])?, vec![3.0]);
let gradient = function.gradient();
assert_eq!(gradient.eval(&[2.0, 6.0])?, vec![-1.5, 0.5]);
assert_eq!(function.value_and_gradient(&[2.0, 6.0])?, (3.0, vec![-1.5, 0.5]));
let jacobian = function.jacobian().eval(&[2.0, 6.0])?;
assert_eq!(jacobian[(0, 1)], 0.5); // dx/db
# Ok::<(), mercury::Error>(())
```

A plan has runtime dimensions. `eval` returns `Vec<f64>` even for one output;
`gradient().eval` returns `Vec<f64>`; `jacobian().eval` returns `faer::Mat<f64>`,
indexed by `(row, column)`. `value_and_gradient` returns `(f64, Vec<f64>)`.
Gradient operations require exactly one published output, checked at evaluation.
The returned `Gradient` and `Jacobian` handles own a shared copy of the immutable plan; results are owned.

## Build a graph

`Plan::builder(n)` creates a `PlanBuilder` with `n` global inputs. `add` connects
an operator and returns a `NodeId`. `node.output(i)` selects one of its outputs.
`build` selects the graph outputs, validates all submitted connections and cycles,
and consumes the builder. It then removes nodes unreachable from the published
outputs and compacts workspace storage. Removed nodes never execute and do not
contribute domain failures or derivative requirements. A node is retained as a
whole if any of its outputs is needed; pruning does not use sampled zeros.
See [`preparation.rs`](https://github.com/pantheon-rs/mercury/blob/main/examples/preparation.rs).

```rust
use mercury::{DenseSolve, Plan, Source};

let mut builder = Plan::builder(2);
let node = builder.add(DenseSolve::new(1)?, [Source::Input(0), Source::Input(1)]);
let output = node.output(0);
assert_eq!(output, Source::Node(node, 0)); // Equivalent explicit spelling.
let function = builder.build([output])?;
assert_eq!(function.eval(&[2.0, 6.0])?, vec![3.0]);
# Ok::<(), mercury::Error>(())
```

`connect` replaces a node's inputs before building. Here it changes `b/a` to `a/b`:

```rust
use mercury::{DenseSolve, Plan, Source};

let mut builder = Plan::builder(2);
let node = builder.add(DenseSolve::new(1)?, [Source::Input(0), Source::Input(1)]);
builder.connect(node, [Source::Input(1), Source::Input(0)])?;
let function = builder.build([node.output(0)])?;
assert_eq!(function.eval(&[6.0, 2.0])?, vec![3.0]);
# Ok::<(), mercury::Error>(())
```

## Implicit roots

`ImplicitSolve::new(residual, initial_guess, tolerance, iteration_limit)` solves
`R(z,q)=0`. Residual arguments put unknowns `z` first, then parameters `q`.
A positive initial guess selects the positive root of `z²-q=0` here:

```rust
#![feature(autodiff)]

#[mercury::function(Residual)]
fn residual(z: f64, q: f64) -> f64 { z * z - q }

fn main() -> mercury::Result<()> {
    let root = mercury::ImplicitSolve::new(Residual::new(), vec![1.0], 1e-12, 20)?;
    let function = mercury::Plan::from_operator(root)?;
    let (value, gradient) = function.value_and_gradient(&[4.0])?;
    assert!((value - 2.0).abs() < 1e-12);
    assert!((gradient[0] - 0.25).abs() < 1e-12);
    Ok(())
}
```

## Errors and ownership

Every calculation returns `mercury::Result<T>`: an owned result or `mercury::Error`.
Use `?` to propagate errors. Invalid inputs, nonfinite numerical results and
failed solves never become successful results. Calls are independent after failure.

```rust
use mercury::{Error, Plan, Source};

let function = Plan::builder(1).build([Source::Input(0)])?;
assert!(matches!(function.eval(&[]), Err(Error::Dimension { .. })));
assert_eq!(function.eval(&[f64::NAN]), Err(Error::NonFinite("point")));
assert_eq!(function.eval(&[3.0])?, vec![3.0]);
# Ok::<(), mercury::Error>(())
```

Bodies must be deterministic and differentiable at the requested point. Panics
are not caught; finite checks do not prove differentiability or compiler correctness.

Typed calls use local fixed-size buffers. Plan calls allocate their own workspace
and results. Separate value and derivative calls can repeat primal work; plan
`value_and_gradient` also permits kernel replay. No allocation or execution-time
bound is promised. Second-order support and its limits are described below.

Workspaces, linearizations, JVPs, VJPs and slice-kernel authoring live in
`mercury::advanced`. They are optional controls for solver and kernel authors.

## Array arguments

Accept vectors as `[f64; N]` and matrices as `[[f64; C]; R]`. Dimensions are
positive integer literals. If any argument is an array, derivatives have fields
named after the arguments, preserving each argument's shape.

```rust
#![feature(autodiff)]
use mercury::function;
#[function(Energy)]
fn energy(velocity: [f64; 2], mass: f64) -> f64 {
    0.5 * mass * (velocity[0] * velocity[0] + velocity[1] * velocity[1])
}
# fn main() -> mercury::Result<()> {
let function = Energy::new();
let (value, gradient) = function.value_and_gradient([3.0, 4.0], 2.0)?;
assert_eq!(value, 25.0);
assert_eq!(gradient.velocity, [6.0, 8.0]);
assert_eq!(gradient.mass, 12.5);
assert_eq!(function.gradient().eval([3.0, 4.0], 2.0)?, gradient);
# Ok(())
# }
```

For a vector or matrix output, each Jacobian field contains one argument-shaped
partial derivative per scalar output row:

```rust
#![feature(autodiff)]
use mercury::function;
#[function(Transform)]
fn transform(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> [f64; 2] {
    [matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
     matrix[1][0] * vector[0] + matrix[1][1] * vector[1]]
}
# fn main() -> mercury::Result<()> {
let matrix = [[2.0, 3.0], [5.0, 7.0]];
let jacobian = Transform::new().jacobian().eval(matrix, [11.0, 13.0])?;
assert_eq!(jacobian.vector, matrix);
assert_eq!(jacobian.matrix[0], [[11.0, 13.0], [0.0, 0.0]]);
# Ok(())
# }
```

All arguments remain active. Reading `.mass` selects a result field; it does
not request a cheaper partial evaluation. Plans flatten arguments in declaration
order and arrays in row-major order. Matrix outputs use the same row-major order.

## Sparse Jacobians

Use `jacobian().eval_sparse(point)` for an owned faer CSC matrix.
`jacobian().sparsity()` returns the fixed pattern without evaluating the function.

```rust
#![feature(autodiff)]
use mercury::{Plan, Source, function};
#[function(Square)]
fn square(x: f64) -> f64 { x * x }
# fn main() -> mercury::Result<()> {
let mut builder = Plan::builder(2);
let first = builder.add(Square::new(), [Source::Input(0)]);
let second = builder.add(Square::new(), [Source::Input(1)]);
let function = builder.build([first.output(0), second.output(0)])?;
let jacobian = function.jacobian();
assert_eq!(jacobian.sparsity().row_idx(), &[0, 1]);
assert_eq!(jacobian.eval_sparse(&[2.0, 3.0])?.val(), &[4.0, 6.0]);
assert_eq!(jacobian.eval_sparse(&[0.0, 3.0])?.val(), &[0.0, 6.0]);
# Ok(())
# }
```

The plan owns the pattern and column coloring. Dense and sparse Jacobians use
one forward product per color. Structural entries remain stored when their
current value is zero. Kernels default to dense local dependencies; graph
composition and explicit operator contracts supply sparsity. Mercury does not
infer expression-level sparsity from Enzyme. Sparse solves and symbolic
factorization caching are not provided by Mercury yet.

## Second derivatives

The Jacobian of a scalar function's gradient is its Hessian:

```rust
#![feature(autodiff)]
use mercury::{Plan, function};
#[function(Cube)]
fn cube(x: f64) -> f64 { x * x * x }
# fn main() -> mercury::Result<()> {
let function = Cube::new();
assert_eq!(function.gradient().jacobian().eval(2.0)?, [[12.0]]);
let plan = Plan::from_operator(function)?;
assert_eq!(plan.gradient().jacobian().eval(&[2.0])?[(0, 0)], 12.0);

// Derivatives can themselves be graph nodes.
let derivative = Plan::from_operator(plan.gradient())?;
assert_eq!(derivative.eval(&[2.0])?, [12.0]);
assert_eq!(derivative.jacobian().eval(&[2.0])?[(0, 0)], 12.0);
# Ok(())
# }
```

Both typed and plan gradient/Jacobian handles implement `Operator`. Jacobian
operators publish flattened rows, in output-then-input order. Plan derivative
handles retain shared ownership of the immutable plan; dropping the original
plan is safe. Cloning a plan shares its structure, never its workspace.

Typed Hessians return arrays; plan Hessians return faer matrices. Hessian axes
follow flattened input order, including for structured arguments. Compiled
functions use Enzyme forward-over-reverse; plans apply the second-order chain
rule. Linear solves use analytic curvature rules; implicit solves require a
second-order residual and a converged, locally regular root. Advanced slice
kernels need an explicit curvature callback. Missing rules return
`Error::UnsupportedDerivative`, with the failing node attached by a plan.
Third derivatives are unsupported. No finite-difference fallback is used.

## First-order functions

`#[function(Name, first_order)]` generates value, JVP and VJP entry points, and
omits nested autodiff and the typed Hessian API. The default remains second order.
This is useful for a kernel whose first-order compiler path has been validated
independently of its curvature path. It does not change the function's domain.

```rust
#![feature(autodiff)]
use mercury::advanced::Operator;
#[mercury::function(Square, first_order)]
fn square(x: f64) -> f64 { x * x }
# fn main() -> mercury::Result<()> {
let function = Square::new();
assert_eq!(function.gradient().eval(3.0)?, [6.0]);
assert_eq!(function.derivative_order(), 1);
assert_eq!(function.gradient().derivative_order(), 0);
# Ok(())
# }
```

A first-order function's gradient/Jacobian handle can evaluate derivative values,
but its own products return `UnsupportedDerivative`. For a typed scalar handle,
`gradient().jacobian()` is absent when `first_order` is selected. Runtime plans
retain their uniform API and report unsupported orders through `Result`.
The [table example](https://github.com/pantheon-rs/mercury/blob/main/examples/first_order.rs)
checks slopes on either side of an interpolation knot; it makes no derivative
claim at the knot itself.

## Scaled roots and reports

Newton uses one tolerance for two dimensionless infinity norms:

- Residual: `max_i abs(R[i]) / residual_scales[i]`.
- Correction: `max_i abs(dz[i]) / max(state_scales[i], abs(z[i]))`,
  where `R_z dz = -R` at the candidate point.

Both must meet the tolerance. Default scales are one. `with_scaling` accepts
positive finite characteristic magnitudes in the residuals' and unknowns' units;
it changes convergence tests, not the residual equations or matrix equilibration.
Multiplying an equation by `s > 0` requires multiplying its residual scale by `s`
to preserve the same stopping criterion. The correction check also prevents a
small raw residual alone from accepting a distant root.

```rust
#![feature(autodiff)]
#[mercury::function(Residual)]
fn residual(z: f64, q: f64) -> f64 { 1e-12 * (z*z - q) }
# fn main() -> mercury::Result<()> {
let solve = mercury::ImplicitSolve::new(Residual::new(), vec![1.0], 1e-10, 20)?
    .with_scaling(vec![1e-12], vec![2.0])?;
let (root, report) = solve.solve_with_report(&[4.0])?;
assert!((root[0] - 2.0).abs() < 1e-10);
assert!(report.residual_norm <= 1e-10);
assert!(report.correction_norm <= 1e-10);
# Ok(())
# }
```

`solve_with_report` allocates one workspace and returns values plus a
`NewtonReport`. `iterations` counts applied Newton updates; both norms refer to
the returned point. Plan execution uses the same acceptance rule and publishes
only the numerical values. On exhaustion, `Error::NonConvergence { report }`
contains diagnostics at the last evaluated iterate. Other errors keep their
original domain/factorization context, wrapped with node indices by plans.

An exactly zero floating-point residual permits value-only success without
factorization, with a reported correction norm of zero. Linearization always
requires nonsingular `R_z`, including at an exact root. These are convergence
diagnostics, not certified forward-error or sensitivity bounds. Scales do not
prevent underflow, overflow, ill-conditioning or convergence to an unintended root.
Newton remains undamped, restarts from its configured guess on every solve, and
requires a suitable basin. There is no hidden warm start.

Implicit JVP/VJP and curvature use the retained residual workspace at the
accepted point. Only `R_z` is assembled; parameter derivatives use residual
products, never a stored dense `R_q`. Each product can replay residual kernels
and can fail independently, invalidating the enclosing plan linearization.
See the [parameter-product example](https://github.com/pantheon-rs/mercury/blob/main/examples/parameter_products.rs).

## Linear solve reports

The optional diagnostic call measures normwise backward error and reciprocal
condition using infinity norms. It performs `n` extra solves with the prepared
LU factors to compute the inverse norm; it does not form a stored inverse.

```rust
let solve = mercury::DenseSolve::new(2)?;
let (value, report) = solve.solve_with_report(&[1.0, 0.0, 0.0, 1e-12, 1.0, 1e-12])?;
assert_eq!(value, vec![1.0, 1.0]);
assert!(report.backward_error < 1e-14);
assert!((report.reciprocal_condition - 1e-12).abs() < 1e-24);
# Ok::<(), mercury::Error>(())
```

A small backward error can accompany high sensitivity when reciprocal condition
is small. The consumer chooses an acceptable conditioning threshold. Ordinary
`DenseSolve` plan execution does not compute these diagnostics or apply such a
threshold. Diagnostic arithmetic can itself overflow and returns `NonFinite`;
no report is a certified error bound. See
[`solve_report.rs`](https://github.com/pantheon-rs/mercury/blob/main/examples/solve_report.rs).

## Compatibility in 0.5

Unreachable nodes no longer execute, and Newton acceptance now also checks the
correction. Code matching the old unit `Error::NonConvergence` must match
`Error::NonConvergence { report }` (or `{ .. }`). Custom second-order operators
should override `Operator::derivative_order()` to return 2; its default is 1.
Existing function attributes retain second-order generation by default.
