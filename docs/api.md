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

`function(Name)` names the generated type. `new()` constructs a stateless handle;
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

Arguments are active `f64` values. A scalar gradient follows argument order.

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
The returned `Gradient` and `Jacobian` handles borrow the plan; results are owned.

## Build a graph

`Plan::builder(n)` creates a `PlanBuilder` with `n` global inputs. `add` connects
an operator and returns a `NodeId`. `node.output(i)` selects one of its outputs.
`build` selects the graph outputs, validates connections, and consumes the builder.

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
bound is promised. The current derivative contract is first order.

Workspaces, linearizations, JVPs, VJPs and slice-kernel authoring live in
`mercury::advanced`. They are optional controls for solver and kernel authors.
