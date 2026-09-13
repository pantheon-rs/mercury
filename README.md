# Mercury

Differentiable Rust functions for simulation and optimization.

Mercury compiles numerical kernels with experimental Rust/Enzyme and connects
them in immutable runtime plans. The host owns simulation state and time.

- Enzyme compiles each numerical block's value, JVP, and VJP entry points.
- Mercury propagates derivatives through shared inputs and runtime connections.
- faer supplies factorizations behind explicit linear and implicit solve rules.

Define a function, select its gradient, and evaluate:

```rust
#![feature(autodiff)]

#[mercury::function(Rosenbrock)]
fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0 - x;
    let b = y - x * x;
    a * a + 100.0 * b * b
}

fn main() -> mercury::Result<()> {
    let function = Rosenbrock::new();
    let gradient = function.gradient();

    let value = function.eval(-1.2, 1.0)?;
    let [df_dx, df_dy] = gradient.eval(-1.2, 1.0)?;
    println!("value = {value}, gradient = [{df_dx}, {df_dy}]");
    Ok(())
}
```

Run it with the pinned compiler:

```sh
./scripts/example.sh rosenbrock
```

For `f(x, y) = (1 - x)² + 100(y - x²)²`, at `(-1.2, 1)` this gives
`f = 24.2` and `gradient = [-215.6, -88]`.
The [complete example](examples/rosenbrock.rs) fits in one file.

The attribute names the generated type explicitly. `gradient()` selects a
compiled derivative; `eval` computes it at the supplied arguments. The gradient
follows argument order. To compute both in one combined reverse call, use
`function.value_and_gradient(x, y)?`.

Functions returning `[f64; N]` expose `function.jacobian().eval(...)`, returning
one row per output and one column per argument. All arguments are active `f64`
values. Calls return owned values and reject nonfinite inputs and results.
Function bodies must be deterministic and differentiable at the requested point;
finite checks cannot establish differentiability or catch panics.

Continue through the [small examples](examples/README.md): arithmetic,
elementary functions, Jacobians, composition, and solves.
See [the function API](docs/api.md) for exact call and storage contracts.

For graphs, pass the same function instance to `builder.add(function, sources)`.
The advanced `#[mercury::advanced::differentiable(inputs = n, outputs = m)]` interface
supports inactive configuration, runtime dimensions, and `.with_domain(validator)`.
Put every value whose derivative you need in its input slice.

Plans expose the same calculation names as typed functions and manage their own
scratch. Use [advanced execution](docs/advanced.md) only for explicit workspaces,
derivative products, and custom operators.

Read [the architecture](docs/architecture.md) for the design. [Validation](docs/validation.md)
records the supported toolchain, checks, and earlier compiler findings.

## Development

The pinned environment targets `x86_64-linux` and uses release builds with fat LTO.
Scripts enter it automatically.

```sh
./scripts/example.sh --list
./scripts/example.sh rosenbrock
./scripts/ci.sh
```

Pass example arguments with `./scripts/example.sh NAME -- ARGS...`.
For direct Cargo commands, enter `nix develop` and use `--release`.

The [flight example](examples/flight.rs) composes 100 vertical-flight steps and
evaluates the final state and its Jacobian. Checkpointed planar RK4 remains
covered by the trajectory regression tests.

Sparse Jacobians, structured array arguments, and second derivatives are supported.
Try `scripts/example.sh sparse`, `structured`, or `hessian`. Batches use scalar
loops; native batch acceleration, sparse solve operators, third derivatives, and
state-triggered event sensitivities remain future work. Newton is undamped and
requires a suitable initial guess. No allocation or performance bound is claimed.

The previous implementation is retained in Git history at `58d2f49`.
