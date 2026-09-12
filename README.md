# Mercury

Differentiable numerical operators for simulation and optimization in Rust.

Mercury compiles numerical kernels with experimental Rust/Enzyme and connects
them in immutable runtime plans. The host owns simulation state and time.

- Enzyme compiles each numerical block's value, JVP, and VJP entry points.
- Mercury propagates derivatives through shared inputs and runtime connections.
- faer supplies factorizations behind explicit linear and implicit solve rules.

```rust
#![feature(autodiff)]
use mercury::{Plan, Source, differentiable};

#[differentiable(inputs = 2, outputs = 1)]
fn energy(scale: &f64, q: &[f64], y: &mut [f64]) {
    y[0] = scale * (q[0] * q[0] + q[1] * q[1]);
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let energy = builder.add(energy_operator(0.5), [Source::Input(0), Source::Input(1)]);
    let plan = builder.build([energy.output(0)])?;
    let mut workspace = plan.workspace();
    let point = [3.0, 4.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let mut gradient = [0.0; 2];
    linearization.vjp(&[1.0], &mut gradient)?;
    assert_eq!(gradient, [3.0, 4.0]);
    Ok(())
}
```

Configuration is inactive. Put every value whose derivative you need in `q`.
The macro also supports dimensions taken from configuration and
`.with_domain(validator)` for checks outside differentiated code.

Linearizations expose values, JVPs, VJPs, contiguous batches, and row-major dense
Jacobians. A plan is itself an operator, so residual groups can be composed and
wrapped in `ImplicitSolve`. Numerical failures require fresh preparation.

Start with [the architecture](docs/architecture.md). [Validation](docs/validation.md)
records the supported toolchain, checks, and earlier compiler findings.

## Development

The pinned environment targets `x86_64-linux` and uses release builds with fat LTO.

```sh
nix develop
./scripts/build.sh
./scripts/ci.sh
cargo run --release --example flight
```

The [flight example](examples/flight.rs) composes a planar RK4 step and terminal
objective, then computes trajectory sensitivities with checkpoint replay.

The current scope is first order and dense solves. Batches use scalar loops;
native batch acceleration, sparse assembly, exact second derivatives, and
state-triggered event sensitivities remain future work. Newton is undamped and
requires a suitable initial guess. No allocation or performance bound is claimed.

The previous implementation is retained in Git history at `58d2f49`.
