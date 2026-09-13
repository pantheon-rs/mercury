# Mercury

Differentiable numerical operators for simulation and optimization in Rust.

Mercury compiles numerical kernels with experimental Rust/Enzyme and connects
them in immutable runtime plans. The host owns simulation state and time.

- Enzyme compiles each numerical block's value, JVP, and VJP entry points.
- Mercury propagates derivatives through shared inputs and runtime connections.
- faer supplies factorizations behind explicit linear and implicit solve rules.

Start with a function and its gradient:

```sh
./scripts/example.sh rosenbrock
```

For `f(x, y) = (1 - x)² + 100(y - x²)²`, at `(-1.2, 1)` this gives
`f = 24.2` and `gradient = [-215.6, -88]`.
The [complete example](examples/rosenbrock.rs) fits in one file.

Continue through the [small examples](examples/README.md): arithmetic,
elementary functions, derivative products, Jacobians, composition, and solves.

Configuration is inactive. Put every value whose derivative you need in `q`.
The macro also supports dimensions taken from configuration and
`.with_domain(validator)` for checks outside differentiated code.

Linearizations expose values, JVPs, VJPs, contiguous batches, and row-major dense
Jacobians. A plan is itself an operator, so residual groups can be composed and
wrapped in `ImplicitSolve`. Numerical failures require fresh preparation.

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

The advanced [flight example](examples/flight.rs) composes a planar RK4 step and terminal
objective, then computes trajectory sensitivities with checkpoint replay.

The current scope is first order and dense solves. Batches use scalar loops;
native batch acceleration, sparse assembly, exact second derivatives, and
state-triggered event sensitivities remain future work. Newton is undamped and
requires a suitable initial guess. No allocation or performance bound is claimed.

The previous implementation is retained in Git history at `58d2f49`.
