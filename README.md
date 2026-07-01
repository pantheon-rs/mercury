# Mercury

`mercury` is the differentiable math substrate for `pantheon-rs`.

The Phase 1 direction is a small, plain-`f64` core differentiated by Rust
nightly `std::autodiff` / Enzyme. This is the Metis idea reduced to the part
that matters first: model code is written once as ordinary numeric Rust, and
Mercury owns the derivative entry points and validation surface.

Phase 1 owns:

- plain `f64` model-kernel conventions
- Enzyme-backed dense derivative evaluators
- finite-difference and analytic derivative checks
- a conservative AD-safe kernel subset
- room for sparse derivative callbacks later

It does not start with a generic scalar trait, a symbolic graph engine, a solver
stack, or a full linear algebra facade. Sparsity, graph coloring, and
optimization-facing callbacks are designed when real problem scale demands
them, without changing ordinary model code into a symbolic DSL.

## Development

```text
nix develop
./scripts/build.sh
./scripts/test.sh
./scripts/ci.sh
```

## Documentation

- [Architecture](docs/architecture.md)
- [Phase 1 Enzyme-backed `f64` decision](docs/decisions/0002-phase-1-enzyme-f64-core.md)
- [Phase 1 gradient validation implementation plan](docs/implementation-plans/phase-1-gradient-validation.md)
- [Decisions](docs/decisions/)
