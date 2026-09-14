# Mercury

[![CI](https://github.com/pantheon-rs/mercury/actions/workflows/ci.yml/badge.svg)](https://github.com/pantheon-rs/mercury/actions/workflows/ci.yml)
[![Docs](https://github.com/pantheon-rs/mercury/actions/workflows/docs.yml/badge.svg)](https://github.com/pantheon-rs/mercury/actions/workflows/docs.yml)
[![Coverage: not reported](https://img.shields.io/badge/coverage-not_reported-lightgrey)](docs/validation.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Differentiable Rust functions for simulation and optimization. Mercury uses
experimental Rust/Enzyme to compile derivatives, connects functions in immutable
runtime plans, and differentiates through linear and implicit solves backed by
faer. Your application owns simulation state and time.

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
    let value = function.eval(-1.2, 1.0)?;
    let gradient = function.gradient().eval(-1.2, 1.0)?;
    println!("value = {value}, gradient = {gradient:?}");
    Ok(())
}
```

Result: `24.2`, with gradient `[-215.6, -88]` in argument order.
Run the [complete example](examples/rosenbrock.rs) with
`./scripts/example.sh rosenbrock`.

## Develop

Requires Nix with flakes on `x86_64-linux`. Run these from the repository root;
the scripts enter the pinned Rust/Enzyme environment and use release builds with
fat LTO automatically.

```sh
./scripts/build.sh                 # Build the workspace and all targets
./scripts/example.sh --list        # Discover examples
./scripts/example.sh rosenbrock    # Run one (NAME -- ARGS... passes arguments)
./scripts/test.sh                  # Tests, including doctests
./scripts/ci.sh                    # Format, lint, test, docs, dependency audit
./scripts/docs.sh                  # Generate target/doc/mercury/index.html
```

Use `./scripts/dev.sh` for an interactive shell, `./scripts/format.sh` to format,
and `./scripts/bench.sh` for execution benchmarks. Direct Cargo builds need
`--release` inside the dev shell.

## Source and docs

| Path | Contents |
| --- | --- |
| [src/](src/) | Kernels, runtime plans, derivative products, sparse Jacobians and solves |
| [macros/](macros/) | Function attributes and generated Enzyme adapters |
| [examples/](examples/README.md) | Small runnable examples, from arithmetic to flight |
| [tests/](tests/) / [benches/](benches/) | Numerical regressions and execution benchmarks |
| [scripts/](scripts/) / [nix/](nix/) | Developer commands and pinned build environment |

Read the [API guide](docs/api.md), [architecture](docs/architecture.md),
[advanced execution reference](docs/advanced.md), and
[validation and compiler limitations](docs/validation.md).
Coverage instrumentation is deferred; see validation for the Enzyme limitation.
