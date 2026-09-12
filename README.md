# Mercury

Differentiable numerical operators for simulation and optimization in Rust.

Mercury is restarting from a pinned experimental Rust/Enzyme scaffold. There
is no public math API yet. The smoke tests exercise compiled forward and
reverse derivatives of an ordinary `f64` function.

The architecture is fixed around three responsibilities:

- Enzyme compiles each numerical block's value, JVP, and VJP entry points.
- Mercury composes those operators across runtime-selected connections.
- faer supplies dynamic matrices and factorizations behind explicit derivative
  rules. It will be the only general linear algebra dependency; none is needed
  by the scaffold.

Start with [the architecture](docs/architecture.md). [Validation](docs/validation.md)
records the supported toolchain, checks, and earlier compiler findings.

## Development

The pinned environment targets `x86_64-linux` and uses release builds with fat LTO.

```sh
nix develop path:.
./scripts/build.sh
./scripts/ci.sh
```

## Next implementation

1. Expose a vector-valued operator with reusable value/JVP/VJP buffers.
2. Compose several operators with shared inputs and verify their derivatives.
3. Add one faer-backed solve operator and an aerospace linearization example.

The previous implementation is retained in Git history at `58d2f49`.
