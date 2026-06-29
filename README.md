# Mercury

`mercury` is the generic differentiable math substrate for `pantheon-rs`.

It owns the core math contract that higher layers build on:

- numeric execution
- automatic differentiation execution
- symbolic tracing
- linear algebra facade
- sparsity and graph coloring
- derivative evaluators
- optimization-facing derivative contracts

The first implementation is intentionally small. The crate starts as a normal
Cargo library wrapped in a reproducible Nix workflow. AD, linalg, symbolic IR,
and optimization backends should be introduced behind Mercury-owned traits and
types rather than leaking backend crates directly through the architecture.

## Development

```text
nix develop
./scripts/build.sh
./scripts/test.sh
./scripts/ci.sh
```

## Documentation

- [Architecture](docs/architecture.md)
- [Decisions](docs/decisions/)
