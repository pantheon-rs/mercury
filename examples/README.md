# Small examples

Start here:

```sh
./scripts/example.sh rosenbrock
```

Each example is a standalone file with a small function, a calculation, printed
results, and assertions against known answers. Run them in this order:

| Example | What it shows | Answer to look for |
| --- | --- | --- |
| [rosenbrock](rosenbrock.rs) | Function value and gradient | At `(-1.2, 1)`: `24.2`, gradient `[-215.6, -88]` |
| [arithmetic](arithmetic.rs) | `+`, `-`, `*`, `/` and both partial derivatives | For `x/y` at `(6, 2)`: `3`, gradient `[0.5, -1.5]` |
| [elementary](elementary.rs) | Square, square root, `exp`, `ln`, `sin`, `cos` | At `x=1`: square derivative `2`, square-root derivative `0.5` |
| [jacobian](jacobian.rs) | Every output's derivative with respect to every argument | `J = [[4, 0], [3, 2]]` |
| [composition](composition.rs) | Two connected functions with a shared input | `x²+x` at `x=3`: value `12`, derivative `7` |
| [linear_solve](linear_solve.rs) | Solve `Ax=b`; differentiate with respect to `A` and `b` | Solution `[1, 2]`; `db=[1, 0]` gives `dx=[0.4, -0.2]` |
| [implicit_solve](implicit_solve.rs) | Solve `z²-q=0`; differentiate the positive root | At `q=4`: `z=2`, `dz/dq=0.25` |

Replace `rosenbrock` in the command with any name above. Use
`./scripts/example.sh --list` to list all targets.

The examples use ordinary arguments and owned results. Scalar functions expose
`gradient()`; vector functions expose `jacobian()`. Plans support both, with
runtime validation of scalar output count.

[Flight](flight.rs) composes 100 vertical-flight steps and requests its final
state and Jacobian. Configured slice kernels have a complete example in the
advanced reference.

Every public operation has a short example in [the API guide](../docs/api.md)
or [the advanced reference](../docs/advanced.md). JVP/VJP and checkpoint replay
remain in the numerical regression tests rather than introductory examples.
