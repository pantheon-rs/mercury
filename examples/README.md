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
| [jvp](jvp.rs) | Change in outputs along an input direction | `J * [1, 2] = [4, 7]` |
| [vjp](vjp.rs) | Gradient of a weighted sum of outputs | `Jᵀ * [1, 2] = [10, 4]` |
| [jacobian](jacobian.rs) | Every output's derivative with respect to every input | `J = [[4, 0], [3, 2]]` |
| [batches](batches.rs) | Several JVPs/VJPs at the same point | Three seed rows, three result rows |
| [configuration](configuration.rs) | A fixed multiplier outside the active inputs | `2x²` at `x=3`: value `18`, derivative `12` |
| [composition](composition.rs) | Two connected functions with a shared input | `x²+x` at `x=3`: value `12`, derivative `7` |
| [linear_solve](linear_solve.rs) | Solve `Ax=b`; differentiate with respect to `A` and `b` | Solution `[1, 2]`; `db=[1, 0]` gives `dx=[0.4, -0.2]` |
| [implicit_solve](implicit_solve.rs) | Solve `z²-q=0`; differentiate the positive root | At `q=4`: `z=2`, `dz/dq=0.25` |

Replace `rosenbrock` in the command with any name above. Use
`./scripts/example.sh --list` to list all targets.

The derivative-product examples deliberately use the same function,
`f(x,y) = [x², xy]` at `(2,3)`, so only the requested operation changes.
A scalar-output VJP with seed `1` is the ordinary gradient.

The repeated setup creates a kernel, wires its inputs and outputs into a plan,
and calls `linearize` at one point. Keep that point unchanged while using its
linearization. Configuration is fixed; values to differentiate belong in the
input array. Scalar math examples use ordinary Rust `f64` operations.

[Flight](flight.rs) is the later integration example: RK4, held inputs, and
trajectory derivatives with checkpoint replay.
