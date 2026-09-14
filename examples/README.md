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
or [the advanced reference](../docs/advanced.md). The parameter-products example introduces JVP/VJP; checkpoint replay remains
in the numerical regression fixture and execution benchmark.

## Loops and conditionals: increasing complexity

These examples use ordinary Rust control flow inside the numerical kernel.
Run them in order:

```sh
./scripts/example.sh loop_power
./scripts/example.sh conditional_abs
./scripts/example.sh loop_conditionals
./scripts/example.sh runtime_loop
./scripts/example.sh while_loop
./scripts/example.sh loop_flight
```

| Step | Example | New idea | Checked result |
| --- | --- | --- | --- |
| 1 | [loop_power](loop_power.rs) | Fixed `for` loop and a mutable accumulator | At `x=2`, `x³=8`, derivative 12, second derivative 12 |
| 2 | [conditional_abs](conditional_abs.rs) | `if/else` selected by an active input | `abs(x)` has slope -1 or +1 away from zero |
| 3 | [loop_conditionals](loop_conditionals.rs) | Branch inside an array loop | Positive-only squared penalty: value 10, gradient `[0, 0, 2, 6]` |
| 4 | [runtime_loop](runtime_loop.rs) | Inactive runtime iteration count in a configured kernel | Four decay steps: value `0.1875`, gradient `[0.0625, 1.5]` |
| 5 | [while_loop](while_loop.rs) | Active stopping condition and explicit iteration cap | Increment `0.375`: three additions, total `1.125`, derivative 3 |
| 6 | [loop_flight](loop_flight.rs) | Runtime rollout with scheduled thrust and state-dependent signed drag | Analytic Euler reference, every Jacobian entry by finite differences, and JVP/VJP adjoint identity |

### What the derivative means

A fixed integer loop count or a count in inactive configuration is not an active
numerical input. In `runtime_loop`, Mercury differentiates the initial state and
decay factor for the selected number of steps. The same compiled kernel is used
for zero, one, or four steps.

An active conditional selects the executed branch. Its derivative agrees with
the mathematical derivative where the function is smooth; branch selection alone
does not prove smoothness. `abs(x)` has no derivative at zero. In contrast,
`max(x,0)²` has derivative zero there because the one-sided slopes agree, although
its second derivative is undefined. The examples request only justified orders.

The `while_loop` example differentiates the executed finite sequence. For inputs
whose stopping decisions stay unchanged locally, its derivative is the number
of additions. At increment `0.25`, a small decrease adds another iteration and
the returned value jumps. The example evaluates this boundary but does not ask
for its derivative. The iteration cap ensures bounded execution even when the
threshold is never reached. This is differentiation of an algorithm; it is not
implicit differentiation of a converged root like `ImplicitSolve`.

The flight example differentiates explicit Euler updates over a fixed time horizon.
Thrust cutoff is an inactive step schedule; drag depends on active velocity and
opposes either sign. Initial state, thrust, mass and drag coefficient are active.
The host retains ownership of time and state; the kernel returns a proposed final
state and performs no external mutation. There is no ground-contact event in this
rollout. State-triggered event-time sensitivity is a separate concern illustrated
by [impact](impact.rs).

The configured slice examples supply first-order products. `loop_power` also
checks second derivatives; the other typed examples explicitly select
`first_order`. These are compiler-tested examples, not a guarantee that every
Rust iterator, container, loop body or nonsmooth expression can be differentiated.

## Preparation, diagnostics and aerospace boundaries

| Example | What it shows | Answer to look for |
| --- | --- | --- |
| [first_order](first_order.rs) | Compile first derivatives without nested autodiff; inspect capabilities | Table slopes 1 and 2 inside separate cells |
| [preparation](preparation.rs) | Prune unused nodes, request structure on demand, reuse curvature scratch | Identity JVP 2 and curvature 0 |
| [scaled_root](scaled_root.rs) | Residual/state scales and Newton reports | Root 2, derivative 0.25 despite residual scale `1e-12` |
| [solve_report](solve_report.rs) | Optional backward-error and conditioning diagnostics | Exact solutions can have very different reciprocal conditions |
| [parameter_products](parameter_products.rs) | Implicit products without materializing parameter Jacobians | Four-parameter direction gives 1; each partial is 0.25 |
| [attitude](attitude.rs) | Quaternion storage versus three local attitude coordinates | Body x-force changes world y/z under local z/y perturbations |
| [impact](impact.rs) | Compose a selected event-time root and reset | Post-impact velocity 5, altitude sensitivity 0.5 |

These aerospace examples define their units, frames, coordinates, and smooth
branches explicitly. They demonstrate consumers of Mercury's existing operators;
they are not replacements for host-owned state, event scheduling, or physical types.
