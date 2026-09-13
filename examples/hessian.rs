#![feature(autodiff)]
//! The Jacobian of a gradient is the Hessian.

use mercury::{Plan, function};

#[function(Rosenbrock)]
fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
}

#[allow(
    clippy::float_cmp,
    reason = "These reference values are exactly representable."
)]
fn main() -> mercury::Result<()> {
    let gradient = Rosenbrock::new().gradient();
    let hessian = gradient.jacobian().eval(1.0, 1.0)?;
    println!("Hessian at (1, 1): {hessian:?}");
    assert_eq!(hessian, [[802.0, -400.0], [-400.0, 200.0]]);

    // A gradient can also be a node in a larger differentiable plan.
    let function = Plan::from_operator(gradient)?;
    assert_eq!(function.jacobian().eval(&[1.0, 1.0])?[(0, 0)], 802.0);
    Ok(())
}
