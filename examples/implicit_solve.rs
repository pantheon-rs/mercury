#![feature(autodiff)]

//! Solve z² - q = 0 and differentiate its positive root.

use mercury::{ImplicitSolve, Plan, Result, function};

#[function(Residual)]
fn residual(z: f64, q: f64) -> f64 {
    // Residual arguments put the unknown z before the active parameter q.
    z * z - q
}

fn main() -> Result<()> {
    // A positive initial guess selects the positive square root.
    let solve = ImplicitSolve::new(Residual::new(), vec![1.0], 1.0e-12, 20)?;
    let function = Plan::from_operator(solve)?;
    let q = [4.0];
    let (z, gradient) = function.value_and_gradient(&q)?;
    println!("Solve z² - q = 0 at q = 4: positive root z = {z:.6}");
    assert!((z - 2.0).abs() < 1.0e-12);

    // Differentiate the residual equation: 2*z*dz = dq.
    println!("dz/dq = {:.6} (expected 1 / (2*z) = 0.25)", gradient[0]);
    assert!((gradient[0] - 0.25).abs() < 1.0e-12);
    Ok(())
}
