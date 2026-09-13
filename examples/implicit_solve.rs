#![feature(autodiff)]

//! Solve z² - q = 0 and differentiate its positive root.

use mercury::{ImplicitSolve, Plan, Result, Source, differentiable};

#[differentiable(inputs = 2, outputs = 1)]
fn residual(_config: &(), input: &[f64], output: &mut [f64]) {
    // Residual inputs put the unknown z before the active parameter q.
    let z = input[0];
    let q = input[1];
    output[0] = z * z - q;
}

fn main() -> Result<()> {
    // A positive initial guess selects the positive square root.
    let solve = ImplicitSolve::new(residual_operator(()), vec![1.0], 1.0e-12, 20)?;
    let mut builder = Plan::builder(1);
    let root = builder.add(solve, [Source::Input(0)]);
    let plan = builder.build([root.output(0)])?;
    let mut workspace = plan.workspace();

    let q = [4.0];
    let mut linearization = plan.linearize(&q, &mut workspace)?;
    let z = linearization.value()?[0];
    println!("Solve z² - q = 0 at q = 4: positive root z = {z:.6}");
    assert!((z - 2.0).abs() < 1.0e-12);

    // Differentiate the residual equation: 2*z*dz = dq.
    let mut gradient = [0.0];
    linearization.vjp(&[1.0], &mut gradient)?;
    println!("dz/dq = {:.6} (expected 1 / (2*z) = 0.25)", gradient[0]);
    assert!((gradient[0] - 0.25).abs() < 1.0e-12);
    Ok(())
}
