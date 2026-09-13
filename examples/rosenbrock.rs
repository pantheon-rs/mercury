#![feature(autodiff)]

//! Rosenbrock: f(x, y) = (1 - x)² + 100(y - x²)². Compute its value and gradient.

use mercury::{Plan, Source, differentiable};

#[differentiable(inputs = 2, outputs = 1)]
fn rosenbrock(_config: &(), input: &[f64], output: &mut [f64]) {
    let x = input[0];
    let y = input[1];
    output[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let function = builder.add(
        rosenbrock_operator(()),
        [Source::Input(0), Source::Input(1)],
    );
    let plan = builder.build([function.output(0)])?;
    let mut workspace = plan.workspace();
    let point = [-1.2, 1.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;

    // For a scalar output, a VJP with seed 1 is the gradient.
    let mut gradient = [0.0; 2];
    linearization.vjp(&[1.0], &mut gradient)?;
    let value = linearization.value()?[0];

    println!("f(x, y) = (1 - x)^2 + 100(y - x^2)^2");
    println!("At (x, y) = (-1.2, 1.0):");
    println!("f = {value:.6}                 expected: 24.2");
    println!(
        "gradient = [{:.6}, {:.6}]  expected: [-215.6, -88]",
        gradient[0], gradient[1]
    );
    assert!((value - 24.2).abs() < 1e-12);
    assert!((gradient[0] + 215.6).abs() < 1e-10);
    assert!((gradient[1] + 88.0).abs() < 1e-10);
    Ok(())
}
