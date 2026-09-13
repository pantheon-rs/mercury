#![feature(autodiff)]

//! A VJP is the gradient of a weighted sum of outputs.

use mercury::{Plan, Source, differentiable};

#[differentiable(inputs = 2, outputs = 2)]
fn function(_config: &(), input: &[f64], output: &mut [f64]) {
    let x = input[0];
    let y = input[1];
    output[0] = x * x;
    output[1] = x * y;
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let function = builder.add(function_operator(()), [Source::Input(0), Source::Input(1)]);
    let plan = builder.build([function.output(0), function.output(1)])?;
    let mut workspace = plan.workspace();
    let point = [2.0, 3.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;

    let weights = [1.0, 2.0];
    let mut gradient = [0.0; 2];
    linearization.vjp(&weights, &mut gradient)?;

    println!("f(x, y) = [x^2, x*y], at (2, 3)");
    println!("weights = {weights:?}: differentiate f[0] + 2*f[1] = x^2 + 2*x*y");
    println!("J^T * weights = {gradient:?}  expected: [10, 4]");
    assert!((gradient[0] - 10.0).abs() < 1e-12);
    assert!((gradient[1] - 4.0).abs() < 1e-12);
    Ok(())
}
