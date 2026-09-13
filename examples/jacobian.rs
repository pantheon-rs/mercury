#![feature(autodiff)]

//! The full Jacobian: one row per output, one column per input.

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

    let mut jacobian = [0.0; 4];
    linearization.jacobian(&mut jacobian)?;

    println!("f(x, y) = [x^2, x*y], at (2, 3)");
    println!("Jacobian rows [df/dx, df/dy]:");
    println!("  {:?}  expected: [4, 0]", &jacobian[..2]);
    println!("  {:?}  expected: [3, 2]", &jacobian[2..]);
    for (actual, expected) in jacobian.into_iter().zip([4.0, 0.0, 3.0, 2.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    Ok(())
}
