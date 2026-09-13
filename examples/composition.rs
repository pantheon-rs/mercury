#![feature(autodiff)]

//! Compose two kernels with a shared input. Run with `./scripts/example.sh composition`.

use mercury::{Plan, Source, differentiable};

#[differentiable(inputs = 1, outputs = 1)]
fn square(_config: &(), q: &[f64], y: &mut [f64]) {
    y[0] = q[0] * q[0];
}

#[differentiable(inputs = 2, outputs = 1)]
fn sum(_config: &(), q: &[f64], y: &mut [f64]) {
    y[0] = q[0] + q[1];
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(1);
    let squared = builder.add(square_operator(()), [Source::Input(0)]);
    let result = builder.add(sum_operator(()), [squared.output(0), Source::Input(0)]);
    let plan = builder.build([result.output(0)])?;

    let point = [3.0];
    let mut workspace = plan.workspace();
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let value = linearization.value()?[0];
    let mut gradient = [0.0];
    // Both paths from x contribute: d(x² + x)/dx = 2x + 1.
    linearization.vjp(&[1.0], &mut gradient)?;

    println!("f(x) = x² + x; x = {}", point[0]);
    println!("f(x) = {value}; df/dx = {}", gradient[0]);
    assert!((value - 12.0).abs() < 1e-12);
    assert!((gradient[0] - 7.0).abs() < 1e-12);
    Ok(())
}
