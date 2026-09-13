#![feature(autodiff)]

//! Keep fixed configuration outside the active inputs.
//! Run with `./scripts/example.sh configuration`.

use mercury::{Plan, Source, differentiable};

#[differentiable(inputs = 1, outputs = 1)]
fn scaled_square(scale: &f64, q: &[f64], y: &mut [f64]) {
    y[0] = *scale * q[0] * q[0];
}

fn main() -> mercury::Result<()> {
    let scale = 2.0;
    let mut builder = Plan::builder(1);
    let scaled = builder.add(scaled_square_operator(scale), [Source::Input(0)]);
    let plan = builder.build([scaled.output(0)])?;

    let point = [3.0];
    let mut workspace = plan.workspace();
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let value = linearization.value()?[0];
    let mut gradient = [0.0];
    linearization.vjp(&[1.0], &mut gradient)?;

    // The gradient covers x only; scale belongs to the operator's configuration.
    // To differentiate scale too, put it in q[1], set inputs = 2, and wire both inputs.
    println!(
        "f(x) = scale × x²; scale = {scale} (fixed); x = {}",
        point[0]
    );
    println!("f(x) = {value}; df/dx = {}", gradient[0]);
    assert!((value - 18.0).abs() < 1e-12);
    assert!((gradient[0] - 12.0).abs() < 1e-12);
    Ok(())
}
