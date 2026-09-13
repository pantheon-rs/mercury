#![feature(autodiff)]

//! Arithmetic values and partial derivatives at one point.

use mercury::{Plan, Source, differentiable};

#[differentiable(inputs = 2, outputs = 4)]
fn arithmetic(_config: &(), input: &[f64], output: &mut [f64]) {
    let x = input[0];
    let y = input[1];
    output[0] = x + y;
    output[1] = x - y;
    output[2] = x * y;
    output[3] = x / y;
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let node = builder.add(
        arithmetic_operator(()),
        [Source::Input(0), Source::Input(1)],
    );
    let outputs = (0..4).map(|i| node.output(i)).collect::<Vec<_>>();
    let plan = builder.build(outputs)?;
    let mut workspace = plan.workspace();
    let point = [6.0, 2.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let mut jacobian = [0.0; 8];
    linearization.jacobian(&mut jacobian)?;

    let cases = [
        ("x + y", 8.0, [1.0, 1.0]),
        ("x - y", 4.0, [1.0, -1.0]),
        ("x * y", 12.0, [2.0, 6.0]),
        ("x / y", 3.0, [0.5, -1.5]),
    ];
    println!("At x = 6, y = 2:");
    for (row, (formula, expected_value, expected_gradient)) in cases.into_iter().enumerate() {
        let value = linearization.value()?[row];
        let gradient = &jacobian[2 * row..2 * row + 2];
        println!(
            "{formula}: value = {value:.4}, df/dx = {:.4}, df/dy = {:.4}",
            gradient[0], gradient[1]
        );
        assert!((value - expected_value).abs() < 1.0e-12);
        for (actual, expected) in gradient.iter().zip(expected_gradient) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
    Ok(())
}
