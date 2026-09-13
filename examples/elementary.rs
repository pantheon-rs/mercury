#![feature(autodiff)]

//! Elementary functions and their scalar derivatives at x = 1.

use mercury::{Plan, Source, differentiable};
use std::f64::consts::E;

#[differentiable(inputs = 1, outputs = 6)]
fn elementary(_config: &(), input: &[f64], output: &mut [f64]) {
    let x = input[0];
    output[0] = x * x;
    output[1] = x.sqrt();
    output[2] = x.exp();
    output[3] = x.ln();
    output[4] = x.sin();
    output[5] = x.cos();
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(1);
    let node = builder.add(elementary_operator(()), [Source::Input(0)]);
    let outputs = (0..6).map(|i| node.output(i)).collect::<Vec<_>>();
    let plan = builder.build(outputs)?;
    let mut workspace = plan.workspace();
    let point = [1.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let mut derivatives = [0.0; 6];
    linearization.jvp(&[1.0], &mut derivatives)?; // Unit input direction gives d/dx.

    let cases = [
        ("x^2", 1.0, 2.0),
        ("sqrt(x)", 1.0, 0.5),
        ("exp(x)", E, E),
        ("ln(x)", 0.0, 1.0),
        ("sin(x)", 0.841_470_984_807_896_5, 0.540_302_305_868_139_8),
        ("cos(x)", 0.540_302_305_868_139_8, -0.841_470_984_807_896_5),
    ];
    println!("At x = 1:");
    for (row, (formula, expected_value, expected_derivative)) in cases.into_iter().enumerate() {
        let value = linearization.value()?[row];
        let derivative = derivatives[row];
        println!("{formula:7}: value = {value:.6}, d/dx = {derivative:.6}");
        assert!((value - expected_value).abs() < 1.0e-12);
        assert!((derivative - expected_derivative).abs() < 1.0e-12);
    }
    Ok(())
}
