#![feature(autodiff)]

//! Elementary functions and their scalar derivatives at x = 1.

use mercury::function;
use std::f64::consts::E;

#[function(Elementary)]
fn elementary(x: f64) -> [f64; 6] {
    [x * x, x.sqrt(), x.exp(), x.ln(), x.sin(), x.cos()]
}

fn main() -> mercury::Result<()> {
    let function = Elementary::new();
    let jacobian = function.jacobian();
    let values = function.eval(1.0)?;
    let derivatives = jacobian.eval(1.0)?;

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
        let value = values[row];
        let derivative = derivatives[row][0];
        println!("{formula:7}: value = {value:.6}, d/dx = {derivative:.6}");
        assert!((value - expected_value).abs() < 1.0e-12);
        assert!((derivative - expected_derivative).abs() < 1.0e-12);
    }
    Ok(())
}
