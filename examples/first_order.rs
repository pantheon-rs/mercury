#![feature(autodiff)]
//! Compile only value, JVP and VJP rules for a piecewise linear table.

use mercury::advanced::Operator;

/// Two interpolation cells. Derivatives are defined only away from the knot x=1.
#[mercury::function(Table, first_order)]
pub fn table(x: f64) -> f64 {
    if x < 1.0 { x } else { 2.0 * x - 1.0 }
}

fn main() -> mercury::Result<()> {
    let function = Table::new();
    assert_eq!(function.derivative_order(), 1);
    for x in [0.5, 1.5] {
        let (value, gradient) = function.value_and_gradient(x)?;
        println!("x={x}: value={value}, slope={}", gradient[0]);
    }
    // A gradient handle can evaluate values, but has no derivatives of its own.
    assert_eq!(function.gradient().derivative_order(), 0);
    Ok(())
}
