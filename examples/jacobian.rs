#![feature(autodiff)]

//! The full Jacobian: one row per output, one column per input.

use mercury::function;

#[function(Polynomial)]
fn polynomial(x: f64, y: f64) -> [f64; 2] {
    [x * x, x * y]
}

fn main() -> mercury::Result<()> {
    let function = Polynomial::new();
    let jacobian = function.jacobian();
    let derivatives = jacobian.eval(2.0, 3.0)?;

    println!("f(x, y) = [x^2, x*y], at (2, 3)");
    println!("Jacobian rows [df/dx, df/dy]:");
    println!("  {:?}  expected: [4, 0]", derivatives[0]);
    println!("  {:?}  expected: [3, 2]", derivatives[1]);
    for (actual, expected) in derivatives.into_iter().flatten().zip([4.0, 0.0, 3.0, 2.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    Ok(())
}
