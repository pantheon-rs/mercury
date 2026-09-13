#![feature(autodiff)]

//! Arithmetic values and partial derivatives at one point.

use mercury::function;

#[function(Arithmetic)]
fn arithmetic(x: f64, y: f64) -> [f64; 4] {
    [x + y, x - y, x * y, x / y]
}

fn main() -> mercury::Result<()> {
    let function = Arithmetic::new();
    let jacobian = function.jacobian();
    let values = function.eval(6.0, 2.0)?;
    let derivatives = jacobian.eval(6.0, 2.0)?;

    let cases = [
        ("x + y", 8.0, [1.0, 1.0]),
        ("x - y", 4.0, [1.0, -1.0]),
        ("x * y", 12.0, [2.0, 6.0]),
        ("x / y", 3.0, [0.5, -1.5]),
    ];
    println!("At x = 6, y = 2:");
    for (row, (formula, expected_value, expected_gradient)) in cases.into_iter().enumerate() {
        let value = values[row];
        let gradient = derivatives[row];
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
