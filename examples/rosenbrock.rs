#![feature(autodiff)]

//! Rosenbrock: f(x, y) = (1 - x)² + 100(y - x²)². Compute its value and gradient.

use mercury::function;

#[function(Rosenbrock)]
fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0 - x;
    let b = y - x * x;
    a * a + 100.0 * b * b
}

#[allow(clippy::similar_names)] // df_dx and df_dy name the mathematical partials.
fn main() -> mercury::Result<()> {
    let function = Rosenbrock::new();
    let gradient = function.gradient();

    let value = function.eval(-1.2, 1.0)?;
    let [df_dx, df_dy] = gradient.eval(-1.2, 1.0)?;

    println!("f(x, y) = (1 - x)^2 + 100(y - x^2)^2");
    println!("At (x, y) = (-1.2, 1.0):");
    println!("f = {value:.6}                 expected: 24.2");
    println!("gradient = [{df_dx:.6}, {df_dy:.6}]  expected: [-215.6, -88]");
    assert!((value - 24.2).abs() < 1e-12);
    assert!((df_dx + 215.6).abs() < 1e-10);
    assert!((df_dy + 88.0).abs() < 1e-10);
    Ok(())
}
