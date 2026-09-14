#![feature(autodiff)]
//! 2. An ordinary if/else: derivatives on each side of a corner.

#[mercury::function(Absolute, first_order)]
fn absolute(x: f64) -> f64 {
    if x >= 0.0 { x } else { -x }
}

fn main() -> mercury::Result<()> {
    let function = Absolute::new();
    for (x, expected_slope) in [(-2.0, -1.0), (2.0, 1.0)] {
        let (value, [slope]) = function.value_and_gradient(x)?;
        assert!((value - 2.0).abs() < 1e-12);
        assert!((slope - expected_slope).abs() < 1e-12);
        println!("x={x}: absolute={value}, derivative={slope}");
    }
    // The value exists at zero, but the mathematical derivative does not.
    // An AD result from the selected branch would not establish differentiability.
    let value = function.eval(0.0)?;
    assert!(value.abs() < 1e-12);
    println!("x=0: absolute={value}; no derivative requested at the corner");
    Ok(())
}
