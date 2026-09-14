#![feature(autodiff)]
//! 1. A fixed loop: repeated multiplication, then first and second derivatives.

#[mercury::function(Cube)]
fn cube(x: f64) -> f64 {
    let mut value = 1.0;
    for _ in 0..3 {
        value *= x;
    }
    value
}

fn main() -> mercury::Result<()> {
    let function = Cube::new();
    for x in [-2.0, 0.0, 2.0] {
        let (value, [slope]) = function.value_and_gradient(x)?;
        let [[curvature]] = function.gradient().jacobian().eval(x)?;
        assert!((value - x * x * x).abs() < 1e-12);
        assert!((slope - 3.0 * x * x).abs() < 1e-12);
        assert!((curvature - 6.0 * x).abs() < 1e-12);
        println!("x={x}: cube={value}, derivative={slope}, second derivative={curvature}");
    }
    Ok(())
}
