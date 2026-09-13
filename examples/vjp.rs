#![feature(autodiff)]

//! A VJP is the gradient of a weighted sum of outputs.

use mercury::{Plan, Source, function};

#[function(Polynomial)]
fn polynomial(x: f64, y: f64) -> [f64; 2] {
    [x * x, x * y]
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let node = builder.add(Polynomial::new(), [Source::Input(0), Source::Input(1)]);
    let plan = builder.build([node.output(0), node.output(1)])?;
    let mut workspace = plan.workspace();
    let point = [2.0, 3.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;

    let weights = [1.0, 2.0];
    let mut gradient = [0.0; 2];
    linearization.vjp(&weights, &mut gradient)?;

    println!("f(x, y) = [x^2, x*y], at (2, 3)");
    println!("weights = {weights:?}: differentiate f[0] + 2*f[1] = x^2 + 2*x*y");
    println!("J^T * weights = {gradient:?}  expected: [10, 4]");
    assert!((gradient[0] - 10.0).abs() < 1e-12);
    assert!((gradient[1] - 4.0).abs() < 1e-12);
    Ok(())
}
