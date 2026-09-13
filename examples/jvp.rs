#![feature(autodiff)]

//! A JVP tells how every output changes along one input direction.

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

    let direction = [1.0, 2.0];
    let mut tangent = [0.0; 2];
    linearization.jvp(&direction, &mut tangent)?;

    println!("f(x, y) = [x^2, x*y], at (2, 3)");
    println!(
        "f = {:?}             expected: [4, 6]",
        linearization.value()?
    );
    println!("J = [[4, 0], [3, 2]], direction = {direction:?}");
    println!("J * direction = {tangent:?}  expected: [4, 7]");
    println!("A small step h in this direction changes f by approximately h * [4, 7].");
    assert!((tangent[0] - 4.0).abs() < 1e-12);
    assert!((tangent[1] - 7.0).abs() < 1e-12);
    Ok(())
}
