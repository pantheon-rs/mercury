#![feature(autodiff)]
//! Two independent outputs give a diagonal global Jacobian.

use mercury::{Plan, Source, function};

#[function(Square)]
fn square(x: f64) -> f64 {
    x * x
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let first = builder.add(Square::new(), [Source::Input(0)]);
    let second = builder.add(Square::new(), [Source::Input(1)]);
    let function = builder.build([first.output(0), second.output(0)])?;

    let jacobian = function.jacobian();
    let matrix = jacobian.eval_sparse(&[2.0, 3.0])?;
    println!("CSC values: {:?}", matrix.val());
    assert_eq!(matrix.val(), &[4.0, 6.0]);
    assert_eq!(jacobian.sparsity().row_idx(), &[0, 1]);
    Ok(())
}
