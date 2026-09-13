//! Solve a 2×2 linear system and differentiate its solution.

use mercury::{DenseSolve, Plan, Result, Source};

fn main() -> Result<()> {
    let mut builder = Plan::builder(6);
    let solve = builder.add(
        DenseSolve::new(2)?,
        (0..6).map(Source::Input).collect::<Vec<_>>(),
    );
    let plan = builder.build([solve.output(0), solve.output(1)])?;
    let mut workspace = plan.workspace();

    // Inputs are [A00, A01, A10, A11, b0, b1]: A is row-major.
    let point = [3.0, 1.0, 1.0, 2.0, 5.0, 5.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    println!("A = [[3, 1], [1, 2]], b = [5, 5]");
    println!("Solution x = {:.6?}", linearization.value()?);
    for (actual, expected) in linearization.value()?.iter().zip([1.0, 2.0]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    // Hold A fixed and perturb b in the direction [1, 0].
    let mut tangent = [0.0; 2];
    linearization.jvp(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], &mut tangent)?;
    println!("dx for dA = 0, db = [1, 0]: {tangent:.6?}");
    for (actual, expected) in tangent.iter().zip([0.4, -0.2]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    // L = x0 + x1: propagate its output gradient [1, 1] back to A and b.
    let mut gradient = [0.0; 6];
    linearization.vjp(&[1.0, 1.0], &mut gradient)?;
    println!(
        "For L = x0 + x1, dL/dA (row-major) = {:.6?}",
        &gradient[..4]
    );
    println!("For L = x0 + x1, dL/db = {:.6?}", &gradient[4..]);
    for (actual, expected) in gradient.iter().zip([-0.2, -0.4, -0.4, -0.8, 0.2, 0.4]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    Ok(())
}
