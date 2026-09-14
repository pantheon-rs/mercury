//! Solve Ax=b and evaluate the solution's Jacobian.

use mercury::{DenseSolve, Plan, Result};

fn main() -> Result<()> {
    let function = Plan::from_operator(DenseSolve::new(2)?)?;
    // Inputs: row-major A, followed by b.
    let point = [3.0, 1.0, 1.0, 2.0, 5.0, 5.0];
    let solution = function.eval(&point)?;
    let jacobian = function.jacobian().eval(&point)?;

    println!("A = [[3, 1], [1, 2]], b = [5, 5]");
    println!("Solution = {solution:.6?}, expected [1, 2]");
    // Column 4 differentiates with respect to b[0].
    println!(
        "dx/db[0] = [{:.6}, {:.6}], expected [0.4, -0.2]",
        jacobian[(0, 4)],
        jacobian[(1, 4)]
    );
    assert!((solution[0] - 1.0).abs() < 1e-12);
    assert!((solution[1] - 2.0).abs() < 1e-12);
    assert!((jacobian[(0, 4)] - 0.4).abs() < 1e-12);
    assert!((jacobian[(1, 4)] + 0.2).abs() < 1e-12);
    Ok(())
}
