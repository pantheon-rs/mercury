#![feature(autodiff)]
//! Scale residuals explicitly; inspect convergence before composing a root.

#[mercury::function(Residual)]
fn residual(z: f64, q: f64) -> f64 {
    1e-12 * (z * z - q)
}

fn main() -> mercury::Result<()> {
    let solve = mercury::ImplicitSolve::new(Residual::new(), vec![1.0], 1e-10, 20)?
        .with_scaling(vec![1e-12], vec![2.0])?;
    let (root, report) = solve.solve_with_report(&[4.0])?;
    assert!((root[0] - 2.0).abs() < 1e-10);
    println!("root={}, convergence={report:?}", root[0]);
    let plan = mercury::Plan::from_operator(solve)?;
    let derivative = plan.gradient().eval(&[4.0])?;
    assert!((derivative[0] - 0.25).abs() < 1e-10);
    println!("dz/dq={}", derivative[0]);
    Ok(())
}
