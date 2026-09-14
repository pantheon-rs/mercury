//! Backward error and conditioning answer different questions about a solution.

fn main() -> mercury::Result<()> {
    let solve = mercury::DenseSolve::new(2)?;
    for scale in [1.0, 1e-12] {
        let (solution, report) = solve.solve_with_report(&[1.0, 0.0, 0.0, scale, 1.0, scale])?;
        assert!((solution[0] - 1.0).abs() < 1e-14);
        assert!((solution[1] - 1.0).abs() < 1e-14);
        println!("solution={solution:?}, diagnostics={report:?}");
    }
    Ok(())
}
