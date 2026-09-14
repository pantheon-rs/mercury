#![feature(autodiff)]
//! Unpublished nodes are pruned; products need no sparsity preparation.

use mercury::advanced::{Operator, PlanExecution, Workspace};
use mercury::{Plan, Source};

#[mercury::function(Unused, first_order)]
fn unused(x: f64) -> f64 {
    x.sqrt()
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(1);
    builder.add(Unused::new(), [Source::Input(0)]);
    let plan = builder.build([Source::Input(0)])?;
    // This identity has no dependence on the unused square root's domain or rules.
    assert_eq!(plan.derivative_order(), 2);
    let point = [-1.0];
    let mut workspace = Workspace::new(&plan);
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let mut product = [0.0];
    linearization.jvp(&[2.0], &mut product)?;
    assert!((product[0] - 2.0).abs() < 1e-14);
    linearization.curvature(&[1.0], &[2.0], &mut product)?;
    assert!(product[0].abs() < 1e-14);
    // Repeated curvature calls reuse this workspace's scratch.
    linearization.curvature(&[1.0], &[3.0], &mut product)?;
    println!("identity Hessian-vector product={product:?}");
    // This explicit structural request prepares and shares the CSC pattern.
    println!(
        "Jacobian entries={}",
        plan.jacobian().sparsity().row_idx().len()
    );
    Ok(())
}
