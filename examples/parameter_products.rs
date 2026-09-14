#![feature(autodiff)]
//! Differentiate a root along one parameter direction without assembling `R_q`.

use mercury::advanced::{PlanExecution, Workspace};
use mercury::{ImplicitSolve, Plan};

#[mercury::function(Residual)]
fn residual(z: f64, parameters: [f64; 4]) -> f64 {
    z * z - (parameters[0] + parameters[1] + parameters[2] + parameters[3])
}

fn main() -> mercury::Result<()> {
    let solve = ImplicitSolve::new(Residual::new(), vec![1.0], 1e-12, 20)?;
    let plan = Plan::from_operator(solve)?;
    let mut workspace = Workspace::new(&plan);
    let mut linearization = plan.linearize(&[1.0; 4], &mut workspace)?;
    let mut tangent = [0.0];
    linearization.jvp(&[1.0; 4], &mut tangent)?;
    assert!((tangent[0] - 1.0).abs() < 1e-12);
    let mut gradient = [0.0; 4];
    linearization.vjp(&[1.0], &mut gradient)?;
    assert!(gradient.iter().all(|entry| (entry - 0.25).abs() < 1e-12));
    println!("directional sensitivity={tangent:?}, gradient={gradient:?}");
    Ok(())
}
