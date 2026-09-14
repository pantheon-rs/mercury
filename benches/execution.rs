#![feature(autodiff)]
//! Wall-clock measurements of preparation, reused products and flight replay.

use mercury::advanced::{PlanExecution, Workspace};
use mercury::{DenseSolve, Plan, Result};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/support/flight.rs"]
#[allow(dead_code)]
mod flight;

#[mercury::function(Energy)]
fn energy(input: [f64; 4]) -> f64 {
    input[0] * input[0] + input[1] * input[1] + input[2] * input[2] + input[3] * input[3]
}

fn measure(name: &str, iterations: u32, mut run: impl FnMut() -> Result<()>) -> Result<()> {
    run()?; // Warm reusable storage outside the timed region.
    let start = Instant::now();
    for _ in 0..iterations {
        run()?;
    }
    println!(
        "{name}: {:?}/call ({iterations} calls)",
        start.elapsed() / iterations
    );
    Ok(())
}

fn main() -> Result<()> {
    for n in [10, 20, 40, 80] {
        measure(&format!("DenseSolve({n}) plan construction"), 100, || {
            black_box(Plan::from_operator(DenseSolve::new(black_box(n))?)?);
            Ok(())
        })?;
    }
    let plan = Plan::from_operator(Energy::new())?;
    let point = [1.0; 4];
    let seed = [0.5; 4];
    let mut workspace = Workspace::new(&plan);
    let mut scalar = [0.0];
    let mut vector = [0.0; 4];
    measure("reused value", 10_000, || {
        plan.evaluate(black_box(&point), &mut workspace, &mut scalar)?;
        black_box(&scalar);
        Ok(())
    })?;
    measure("reused linearization", 10_000, || {
        black_box(plan.linearize(black_box(&point), &mut workspace)?.value()?);
        Ok(())
    })?;
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    measure("JVP", 10_000, || {
        linearization.jvp(black_box(&seed), &mut scalar)?;
        black_box(&scalar);
        Ok(())
    })?;
    measure("VJP", 10_000, || {
        linearization.vjp(black_box(&[1.0]), &mut vector)?;
        black_box(&vector);
        Ok(())
    })?;
    measure("Hessian-vector product", 10_000, || {
        linearization.curvature(black_box(&[1.0]), black_box(&seed), &mut vector)?;
        black_box(&vector);
        Ok(())
    })?;
    measure("wide Jacobian", 10_000, || {
        linearization.jacobian(&mut vector)?;
        black_box(&vector);
        Ok(())
    })?;

    let flight = flight::flight_plan(
        flight::FlightConfig {
            dt: 0.02,
            gravity: 9.806_65,
            linear_drag: 0.4,
        },
        flight::Target { x: 22.0, z: 99.0 },
    )?;
    let initial = [0.0, 100.0, 30.0, 0.0, 0.9, 0.02];
    let parameters = flight::Parameters {
        mass: 50.0,
        inertia: 25.0,
        thrust_scale: 1.0,
        torque_scale: 1.0,
    };
    let history = flight::held_history(100);
    measure("100-step checkpointed RK4 adjoint (stride 10)", 100, || {
        black_box(flight::trajectory_vjp(
            &flight,
            black_box(initial),
            parameters,
            &history,
            10,
        )?);
        Ok(())
    })?;
    Ok(())
}
