#![feature(autodiff)]

//! Planar flight with compiled RK4 kernels, runtime composition, and a
//! checkpointed terminal-objective adjoint. Run with `./scripts/example.sh flight`.

#[path = "flight/model.rs"]
mod model;

use model::{FlightConfig, Parameters, Target, flight_plan, held_history, rollout, trajectory_vjp};

fn main() -> mercury::Result<()> {
    let plan = flight_plan(
        FlightConfig {
            dt: 0.02,
            gravity: 9.806_65,
            linear_drag: 0.4,
        },
        Target { x: 75.0, z: 101.0 },
    )?;
    let parameters = Parameters {
        mass: 50.0,
        inertia: 25.0,
        thrust_scale: 1.0,
        torque_scale: 1.0,
    };
    let initial = [0.0, 100.0, 30.0, 0.0, 0.9, 0.02];
    let history = held_history(100);
    let direct = rollout(&plan, initial, parameters, &history)?;
    let (replayed, gradient) = trajectory_vjp(&plan, initial, parameters, &history, 20)?;

    println!("Planar flight: 100 RK4 steps, 0.02 s per step");
    println!(
        "Final position: x={:.6} m, z={:.6} m",
        direct.state[0], direct.state[1]
    );
    println!("Final pitch: {:.6} rad", direct.state[4]);
    println!("Terminal objective: {:.9}", direct.score);
    println!(
        "Replay objective difference: {:.3e}",
        replayed.score - direct.score
    );
    println!("Sensitivity to mass: {:.9}", gradient[6]);
    println!("Sensitivity to pitch inertia: {:.9}", gradient[7]);
    println!("Sensitivity to thrust scale: {:.9}", gradient[8]);
    println!("Sensitivity to torque scale: {:.9}", gradient[9]);
    Ok(())
}
