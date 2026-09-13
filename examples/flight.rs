#![feature(autodiff)]

//! Compose 100 vertical-flight steps and differentiate the final state.

use mercury::{Plan, Source, function};

#[function(FlightStep)]
fn flight_step(height: f64, velocity: f64, mass: f64, thrust: f64) -> [f64; 2] {
    let dt = 0.01;
    let acceleration = thrust / mass - 9.806_65;
    [
        height + velocity * dt + 0.5 * acceleration * dt * dt,
        velocity + acceleration * dt,
    ]
}

fn main() -> mercury::Result<()> {
    // Global inputs: initial height, initial velocity, mass, thrust.
    let mut builder = Plan::builder(4);
    let mut height = Source::Input(0);
    let mut velocity = Source::Input(1);
    for _ in 0..100 {
        let step = builder.add(
            FlightStep::new(),
            [height, velocity, Source::Input(2), Source::Input(3)],
        );
        height = step.output(0);
        velocity = step.output(1);
    }
    let flight = builder.build([height, velocity])?;
    let point = [100.0, 0.0, 2.0, 20.0];
    let final_state = flight.eval(&point)?;
    let jacobian = flight.jacobian().eval(&point)?;

    println!(
        "After 1 second: height = {:.6}, velocity = {:.6}",
        final_state[0], final_state[1]
    );
    println!(
        "Sensitivity to mass: height = {:.6}, velocity = {:.6}",
        jacobian[(0, 2)],
        jacobian[(1, 2)]
    );
    assert!((final_state[0] - 100.096_675).abs() < 1e-10);
    assert!((final_state[1] - 0.193_35).abs() < 1e-10);
    assert!((jacobian[(0, 2)] + 2.5).abs() < 1e-10);
    assert!((jacobian[(1, 2)] + 5.0).abs() < 1e-10);
    Ok(())
}
