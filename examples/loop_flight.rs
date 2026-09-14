#![feature(autodiff)]
//! 6. A fixed-horizon rollout: runtime loop, scheduled thrust and signed drag.

use mercury::advanced::{PlanExecution, Workspace};
use mercury::{Error, Plan};

struct Config {
    steps: usize,
    burn_steps: usize,
    dt: f64,
    gravity: f64,
}

// Input order: altitude (m), upward velocity (m/s), thrust (N), mass (kg),
// quadratic drag coefficient (kg/m). Output: final altitude and velocity.
#[mercury::advanced::differentiable(inputs = 5, outputs = 2)]
fn rollout(config: &Config, input: &[f64], output: &mut [f64]) {
    // Constant-initialized increments avoid a loop type-inference failure on
    // the pinned compiler when state accumulators start directly from a slice.
    let mut altitude_change = 0.0;
    let mut velocity_change = 0.0;
    for step in 0..config.steps {
        let velocity = input[1] + velocity_change;
        // This branch depends on inactive configuration and the loop index.
        let thrust = if step < config.burn_steps {
            input[2]
        } else {
            0.0
        };
        // This branch depends on active state. Drag opposes either velocity sign.
        let drag = if velocity >= 0.0 {
            -input[4] * velocity * velocity
        } else {
            input[4] * velocity * velocity
        };
        let acceleration = (thrust + drag) / input[3] - config.gravity;
        // Explicit Euler: both updates use the beginning-of-step state.
        altitude_change += config.dt * velocity;
        velocity_change += config.dt * acceleration;
    }
    output[0] = input[0] + altitude_change;
    output[1] = input[1] + velocity_change;
}

fn domain(config: &Config, input: &[f64]) -> mercury::Result<()> {
    if !config.dt.is_finite()
        || config.dt <= 0.0
        || !config.gravity.is_finite()
        || config.burn_steps > config.steps
        || input[3] <= 0.0
        || input[4] < 0.0
    {
        return Err(Error::Domain(
            "finite positive dt/mass, finite gravity, nonnegative drag, and burn_steps <= steps required",
        ));
    }
    Ok(())
}

fn main() -> mercury::Result<()> {
    let ballistic = Plan::from_operator(
        rollout_operator(Config {
            steps: 4,
            burn_steps: 0,
            dt: 0.5,
            gravity: 2.0,
        })
        .with_domain(domain),
    )?;
    let value = ballistic.eval(&[10.0, 3.0, 0.0, 2.0, 0.0])?;
    // Four Euler updates use velocities 3, 2, 1, 0 for position; final velocity -1.
    assert!((value[0] - 13.0).abs() < 1e-12);
    assert!((value[1] + 1.0).abs() < 1e-12);
    println!(
        "ballistic Euler reference: altitude={}, velocity={}",
        value[0], value[1]
    );

    let flight = Plan::from_operator(
        rollout_operator(Config {
            steps: 20,
            burn_steps: 10,
            dt: 0.05,
            gravity: 9.8,
        })
        .with_domain(domain),
    )?;
    for velocity in [-3.0, 3.0] {
        let point = [10.0, velocity, 30.0, 2.0, 0.2];
        let jacobian = flight.jacobian().eval(&point)?;
        // Perturb and rerun the actual discretized rollout, checking every partial.
        for column in 0..point.len() {
            let mut upper = point;
            let mut lower = point;
            upper[column] += 1e-5;
            lower[column] -= 1e-5;
            let upper_value = flight.eval(&upper)?;
            let lower_value = flight.eval(&lower)?;
            for row in 0..2 {
                let difference = (upper_value[row] - lower_value[row]) / 2e-5;
                assert!((jacobian[(row, column)] - difference).abs() < 1e-7);
            }
        }
        let mut workspace = Workspace::new(&flight);
        let mut linearization = flight.linearize(&point, &mut workspace)?;
        let direction = [0.1, 0.2, 0.3, -0.1, 0.05];
        let weights = [0.4, -0.7];
        let mut tangent = [0.0; 2];
        let mut cotangent = [0.0; 5];
        linearization.jvp(&direction, &mut tangent)?;
        linearization.vjp(&weights, &mut cotangent)?;
        let forward_dot: f64 = weights.iter().zip(tangent).map(|(a, b)| a * b).sum();
        let reverse_dot: f64 = direction.iter().zip(cotangent).map(|(a, b)| a * b).sum();
        assert!((forward_dot - reverse_dot).abs() < 1e-11);
        println!(
            "initial velocity={velocity}: final state={:?}",
            linearization.value()?
        );
        println!(
            "  [d_altitude/d_thrust, d_velocity/d_thrust]=[{}, {}]",
            jacobian[(0, 2)],
            jacobian[(1, 2)]
        );
    }
    Ok(())
}
