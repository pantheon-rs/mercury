#![feature(autodiff)]

//! Numerical and derivative checks for the complete planar-flight step graph.

#[path = "../examples/flight/model.rs"]
mod model;

use mercury::Plan;
use model::{
    FlightConfig, HeldInput, INPUTS, OUTPUTS, Parameters, STATES, State, Target, flight_plan,
    held_history, point, rk4_step, rollout, trajectory_vjp,
};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "actual={actual:.16e}, expected={expected:.16e}"
    );
}

const fn scenario() -> (FlightConfig, State, Parameters, Target) {
    (
        FlightConfig {
            dt: 0.02,
            gravity: 9.806_65,
            linear_drag: 0.4,
        },
        [0.0, 100.0, 30.0, 0.0, 0.9, 0.02],
        Parameters {
            mass: 50.0,
            inertia: 25.0,
            thrust_scale: 1.0,
            torque_scale: 1.0,
        },
        Target { x: 22.0, z: 99.0 },
    )
}

fn displaced(
    initial: State,
    parameters: Parameters,
    direction: &[f64; 10],
    amount: f64,
) -> (State, Parameters) {
    let mut state = initial;
    for i in 0..STATES {
        state[i] += amount * direction[i];
    }
    (
        state,
        Parameters {
            mass: parameters.mass + amount * direction[6],
            inertia: parameters.inertia + amount * direction[7],
            thrust_scale: parameters.thrust_scale + amount * direction[8],
            torque_scale: parameters.torque_scale + amount * direction[9],
        },
    )
}

fn trajectory_jvp(
    plan: &Plan,
    initial: State,
    parameters: Parameters,
    history: &[HeldInput],
    direction: &[f64; 10],
) -> mercury::Result<f64> {
    let mut workspace = plan.workspace();
    let mut state = initial;
    let mut state_tangent = [0.0; STATES];
    state_tangent.copy_from_slice(&direction[..STATES]);
    let mut tangent = [0.0; OUTPUTS];
    for &held in history {
        let input = point(&state, parameters, held);
        let mut seed = [0.0; INPUTS];
        seed[..STATES].copy_from_slice(&state_tangent);
        seed[6] = held.thrust * direction[8];
        seed[7] = held.torque * direction[9];
        seed[8] = direction[6];
        seed[9] = direction[7];
        let mut linearization = plan.linearize(&input, &mut workspace)?;
        linearization.jvp(&seed, &mut tangent)?;
        state.copy_from_slice(&linearization.value()?[..STATES]);
        state_tangent.copy_from_slice(&tangent[..STATES]);
    }
    Ok(tangent[STATES])
}

#[test]
fn ballistic_motion_and_constant_torque_have_analytic_trajectory_gradients() -> mercury::Result<()>
{
    let config = FlightConfig {
        dt: 0.05,
        gravity: 9.806_65,
        linear_drag: 0.0,
    };
    let initial = [3.0, 20.0, 4.0, 2.0, 0.2, -0.1];
    let parameters = Parameters {
        mass: 10.0,
        inertia: 4.0,
        thrust_scale: 1.0,
        torque_scale: 1.0,
    };
    let target = Target { x: 8.0, z: 18.0 };
    let plan = flight_plan(config, target)?;
    let history = vec![
        HeldInput {
            thrust: 0.0,
            torque: 2.0,
            wind: [0.0; 2],
        };
        20
    ];
    // Twenty steps span exactly one second. With no drag/thrust, position and
    // pitch are quadratic in time, so RK4 must reproduce this closed form.
    let expected = [
        initial[0] + initial[2],
        initial[1] + initial[3] - 0.5 * config.gravity,
        initial[2],
        initial[3] - config.gravity,
        initial[4] + initial[5] + 0.5 * 2.0 / parameters.inertia,
        initial[5] + 2.0 / parameters.inertia,
    ];
    let (result, gradient) = trajectory_vjp(&plan, initial, parameters, &history, 6)?;
    for (&actual, expected) in result.state.iter().zip(expected) {
        assert_close(actual, expected, 1.0e-12);
    }
    let x_bar = 2.0 * (expected[0] - target.x);
    let z_bar = 2.0 * (expected[1] - target.z);
    let pitch_bar = 50.0 * expected[4];
    let expected_gradient = [
        x_bar,
        z_bar,
        x_bar,
        z_bar,
        pitch_bar,
        pitch_bar,
        0.0,
        -pitch_bar / (parameters.inertia * parameters.inertia),
        0.0,
        pitch_bar / parameters.inertia,
    ];
    for (actual, expected) in gradient.into_iter().zip(expected_gradient) {
        assert_close(actual, expected, 1.0e-11);
    }
    Ok(())
}

#[test]
fn runtime_plan_and_checkpoint_replay_preserve_the_actual_step_sequence() -> mercury::Result<()> {
    let (config, initial, parameters, target) = scenario();
    let plan = flight_plan(config, target)?;
    let history = held_history(31);
    let direct = rollout(&plan, initial, parameters, &history)?;
    let mut state = initial;
    for &held in &history {
        let input = point(&state, parameters, held);
        rk4_step(&config, &input, &mut state);
    }
    assert_eq!(direct.state.map(f64::to_bits), state.map(f64::to_bits));

    let (retained, retained_gradient) = trajectory_vjp(&plan, initial, parameters, &history, 1)?;
    for stride in [7, history.len(), usize::MAX] {
        let (replayed, gradient) = trajectory_vjp(&plan, initial, parameters, &history, stride)?;
        assert_eq!(
            replayed.state.map(f64::to_bits),
            retained.state.map(f64::to_bits)
        );
        assert_eq!(replayed.score.to_bits(), direct.score.to_bits());
        assert_eq!(
            gradient.map(f64::to_bits),
            retained_gradient.map(f64::to_bits)
        );
    }
    Ok(())
}

#[test]
fn trajectory_adjoint_matches_perturbed_initial_conditions_and_parameters() -> mercury::Result<()> {
    let (config, initial, parameters, target) = scenario();
    let plan = flight_plan(config, target)?;
    let history = held_history(31);
    let (_, gradient) = trajectory_vjp(&plan, initial, parameters, &history, 7)?;
    for variable in 0..gradient.len() {
        let mut direction = [0.0; 10];
        direction[variable] = 1.0;
        let step = 1.0e-5;
        let (plus_state, plus_parameters) = displaced(initial, parameters, &direction, step);
        let (minus_state, minus_parameters) = displaced(initial, parameters, &direction, -step);
        let plus = rollout(&plan, plus_state, plus_parameters, &history)?.score;
        let minus = rollout(&plan, minus_state, minus_parameters, &history)?.score;
        assert_close(gradient[variable], (plus - minus) / (2.0 * step), 2.0e-6);
    }
    Ok(())
}

#[test]
fn trajectory_forward_reverse_products_satisfy_the_adjoint_identity() -> mercury::Result<()> {
    let (config, initial, parameters, target) = scenario();
    let plan = flight_plan(config, target)?;
    let history = held_history(31);
    let direction = [0.2, -0.4, 0.1, 0.3, -0.02, 0.01, 0.7, -0.4, 0.05, -0.08];
    let forward = trajectory_jvp(&plan, initial, parameters, &history, &direction)?;
    let (_, gradient) = trajectory_vjp(&plan, initial, parameters, &history, 7)?;
    let reverse: f64 = gradient.iter().zip(direction).map(|(g, d)| g * d).sum();
    assert_close(forward, reverse, 1.0e-11);
    Ok(())
}

#[test]
fn invalid_physical_parameters_and_empty_replay_are_errors() -> mercury::Result<()> {
    let (config, initial, mut parameters, target) = scenario();
    let plan = flight_plan(config, target)?;
    let history = held_history(2);
    assert!(rollout(&plan, initial, parameters, &[]).is_err());
    assert!(trajectory_vjp(&plan, initial, parameters, &history, 0).is_err());
    parameters.mass = 0.0;
    assert!(rollout(&plan, initial, parameters, &history).is_err());
    parameters.mass = 50.0;
    parameters.inertia = -1.0;
    assert!(trajectory_vjp(&plan, initial, parameters, &history, 1).is_err());
    Ok(())
}
