//! A planar powered-flight model with explicit sample-and-hold inputs.
//!
//! The six states are horizontal position, altitude, horizontal velocity,
//! vertical velocity, pitch, and pitch rate. This is a two-dimensional flight
//! model, not a six-degree-of-freedom vehicle model.

use mercury::advanced::PlanExecution;
use mercury::{Error, Plan, Result, Source};

pub const STATES: usize = 6;
pub const INPUTS: usize = 12;
pub const OUTPUTS: usize = STATES + 1;

pub type State = [f64; STATES];

/// Configuration held constant when differentiating a numerical step.
#[derive(Clone, Copy, Debug)]
pub struct FlightConfig {
    pub dt: f64,
    pub gravity: f64,
    pub linear_drag: f64,
}

/// Physical and command parameters whose trajectory sensitivities are requested.
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub mass: f64,
    pub inertia: f64,
    pub thrust_scale: f64,
    pub torque_scale: f64,
}

/// An input sample held unchanged during every RK stage in one interval.
#[derive(Clone, Copy, Debug)]
pub struct HeldInput {
    pub thrust: f64,
    pub torque: f64,
    pub wind: [f64; 2],
}

/// Fixed terminal position used to define the flight objective.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub x: f64,
    pub z: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FlightResult {
    pub state: State,
    pub score: f64,
}

fn derivatives(config: &FlightConfig, state: &State, input: &[f64]) -> State {
    let thrust = input[6];
    let torque = input[7];
    let mass = input[8];
    let inertia = input[9];
    let horizontal_air_speed = state[2] - input[10];
    let vertical_air_speed = state[3] - input[11];
    [
        state[2],
        state[3],
        (thrust * state[4].cos() - config.linear_drag * horizontal_air_speed) / mass,
        (thrust * state[4].sin() - config.linear_drag * vertical_air_speed) / mass - config.gravity,
        state[5],
        torque / inertia,
    ]
}

// Explicit element construction avoids the zero-filled temporary's memset,
// whose type the pinned Enzyme pass cannot infer across the RK stage loops.
fn shifted(state: &State, rate: &State, dt: f64) -> State {
    [
        state[0] + dt * rate[0],
        state[1] + dt * rate[1],
        state[2] + dt * rate[2],
        state[3] + dt * rate[3],
        state[4] + dt * rate[4],
        state[5] + dt * rate[5],
    ]
}

/// Advance one RK4 step. Mass and pitch inertia must be positive.
///
/// Active inputs pack state, thrust, torque, mass, inertia, and wind in that
/// order. The host commits the returned state only after successful evaluation.
/// No RK stage samples a new command or changes host state.
#[mercury::advanced::differentiable(inputs = INPUTS, outputs = STATES)]
pub fn rk4_step(config: &FlightConfig, input: &[f64], output: &mut [f64]) {
    let initial = [input[0], input[1], input[2], input[3], input[4], input[5]];
    let k1 = derivatives(config, &initial, input);
    let k2 = derivatives(config, &shifted(&initial, &k1, 0.5 * config.dt), input);
    let k3 = derivatives(config, &shifted(&initial, &k2, 0.5 * config.dt), input);
    let k4 = derivatives(config, &shifted(&initial, &k3, config.dt), input);
    for i in 0..STATES {
        output[i] = initial[i] + config.dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
}

/// Terminal position and pitch error; target coordinates are inactive.
#[mercury::advanced::differentiable(inputs = STATES, outputs = 1)]
pub fn terminal_score(target: &Target, input: &[f64], output: &mut [f64]) {
    let dx = input[0] - target.x;
    let dz = input[1] - target.z;
    output[0] = dx * dx + dz * dz + 25.0 * input[4] * input[4];
}

/// Pack an explicit state, parameter set, and held input for the step operator.
pub fn point(state: &State, parameters: Parameters, held: HeldInput) -> [f64; INPUTS] {
    [
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
        parameters.thrust_scale * held.thrust,
        parameters.torque_scale * held.torque,
        parameters.mass,
        parameters.inertia,
        held.wind[0],
        held.wind[1],
    ]
}

/// An already sampled command/wind history, independent of RHS evaluation count.
pub fn held_history(steps: usize) -> Vec<HeldInput> {
    (0..steps)
        .map(|step| {
            let sample = step / 5;
            HeldInput {
                thrust: if sample % 2 == 0 { 620.0 } else { 570.0 },
                torque: if sample % 3 == 0 { 3.0 } else { -1.5 },
                wind: if sample % 2 == 0 {
                    [4.0, 0.5]
                } else {
                    [2.0, -0.25]
                },
            }
        })
        .collect()
}

fn validate_flight(config: &FlightConfig, input: &[f64]) -> Result<()> {
    if !config.dt.is_finite() || config.dt <= 0.0 {
        return Err(Error::Domain("step duration must be finite and positive"));
    }
    if !config.gravity.is_finite() || !config.linear_drag.is_finite() || config.linear_drag < 0.0 {
        return Err(Error::Domain("gravity and nonnegative drag must be finite"));
    }
    if input[8] <= 0.0 || input[9] <= 0.0 {
        return Err(Error::Domain("mass and pitch inertia must be positive"));
    }
    Ok(())
}

/// Compose a step and terminal objective at runtime. The step output is shared
/// between the exported next state and the objective's inputs.
pub fn flight_plan(config: FlightConfig, target: Target) -> Result<Plan> {
    let mut builder = Plan::builder(INPUTS);
    let step = builder.add(
        rk4_step_operator(config).with_domain(validate_flight),
        (0..INPUTS).map(Source::Input).collect::<Vec<_>>(),
    );
    let score = builder.add(
        terminal_score_operator(target),
        (0..STATES).map(|i| step.output(i)).collect::<Vec<_>>(),
    );
    let mut outputs = (0..STATES).map(|i| step.output(i)).collect::<Vec<_>>();
    outputs.push(score.output(0));
    builder.build(outputs)
}

/// Numerical rollout with no retained derivative workspaces or hidden state.
pub fn rollout(
    plan: &Plan,
    initial: State,
    parameters: Parameters,
    history: &[HeldInput],
) -> Result<FlightResult> {
    if history.is_empty() {
        return Err(Error::Domain(
            "a trajectory requires at least one input sample",
        ));
    }
    let mut workspace = mercury::advanced::Workspace::new(plan);
    let mut state = initial;
    let mut output = [0.0; OUTPUTS];
    for &held in history {
        plan.evaluate(
            &point(&state, parameters, held),
            &mut workspace,
            &mut output,
        )?;
        state.copy_from_slice(&output[..STATES]);
    }
    Ok(FlightResult {
        state,
        score: output[STATES],
    })
}

/// A checkpointed terminal-objective adjoint over the actual RK4 update.
///
/// The returned gradient orders the six initial states, mass, inertia, thrust
/// scale, and torque scale. The caller retains the immutable plan and sampled
/// history; only checkpoint states and one replay segment's points are stored.
pub fn trajectory_vjp(
    plan: &Plan,
    initial: State,
    parameters: Parameters,
    history: &[HeldInput],
    checkpoint_stride: usize,
) -> Result<(FlightResult, [f64; 10])> {
    if history.is_empty() || checkpoint_stride == 0 {
        return Err(Error::Domain(
            "trajectory and checkpoint stride must be nonzero",
        ));
    }
    let mut workspace = mercury::advanced::Workspace::new(plan);
    let mut state = initial;
    let mut output = [0.0; OUTPUTS];
    let mut checkpoints = Vec::new();
    for (index, &held) in history.iter().enumerate() {
        if index % checkpoint_stride == 0 {
            checkpoints.push((index, state));
        }
        plan.evaluate(
            &point(&state, parameters, held),
            &mut workspace,
            &mut output,
        )?;
        state.copy_from_slice(&output[..STATES]);
    }
    let result = FlightResult {
        state,
        score: output[STATES],
    };

    let mut gradient = [0.0; 10];
    let mut state_cotangent = [0.0; STATES];
    for &(start, checkpoint) in checkpoints.iter().rev() {
        let end = start.saturating_add(checkpoint_stride).min(history.len());
        let mut replay_points = Vec::with_capacity(end - start);
        state = checkpoint;
        for &held in &history[start..end] {
            let input = point(&state, parameters, held);
            plan.evaluate(&input, &mut workspace, &mut output)?;
            state.copy_from_slice(&output[..STATES]);
            replay_points.push(input);
        }
        for index in (start..end).rev() {
            let mut seed = [0.0; OUTPUTS];
            seed[..STATES].copy_from_slice(&state_cotangent);
            if index == history.len() - 1 {
                seed[STATES] = 1.0;
            }
            let mut local = [0.0; INPUTS];
            let mut linearization =
                plan.linearize(&replay_points[index - start], &mut workspace)?;
            linearization.vjp(&seed, &mut local)?;
            state_cotangent.copy_from_slice(&local[..STATES]);
            gradient[6] += local[8];
            gradient[7] += local[9];
            gradient[8] += local[6] * history[index].thrust;
            gradient[9] += local[7] * history[index].torque;
        }
    }
    gradient[..STATES].copy_from_slice(&state_cotangent);
    Ok((result, gradient))
}
