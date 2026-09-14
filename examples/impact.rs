#![feature(autodiff)]
//! One transverse ground impact: differentiate event time and velocity reset.
//! The host selects this event branch. No event scheduler is hidden in Mercury.

use mercury::{ImplicitSolve, Plan, Source};

#[mercury::function(Height)]
fn height(time: f64, altitude: f64, velocity: f64, gravity: f64) -> f64 {
    altitude + velocity * time - 0.5 * gravity * time * time
}

#[mercury::function(Reset)]
fn reset(time: f64, velocity: f64, gravity: f64, restitution: f64) -> f64 {
    -restitution * (velocity - gravity * time)
}

/// Inputs: altitude (m), upward velocity (m/s), downward gravity (m/s²),
/// restitution (dimensionless). Output: upward velocity just after impact.
///
/// This example's positive root branch requires positive altitude and gravity
/// and an initial time guess in its Newton basin. Grazing is outside its domain.
///
/// # Errors
/// Propagates solve-constructor or plan validation errors.
pub fn impact_plan() -> mercury::Result<Plan> {
    let time = ImplicitSolve::new(Height::new(), vec![1.0], 1e-12, 30)?;
    let mut builder = Plan::builder(4);
    let event = builder.add(time, [Source::Input(0), Source::Input(1), Source::Input(2)]);
    let reset = builder.add(
        Reset::new(),
        [
            event.output(0),
            Source::Input(1),
            Source::Input(2),
            Source::Input(3),
        ],
    );
    builder.build([reset.output(0)])
}

fn main() -> mercury::Result<()> {
    let plan = impact_plan()?;
    let (velocity, gradient) = plan.value_and_gradient(&[5.0, 0.0, 10.0, 0.5])?;
    assert!((velocity - 5.0).abs() < 1e-12);
    for (actual, expected) in gradient.iter().zip([0.5, 0.0, 0.25, 10.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    println!("post-impact velocity={velocity}, gradient={gradient:?}");
    Ok(())
}
