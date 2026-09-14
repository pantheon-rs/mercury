#![feature(autodiff)]
//! Independent invariants for the documented aerospace examples and domains.

#[path = "../examples/attitude.rs"]
#[allow(dead_code, unused_attributes)]
mod attitude;
#[path = "../examples/first_order.rs"]
#[allow(dead_code, unused_attributes)]
mod first_order;
#[path = "../examples/impact.rs"]
#[allow(dead_code, unused_attributes)]
mod impact;

use mercury::advanced::{Operator, PlanExecution, Workspace};
use mercury::{Plan, Result};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn quaternion_normalization_preserves_rotation_and_force_norm() -> Result<()> {
    for angle in [-2.0_f64, -0.1, 0.0, 0.7, 2.0] {
        for scale in [0.001, 1.0, 1000.0] {
            let q = [
                scale * (angle / 2.0).cos(),
                0.0,
                0.0,
                scale * (angle / 2.0).sin(),
            ];
            for force in [1e-6, 1.0, 1e6] {
                let result = attitude::RotatedForce::new().eval(q, [force, 0.0, 0.0])?;
                close(result[0], force * angle.cos(), 1e-9);
                close(result[1], force * angle.sin(), 1e-9);
                close(result[2], 0.0, 1e-12);
                close(
                    result.iter().map(|entry| entry * entry).sum::<f64>(),
                    force * force,
                    1e-12,
                );
            }
        }
    }
    assert!(
        attitude::RotatedForce::new()
            .eval([0.0; 4], [1.0; 3])
            .is_err()
    );
    let jacobian = attitude::LocalForce::new()
        .jacobian()
        .eval([0.0; 3], [2.0, 3.0, 5.0])?;
    let expected = [[0.0, 5.0, -3.0], [-5.0, 0.0, 2.0], [3.0, -2.0, 0.0]];
    for (actual, expected) in jacobian
        .delta
        .iter()
        .flatten()
        .zip(expected.iter().flatten())
    {
        close(*actual, *expected, 1e-12);
    }
    Ok(())
}

#[test]
fn attitude_products_agree_with_finite_differences_and_adjoint_identity() -> Result<()> {
    let plan = Plan::from_operator(attitude::LocalForce::new())?;
    let point = [0.2, -0.1, 0.3, 2.0, 3.0, 5.0];
    let direction = [0.1, 0.3, -0.2, 0.5, -0.4, 0.7];
    let weights = [0.4, -0.2, 0.5];
    let mut workspace = Workspace::new(&plan);
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let mut tangent = [0.0; 3];
    let mut gradient = [0.0; 6];
    let mut curvature = [0.0; 6];
    linearization.jvp(&direction, &mut tangent)?;
    linearization.vjp(&weights, &mut gradient)?;
    linearization.curvature(&weights, &direction, &mut curvature)?;
    close(
        weights.iter().zip(tangent).map(|(a, b)| a * b).sum(),
        direction.iter().zip(gradient).map(|(a, b)| a * b).sum(),
        1e-12,
    );
    for h in [1e-4, 1e-5, 1e-6] {
        let upper: [f64; 6] = std::array::from_fn(|i| point[i] + h * direction[i]);
        let lower: [f64; 6] = std::array::from_fn(|i| point[i] - h * direction[i]);
        let upper_value = plan.eval(&upper)?;
        let lower_value = plan.eval(&lower)?;
        for i in 0..3 {
            close(
                tangent[i],
                (upper_value[i] - lower_value[i]) / (2.0 * h),
                1e-7,
            );
        }
        let mut upper_gradient = [0.0; 6];
        let mut lower_gradient = [0.0; 6];
        plan.linearize(&upper, &mut workspace)?
            .vjp(&weights, &mut upper_gradient)?;
        plan.linearize(&lower, &mut workspace)?
            .vjp(&weights, &mut lower_gradient)?;
        for i in 0..6 {
            close(
                curvature[i],
                (upper_gradient[i] - lower_gradient[i]) / (2.0 * h),
                1e-7,
            );
        }
    }
    Ok(())
}

#[test]
fn first_order_tables_have_explicit_capabilities_and_cellwise_derivatives() -> Result<()> {
    let table = first_order::Table::new();
    assert_eq!(table.derivative_order(), 1);
    assert_eq!(table.gradient().derivative_order(), 0);
    for (x, slope) in [(0.0, 1.0), (0.999, 1.0), (1.001, 2.0), (2.0, 2.0)] {
        let (value, derivative) = table.value_and_gradient(x)?;
        close(value, first_order::table(x), 1e-12);
        close(derivative[0], slope, 1e-12);
        let h = 1e-5;
        close(
            derivative[0],
            (first_order::table(x + h) - first_order::table(x - h)) / (2.0 * h),
            1e-9,
        );
    }
    // No claim is made about differentiability at the knot x=1.
    let plan = Plan::from_operator(table)?;
    assert_eq!(plan.derivative_order(), 1);
    assert!(plan.gradient().jacobian().eval(&[0.5]).is_err());
    let gradient = Plan::from_operator(table.gradient())?;
    close(gradient.eval(&[0.5])?[0], 1.0, 1e-12);
    assert!(gradient.jacobian().eval(&[0.5]).is_err());
    Ok(())
}

#[test]
fn impact_sensitivities_include_event_time_and_reset() -> Result<()> {
    let plan = impact::impact_plan()?;
    for point in [
        [5.0_f64, 0.0, 10.0, 0.5],
        [4.0, 1.0, 9.0, 0.8],
        [6.0, -2.0, 9.81, 0.3],
    ] {
        let [height, velocity, gravity, restitution] = point;
        let speed = (velocity * velocity + 2.0 * gravity * height).sqrt();
        let (actual, gradient) = plan.value_and_gradient(&point)?;
        close(actual, restitution * speed, 1e-11);
        let expected = [
            restitution * gravity / speed,
            restitution * velocity / speed,
            restitution * height / speed,
            speed,
        ];
        for i in 0..4 {
            close(gradient[i], expected[i], 1e-11);
            let mut upper = point;
            let mut lower = point;
            upper[i] += 1e-5;
            lower[i] -= 1e-5;
            close(
                gradient[i],
                (plan.eval(&upper)?[0] - plan.eval(&lower)?[0]) / 2e-5,
                1e-8,
            );
        }
    }
    let hessian = plan.gradient().jacobian().eval(&[5.0, 0.0, 10.0, 0.5])?;
    close(hessian[(1, 1)], 0.05, 1e-12);
    // Freezing impact time would incorrectly give zero altitude sensitivity.
    close(plan.gradient().eval(&[5.0, 0.0, 10.0, 0.5])?[0], 0.5, 1e-12);
    Ok(())
}
