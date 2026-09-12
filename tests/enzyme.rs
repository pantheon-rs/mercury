#![feature(autodiff)]

//! Compiler smoke tests, independent of Mercury's future public API.

use std::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_forward(jvp, Dual, Dual)]
#[autodiff_reverse(vjp, Duplicated, Duplicated)]
fn evaluate(x: &[f64], y: &mut [f64]) {
    y[0] = x[0] * x[0] + x[1];
    y[1] = x[0] * x[1];
    y[2] = x[0] - 3.0 * x[1] * x[1];
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < tolerance,
            "entry {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn forward_matches_analytic_derivative() {
    let mut value = [0.0; 3];
    let mut tangent = [0.0; 3];
    jvp(&[1.2, -0.7], &[0.3, -0.4], &mut value, &mut tangent);

    assert_close(&value, &[0.74, -0.84, -0.27], 1.0e-12);
    assert_close(&tangent, &[0.32, -0.69, -1.38], 1.0e-12);
}

#[test]
fn reverse_matches_analytic_derivative() {
    let mut value = [0.0; 3];
    let mut input_cotangent = [0.0; 2];
    let mut output_seed = [0.5, -1.1, 0.8];
    vjp(
        &[1.2, -0.7],
        &mut input_cotangent,
        &mut value,
        &mut output_seed,
    );

    assert_close(&value, &[0.74, -0.84, -0.27], 1.0e-12);
    assert_close(&input_cotangent, &[2.77, 2.54], 1.0e-12);
}

#[test]
fn products_agree_with_finite_difference_and_adjoint_identity() {
    let point = [1.2, -0.7];
    let direction = [0.3, -0.4];
    let weights = [0.5, -1.1, 0.8];
    let mut value = [0.0; 3];
    let mut tangent = [0.0; 3];
    jvp(&point, &direction, &mut value, &mut tangent);

    let step = 1.0e-6;
    let plus: [f64; 2] = std::array::from_fn(|i| point[i] + step * direction[i]);
    let minus: [f64; 2] = std::array::from_fn(|i| point[i] - step * direction[i]);
    let mut plus_value = [0.0; 3];
    let mut minus_value = [0.0; 3];
    evaluate(&plus, &mut plus_value);
    evaluate(&minus, &mut minus_value);
    let difference: [f64; 3] =
        std::array::from_fn(|i| (plus_value[i] - minus_value[i]) / (2.0 * step));
    assert_close(&tangent, &difference, 1.0e-8);

    let mut input_cotangent = [0.0; 2];
    let mut output_seed = weights;
    vjp(&point, &mut input_cotangent, &mut value, &mut output_seed);
    let forward_dot: f64 = weights.iter().zip(tangent).map(|(a, b)| a * b).sum();
    let reverse_dot: f64 = direction
        .iter()
        .zip(input_cotangent)
        .map(|(a, b)| a * b)
        .sum();
    assert_close(&[forward_dot], &[reverse_dot], 1.0e-12);
}
