//! Validation utility tests.

use mercury::validation::{ValidationError, central_difference_gradient, compare_gradients};

fn rosenbrock_value(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i].mul_add(-x[i], x[i + 1]);
        let b = 1.0 - x[i];
        acc += (100.0 * a).mul_add(a, b * b);
    }
    acc
}

fn rosenbrock_gradient_analytic(x: &[f64]) -> Vec<f64> {
    let mut gradient = vec![0.0; x.len()];
    for i in 0..x.len() - 1 {
        let a = x[i].mul_add(-x[i], x[i + 1]);
        gradient[i] += (-400.0 * x[i]).mul_add(a, -2.0 * (1.0 - x[i]));
        gradient[i + 1] = (200.0_f64).mul_add(a, gradient[i + 1]);
    }
    gradient
}

#[test]
fn central_difference_matches_quadratic_gradient() {
    let x = [3.0, -2.0];
    let gradient = central_difference_gradient(
        |value| value[0].mul_add(value[0], 3.0 * value[1] * value[1]),
        &x,
        1.0e-6,
    )
    .expect("valid finite-difference step");

    let expected = [6.0, -12.0];
    let check = compare_gradients(&gradient, &expected).expect("same gradient shape");

    assert!(check.max_abs_error < 1.0e-8, "{check:?}");
    assert!(check.max_rel_error < 1.0e-8, "{check:?}");
}

#[test]
fn central_difference_matches_small_rosenbrock_gradient() {
    let x = [0.5, 0.5, 0.5, 0.5];
    let gradient = central_difference_gradient(rosenbrock_value, &x, 1.0e-6).expect("valid step");
    let expected = rosenbrock_gradient_analytic(&x);
    let check = compare_gradients(&gradient, &expected).expect("same gradient shape");

    assert!(check.max_abs_error < 1.0e-6, "{check:?}");
    assert!(check.max_rel_error < 1.0e-8, "{check:?}");
}

#[test]
fn central_difference_rejects_invalid_step() {
    let err = central_difference_gradient(|value| value[0], &[1.0], 0.0)
        .expect_err("zero step should be rejected");

    assert_eq!(err, ValidationError::InvalidStep { step: 0.0 });
}

#[test]
fn compare_gradients_reports_worst_index() {
    let actual = [1.0, 2.5, 3.0];
    let expected = [1.0, 2.0, 2.9];
    let check = compare_gradients(&actual, &expected).expect("same gradient shape");

    assert_eq!(check.worst_index, 1);
    assert!((check.max_abs_error - 0.5).abs() < 1.0e-12);
}

#[test]
fn compare_gradients_rejects_length_mismatch() {
    let err = compare_gradients(&[1.0, 2.0], &[1.0]).expect_err("length mismatch");

    assert_eq!(
        err,
        ValidationError::LengthMismatch {
            actual: 2,
            expected: 1
        }
    );
}
