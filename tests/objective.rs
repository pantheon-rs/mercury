#![feature(autodiff)]
// The scalar_objective! macro expands to #[autodiff_reverse] code, which is
// incompatible with coverage instrumentation (Enzyme cannot differentiate
// atomic profile counters) — this suite is excluded from coverage builds.
#![cfg(not(coverage))]

//! Enzyme-backed scalar objective API tests.

use mercury::validation::{central_difference_gradient, compare_gradients};

mercury::scalar_objective! {
    mod rosenbrock(x) {
        let mut acc = 0.0;
        for i in 0..x.len() - 1 {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            acc += 100.0 * a * a + b * b;
        }
        acc
    }
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
fn scalar_objective_exposes_value_and_gradient() {
    let x = vec![0.5_f64; 6];

    let direct_value = rosenbrock::value(&x);
    let result = rosenbrock::value_and_gradient(&x);
    let gradient = rosenbrock::gradient(&x);
    let analytic_gradient = rosenbrock_gradient_analytic(&x);
    let finite_difference_gradient = central_difference_gradient(rosenbrock::value, &x, 1.0e-6)
        .expect("finite-difference gradient should compute");

    assert!((direct_value - 32.5).abs() < 1.0e-12);
    assert!((result.value - direct_value).abs() < 1.0e-12);
    assert_eq!(result.dimension(), x.len());
    assert_eq!(gradient, result.gradient);

    let analytic_check = compare_gradients(&result.gradient, &analytic_gradient)
        .expect("analytic gradient shape should match");
    assert!(
        analytic_check.max_abs_error < 1.0e-9,
        "Enzyme vs analytic mismatch: {analytic_check:?}\n  enzyme={:?}\n  analytic={analytic_gradient:?}",
        result.gradient
    );

    let fd_check = compare_gradients(&result.gradient, &finite_difference_gradient)
        .expect("finite-difference gradient shape should match");
    assert!(
        fd_check.max_abs_error < 1.0e-4,
        "Enzyme vs finite-difference mismatch: {fd_check:?}\n  enzyme={:?}\n  fd={finite_difference_gradient:?}",
        result.gradient
    );
}
