#![feature(autodiff)]
// Exact float asserts, tiny index->f64 casts, and short math names are intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Adjoint-rule tests: `solve_vjp`/`solve_jvp` vs finite differences vs Enzyme.
//!
//! The three-way agreement test is the Phase 2 thesis demo: the same
//! gradient from (1) finite differences, (2) Enzyme differentiating through
//! `solve_fixed_unchecked`, (3) the adjoint rule composing two LU solves.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{
    Matrix, SMatrix, SVector, Vector, lu_factor, solve_fixed_unchecked, solve_jvp, solve_vjp,
};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

/// Objective: theta[0..9] -> A = M + 5I, theta[9..12] -> b, f = |x|^2.
#[cfg(not(coverage))]
fn objective(theta: &[f64]) -> f64 {
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let x = lu_factor(&a).expect("wc").solve(&b).expect("wc");
    x.norm_squared()
}

/// The same objective as an Enzyme kernel through `solve_fixed_unchecked`.
// Coverage instrumentation injects atomic profile counters that Enzyme
// cannot differentiate — Enzyme legs are excluded from coverage builds.
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a =
        SMatrix::<3, 3>::from_fn(|i, j| theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    // Kernel path uses the infallible variant: Result-enum returns from
    // kernel-reachable fns fail Enzyme (Task 5 finding).
    let x = solve_fixed_unchecked(&a, &b);
    *out = x.norm_squared();
}

#[cfg(not(coverage))]
fn adjoint_gradient(theta: &[f64]) -> Vec<f64> {
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let f = lu_factor(&a).expect("wc");
    let x = f.solve(&b).expect("wc");

    // f = |x|^2  =>  x_bar = 2x.
    let x_bar = &x * 2.0;
    let grads = solve_vjp(&f, &x, &x_bar).expect("vjp");

    // d theta_(3i+j) = a_bar[i][j] (diag shift is constant), d theta_(9+i) = b_bar[i].
    let mut g = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            g[3 * i + j] = grads.a_bar[(i, j)];
        }
        g[9 + i] = grads.b_bar[i];
    }
    g
}

#[test]
#[cfg(not(coverage))]
fn three_way_gradient_agreement() {
    let theta = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];

    // (1) finite differences
    let fd = central_difference_gradient(objective, &theta, 1.0e-6).expect("fd");

    // (2) Enzyme through solve_fixed
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&theta, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&theta)).abs() < 1e-12, "primal mismatch");

    // (3) adjoint rule
    let adjoint = adjoint_gradient(&theta);

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs adjoint: {enzyme_vs_adjoint:?}\n enzyme={enzyme:?}\n adjoint={adjoint:?}"
    );

    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "adjoint vs fd: {adjoint_vs_fd:?}"
    );
}

#[test]
fn jvp_matches_directional_finite_difference() {
    let theta = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let f = lu_factor(&a).expect("wc");
    let x = f.solve(&b).expect("wc");

    // Direction: perturb every A entry and b entry.
    let a_dot = Matrix::from_fn(3, 3, |i, j| 0.1 * ((i + 2 * j) as f64) - 0.2);
    let b_dot = Vector::from_slice(&[0.3, -0.1, 0.2]);
    let x_dot = solve_jvp(&f, &x, &a_dot, &b_dot).expect("jvp");

    // FD in that direction: x(A + h*Adot, b + h*bdot).
    let h = 1.0e-7;
    let a_p = Matrix::from_fn(3, 3, |i, j| a[(i, j)] + h * a_dot[(i, j)]);
    let a_m = Matrix::from_fn(3, 3, |i, j| a[(i, j)] - h * a_dot[(i, j)]);
    let b_p = Vector::from_slice(&[
        b[0] + h * b_dot[0],
        b[1] + h * b_dot[1],
        b[2] + h * b_dot[2],
    ]);
    let b_m = Vector::from_slice(&[
        b[0] - h * b_dot[0],
        b[1] - h * b_dot[1],
        b[2] - h * b_dot[2],
    ]);
    let x_p = lu_factor(&a_p).expect("wc").solve(&b_p).expect("wc");
    let x_m = lu_factor(&a_m).expect("wc").solve(&b_m).expect("wc");

    for i in 0..3 {
        let fd_i = (x_p[i] - x_m[i]) / (2.0 * h);
        assert!(
            (x_dot[i] - fd_i).abs() < 1.0e-5,
            "component {i}: jvp={} fd={fd_i}",
            x_dot[i]
        );
    }
}

#[test]
fn vjp_dimension_mismatch_errors() {
    let a = Matrix::from_fn(3, 3, |i, j| if i == j { 5.0 } else { 0.1 });
    let f = lu_factor(&a).expect("wc");
    let x = Vector::zeros(3);
    let bad = Vector::zeros(2);
    assert!(solve_vjp(&f, &x, &bad).is_err());
}

#[test]
fn jvp_dimension_mismatch_errors() {
    let a = Matrix::from_fn(3, 3, |i, j| if i == j { 5.0 } else { 0.1 });
    let f = lu_factor(&a).expect("wc");
    let x = Vector::zeros(3);
    let a_dot = Matrix::from_fn(3, 3, |_, _| 0.0);
    let bad_b_dot = Vector::zeros(2);
    assert!(solve_jvp(&f, &x, &a_dot, &bad_b_dot).is_err());
}
