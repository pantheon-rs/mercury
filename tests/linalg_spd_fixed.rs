#![feature(autodiff)]
// Exact float asserts, tiny index->f64 casts, and short math names are
// intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Three-legged tests for the kernel-safe SPD solve: primal correctness vs
//! LU, NaN propagation on non-SPD input, Enzyme compile + gradient leg, and
//! FD cross-check.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{Matrix, SMatrix, SVector, Vector, lu_factor, solve_spd_fixed_unchecked};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

const THETA: [f64; 12] = [
    0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
];

fn spd_smatrix(theta: &[f64]) -> SMatrix<3, 3> {
    SMatrix::<3, 3>::from_fn(|i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    })
}

#[test]
fn primal_matches_lu_solve() {
    let a = spd_smatrix(&THETA);
    let b = SVector::<3>::from_fn(|i| THETA[9 + i]);
    let x = solve_spd_fixed_unchecked(&a, &b);

    let a_dyn = Matrix::from_fn(3, 3, |i, j| a[(i, j)]);
    let b_dyn = Vector::from_slice(&THETA[9..12]);
    let x_lu = lu_factor(&a_dyn).expect("wc").solve(&b_dyn).expect("wc");
    for i in 0..3 {
        assert!((x[i] - x_lu[i]).abs() < 1e-12, "component {i}");
    }
}

#[test]
fn non_spd_propagates_nan() {
    // Indefinite: sqrt of a negative pivot must produce NaN, not panic.
    let a = SMatrix::<2, 2>::from_fn(|i, j| if i == j { 1.0 } else { 2.0 });
    let b = SVector::<2>::from_fn(|_| 1.0);
    let x = solve_spd_fixed_unchecked(&a, &b);
    assert!(x[0].is_nan() || x[1].is_nan(), "expected NaN propagation");
}

/// Objective f = |x|^2 through the SPD kernel solve.
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    let x = solve_spd_fixed_unchecked(&a, &b);
    *out = x.norm_squared();
}

#[cfg(not(coverage))]
fn objective(theta: &[f64]) -> f64 {
    let a = spd_smatrix(theta);
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    let x = solve_spd_fixed_unchecked(&a, &b);
    x.norm_squared()
}

#[test]
#[cfg(not(coverage))]
fn enzyme_gradient_matches_finite_differences() {
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&THETA)).abs() < 1e-12, "primal mismatch");
    let cmp = compare_gradients(&enzyme, &fd).expect("shape");
    assert!(cmp.max_abs_error < 1.0e-4, "Enzyme vs fd: {cmp:?}");
}
