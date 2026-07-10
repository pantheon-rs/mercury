#![feature(autodiff)]
// Exact float asserts, tiny index->f64 casts, and short math names are
// intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Cholesky LLT tests: correctness, reconstruction, errors, cross-check vs
//! LU, and the three-way gradient agreement through the generic adjoint rule.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{
    LinalgError, Matrix, SMatrix, SVector, Vector, ldlt_factor, llt_factor, lu_factor,
    solve_fixed_unchecked, solve_jvp, solve_vjp,
};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

/// Symmetric SPD test matrix from 9 raw params:
/// `a_ij = 0.5*(theta[3i+j] + theta[3j+i]) + 5*delta_ij`.
fn spd_from(theta: &[f64]) -> Matrix {
    Matrix::from_fn(3, 3, |i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    })
}

const THETA: [f64; 12] = [
    0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
];

#[test]
fn known_solution_2x2() {
    // A = [[4, 2], [2, 3]], b = [8, 7] => x = [1.25, 1.5]
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let b = Vector::from_slice(&[8.0, 7.0]);
    let x = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    assert!((x[0] - 1.25).abs() < 1e-14);
    assert!((x[1] - 1.5).abs() < 1e-14);
}

#[test]
fn reconstruction_l_lt_matches_a() {
    let a = spd_from(&THETA);
    let f = llt_factor(&a).expect("spd");
    let l = f.l();
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                // L is lower: entry (i, k) is zero for k > i.
                let lik = if k <= i { l[(i, k)] } else { 0.0 };
                let ljk = if k <= j { l[(j, k)] } else { 0.0 };
                acc += lik * ljk;
            }
            assert!(
                (acc - a[(i, j)]).abs() < 1e-12,
                "reconstruction ({i},{j}): {acc} vs {}",
                a[(i, j)]
            );
        }
    }
}

#[test]
fn llt_solve_matches_lu_solve() {
    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let x_llt = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    let x_lu = lu_factor(&a).expect("wc").solve(&b).expect("solve");
    for i in 0..3 {
        assert!((x_llt[i] - x_lu[i]).abs() < 1e-12, "component {i}");
    }
}

#[test]
fn only_lower_triangle_is_read() {
    // Poison the strict upper triangle: result must be identical.
    let a = spd_from(&THETA);
    let poisoned = Matrix::from_fn(3, 3, |i, j| if j > i { f64::NAN } else { a[(i, j)] });
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let x_clean = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    let x_poisoned = llt_factor(&poisoned)
        .expect("spd")
        .solve(&b)
        .expect("solve");
    assert_eq!(x_clean, x_poisoned);
}

#[test]
fn indefinite_matrix_errors() {
    // Eigenvalues 3 and -1: not positive definite.
    let a = Matrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 2.0 });
    assert_eq!(
        llt_factor(&a).map(|_| ()),
        Err(LinalgError::NotPositiveDefinite { pivot_index: 1 })
    );
}

#[test]
fn dimension_mismatches_error() {
    let rect = Matrix::zeros(2, 3);
    assert!(llt_factor(&rect).is_err());
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let f = llt_factor(&a).expect("spd");
    assert!(f.solve(&Vector::zeros(3)).is_err());
}

#[test]
fn llt_dimension_accessor_matches_input_size() {
    let a = spd_from(&THETA);
    let f = llt_factor(&a).expect("spd");
    assert_eq!(f.dimension(), 3);
}

#[test]
fn llt_pivot_at_tolerance_boundary_succeeds() {
    // sum == 1e-12 hits the `sum <= PIVOT_TOLERANCE` boundary exactly:
    // this is a breakdown (non-strict comparison), so it must error, not
    // silently take sqrt of a near-zero value.
    let a = Matrix::from_fn(1, 1, |_, _| 1.0e-12);
    assert_eq!(
        llt_factor(&a).map(|_| ()),
        Err(LinalgError::NotPositiveDefinite { pivot_index: 0 })
    );
    // Comfortably above tolerance succeeds.
    let ok = Matrix::from_fn(1, 1, |_, _| 1.0e-6);
    assert!(llt_factor(&ok).is_ok());
}

#[test]
fn llt_jvp_matches_directional_finite_difference() {
    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let f = llt_factor(&a).expect("spd");
    let x = f.solve(&b).expect("solve");

    // The direction must be SYMMETRIC: llt_factor reads only the lower
    // triangle, so the finite-difference leg below actually perturbs by
    // the symmetric part of whatever lands in `a_p`/`a_m`. An asymmetric
    // a_dot would make the JVP (which uses a_dot verbatim) and the FD
    // disagree by construction.
    let a_dot = Matrix::from_fn(3, 3, |i, j| 0.05 * ((i + j) as f64) - 0.1);
    let b_dot = Vector::from_slice(&[0.2, -0.1, 0.05]);
    let x_dot = solve_jvp(&f, &x, &a_dot, &b_dot).expect("jvp");

    let h = 1.0e-7;
    let a_p = Matrix::from_fn(3, 3, |i, j| a[(i, j)] + h * a_dot[(i, j)]);
    let a_m = Matrix::from_fn(3, 3, |i, j| a[(i, j)] - h * a_dot[(i, j)]);
    let mut bp = Vec::new();
    let mut bm = Vec::new();
    for i in 0..3 {
        bp.push(b[i] + h * b_dot[i]);
        bm.push(b[i] - h * b_dot[i]);
    }
    // Perturbed A stays SPD for this small h (diagonal shift dominates).
    let x_p = llt_factor(&a_p)
        .expect("spd")
        .solve(&Vector::from_vec(bp))
        .expect("solve");
    let x_m = llt_factor(&a_m)
        .expect("spd")
        .solve(&Vector::from_vec(bm))
        .expect("solve");
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
fn llt_solve_transposed_equals_solve() {
    // A is symmetric, so Factorization::solve_transposed must be
    // byte-for-byte the same computation as solve.
    use mercury::Factorization;
    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let f = llt_factor(&a).expect("spd");
    let x = Factorization::solve(&f, &b).expect("solve");
    let xt = f.solve_transposed(&b).expect("solve_transposed");
    assert_eq!(x, xt);
}

/// Objective: theta[0..9] -> symmetrized SPD A, theta[9..12] -> b, f = |x|^2.
#[cfg(not(coverage))]
fn objective(theta: &[f64]) -> f64 {
    let a = spd_from(theta);
    let b = Vector::from_slice(&theta[9..12]);
    let x = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    x.norm_squared()
}

/// Same objective as an Enzyme kernel (LU kernel path — mathematically the
/// same function of theta, so gradients must agree with the LLT rule leg).
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    let x = solve_fixed_unchecked(&a, &b);
    *out = x.norm_squared();
}

#[test]
#[cfg(not(coverage))]
fn three_way_gradient_agreement_llt() {
    // (1) finite differences
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");

    // (2) Enzyme
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&THETA)).abs() < 1e-12, "primal mismatch");

    // (3) generic adjoint rule through LltFactors
    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let f = llt_factor(&a).expect("spd");
    let x = f.solve(&b).expect("solve");
    let x_bar = &x * 2.0;
    let grads = solve_vjp(&f, &x, &x_bar).expect("vjp");
    // Chain through the symmetrization: d a_kl / d theta_(3i+j) contributes
    // 0.5*(a_bar[i][j] + a_bar[j][i]) to g[3i+j].
    let mut adjoint = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            adjoint[3 * i + j] = 0.5 * (grads.a_bar[(i, j)] + grads.a_bar[(j, i)]);
        }
        adjoint[9 + i] = grads.b_bar[i];
    }

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs LLT adjoint: {enzyme_vs_adjoint:?}\n enzyme={enzyme:?}\n adjoint={adjoint:?}"
    );
    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "LLT adjoint vs fd: {adjoint_vs_fd:?}"
    );
}

#[test]
fn ldlt_known_solution_2x2() {
    // A = [[4, 2], [2, 3]]: d = [4, 2], l10 = 0.5.
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let f = ldlt_factor(&a).expect("factor");
    assert!((f.d()[0] - 4.0).abs() < 1e-14);
    assert!((f.d()[1] - 2.0).abs() < 1e-14);
    assert!((f.l()[(1, 0)] - 0.5).abs() < 1e-14);
    let b = Vector::from_slice(&[8.0, 7.0]);
    let x = f.solve(&b).expect("solve");
    assert!((x[0] - 1.25).abs() < 1e-14);
    assert!((x[1] - 1.5).abs() < 1e-14);
}

#[test]
fn ldlt_handles_indefinite_and_matches_lu() {
    // Eigenvalues 3 and -1: indefinite, LLT fails but unpivoted LDLT works.
    let a = Matrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 2.0 });
    let b = Vector::from_slice(&[1.0, -1.0]);
    let x_ldlt = ldlt_factor(&a).expect("factor").solve(&b).expect("solve");
    let x_lu = lu_factor(&a).expect("wc").solve(&b).expect("solve");
    for i in 0..2 {
        assert!((x_ldlt[i] - x_lu[i]).abs() < 1e-12, "component {i}");
    }
}

#[test]
fn ldlt_breakdown_errors() {
    // Second pivot d_1 = 1 - 1*1*1 = 0: breakdown.
    let a = Matrix::from_fn(2, 2, |_, _| 1.0);
    assert_eq!(
        ldlt_factor(&a).map(|_| ()),
        Err(LinalgError::NotPositiveDefinite { pivot_index: 1 })
    );
}

#[test]
fn ldlt_dimension_mismatches_error() {
    let rect = Matrix::zeros(2, 3);
    assert!(ldlt_factor(&rect).is_err());
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let f = ldlt_factor(&a).expect("factor");
    assert_eq!(f.dimension(), 2);
    assert!(f.solve(&Vector::zeros(3)).is_err());
}

#[test]
fn ldlt_solve_transposed_equals_solve() {
    use mercury::Factorization;
    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let f = ldlt_factor(&a).expect("factor");
    let x = Factorization::solve(&f, &b).expect("solve");
    let xt = f.solve_transposed(&b).expect("solve_transposed");
    assert_eq!(x, xt);
}

#[test]
#[cfg(not(coverage))]
fn ldlt_adjoint_matches_enzyme_and_fd() {
    // Same objective/kernel as the LLT three-way test; only the rule leg
    // changes factorization backend. Gradients must be identical.
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);

    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let f = ldlt_factor(&a).expect("factor");
    let x = f.solve(&b).expect("solve");
    let x_bar = &x * 2.0;
    let grads = solve_vjp(&f, &x, &x_bar).expect("vjp");
    let mut adjoint = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            adjoint[3 * i + j] = 0.5 * (grads.a_bar[(i, j)] + grads.a_bar[(j, i)]);
        }
        adjoint[9 + i] = grads.b_bar[i];
    }

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs LDLT adjoint: {enzyme_vs_adjoint:?}"
    );
    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "LDLT adjoint vs fd: {adjoint_vs_fd:?}"
    );
}
