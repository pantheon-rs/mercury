#![feature(autodiff)]
// Exact float asserts, tiny index->f64 casts, and short math names are
// intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Least-squares adjoint rule: three-way agreement on a 5x3 full-rank
//! problem. The Enzyme leg solves the normal equations `(AᵀA)x = Aᵀb` with
//! the kernel-safe SPD solve — mathematically the same function of theta as
//! the host-side QR least-squares solve.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{
    Matrix, SMatrix, SVector, Vector, lstsq_jvp, lstsq_vjp, qr_factor, solve_spd_fixed_unchecked,
};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

// theta[0..15]: A (5x3, row-major, offset to be well-conditioned);
// theta[15..20]: b.
const THETA: [f64; 20] = [
    1.0, 0.2, -0.1, 0.3, 1.5, 0.4, -0.2, 0.1, 2.0, 0.5, -0.3, 0.2, 0.7, 0.6, 1.1, 2.1, -0.4, 0.9,
    1.3, -1.7,
];

fn a_from(theta: &[f64]) -> Matrix {
    Matrix::from_fn(5, 3, |i, j| theta[3 * i + j])
}

fn b_from(theta: &[f64]) -> Vector {
    Vector::from_slice(&theta[15..20])
}

/// Host objective f = |x|^2 through QR least squares.
fn objective(theta: &[f64]) -> f64 {
    let a = a_from(theta);
    let b = b_from(theta);
    let x = qr_factor(&a).expect("qr").solve_lstsq(&b).expect("lstsq");
    x.norm_squared()
}

/// Same objective as an Enzyme kernel via the normal equations.
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    // G = AᵀA (3x3), rhs = Aᵀb — both built element-wise from theta.
    let g = SMatrix::<3, 3>::from_fn(|p, q| {
        let mut acc = 0.0;
        for i in 0..5 {
            acc += theta[3 * i + p] * theta[3 * i + q];
        }
        acc
    });
    let rhs = SVector::<3>::from_fn(|p| {
        let mut acc = 0.0;
        for i in 0..5 {
            acc += theta[3 * i + p] * theta[15 + i];
        }
        acc
    });
    let x = solve_spd_fixed_unchecked(&g, &rhs);
    *out = x.norm_squared();
}

#[test]
#[cfg(not(coverage))]
fn three_way_gradient_agreement_lstsq() {
    // (1) finite differences
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");

    // (2) Enzyme through the normal-equations kernel
    let mut enzyme = vec![0.0; 20];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&THETA)).abs() < 1e-10, "primal mismatch");

    // (3) the least-squares adjoint rule
    let a = a_from(&THETA);
    let b = b_from(&THETA);
    let f = qr_factor(&a).expect("qr");
    let x = f.solve_lstsq(&b).expect("lstsq");
    let x_bar = &x * 2.0;
    let grads = lstsq_vjp(&f, &a, &b, &x, &x_bar).expect("vjp");
    let mut adjoint = vec![0.0; 20];
    for i in 0..5 {
        for j in 0..3 {
            adjoint[3 * i + j] = grads.a_bar[(i, j)];
        }
        adjoint[15 + i] = grads.b_bar[i];
    }

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs lstsq adjoint: {enzyme_vs_adjoint:?}\n enzyme={enzyme:?}\n adjoint={adjoint:?}"
    );
    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "lstsq adjoint vs fd: {adjoint_vs_fd:?}"
    );
}

#[test]
fn jvp_matches_directional_finite_difference() {
    let a = a_from(&THETA);
    let b = b_from(&THETA);
    let f = qr_factor(&a).expect("qr");
    let x = f.solve_lstsq(&b).expect("lstsq");

    let a_dot = Matrix::from_fn(5, 3, |i, j| 0.05 * ((i + 2 * j) as f64) - 0.1);
    let b_dot = Vector::from_slice(&[0.3, -0.1, 0.2, 0.05, -0.25]);
    let x_dot = lstsq_jvp(&f, &a, &b, &x, &a_dot, &b_dot).expect("jvp");

    let h = 1.0e-7;
    let a_p = Matrix::from_fn(5, 3, |i, j| a[(i, j)] + h * a_dot[(i, j)]);
    let a_m = Matrix::from_fn(5, 3, |i, j| a[(i, j)] - h * a_dot[(i, j)]);
    let mut bp = Vec::new();
    let mut bm = Vec::new();
    for i in 0..5 {
        bp.push(b[i] + h * b_dot[i]);
        bm.push(b[i] - h * b_dot[i]);
    }
    let x_p = qr_factor(&a_p)
        .expect("qr")
        .solve_lstsq(&Vector::from_vec(bp))
        .expect("lstsq");
    let x_m = qr_factor(&a_m)
        .expect("qr")
        .solve_lstsq(&Vector::from_vec(bm))
        .expect("lstsq");
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
fn lstsq_vjp_dimension_mismatches_error() {
    let a = a_from(&THETA);
    let b = b_from(&THETA);
    let f = qr_factor(&a).expect("qr");
    let x = f.solve_lstsq(&b).expect("lstsq");
    let x_bar = &x * 2.0;
    // wrong x_bar length
    assert!(lstsq_vjp(&f, &a, &b, &x, &Vector::zeros(4)).is_err());
    // wrong x length
    assert!(lstsq_vjp(&f, &a, &b, &Vector::zeros(4), &x_bar).is_err());
    // wrong b length
    assert!(lstsq_vjp(&f, &a, &Vector::zeros(4), &x, &x_bar).is_err());
    // wrong a shape
    assert!(lstsq_vjp(&f, &Matrix::zeros(5, 2), &b, &x, &x_bar).is_err());
}
