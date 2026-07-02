//! Phase 2 demo: one gradient, three ways.
//!
//! d/d(theta) |A(theta)^{-1} b(theta)|^2 computed by
//!   (1) central finite differences,
//!   (2) Enzyme reverse-mode differentiating THROUGH `solve_fixed_unchecked`,
//!   (3) the adjoint rule (two LU solves) — no AD inside the solve at all.
//!
//! Run: `nix develop "path:$PWD" --command cargo run --release --example solve_gradient`
#![feature(autodiff)]

use mercury::validation::central_difference_gradient;
use mercury::{Matrix, SMatrix, SVector, Vector, lu_factor, solve_fixed_unchecked, solve_vjp};
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

fn objective(theta: &[f64]) -> f64 {
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    lu_factor(&a).unwrap().solve(&b).unwrap().norm_squared()
}

#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a =
        SMatrix::<3, 3>::from_fn(|i, j| theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    *out = solve_fixed_unchecked(&a, &b).norm_squared();
}

fn main() {
    let theta = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];

    let fd = central_difference_gradient(objective, &theta, 1.0e-6).unwrap();

    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&theta, &mut enzyme, &mut out, &mut dout);

    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let f = lu_factor(&a).unwrap();
    let x = f.solve(&b).unwrap();
    let grads = solve_vjp(&f, &x, &(&x * 2.0)).unwrap();
    let mut adjoint = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            adjoint[3 * i + j] = grads.a_bar[(i, j)];
        }
        adjoint[9 + i] = grads.b_bar[i];
    }

    println!("objective value = {out:.12}\n");
    println!(
        "{:>4}  {:>18}  {:>18}  {:>18}",
        "i", "finite-diff", "enzyme", "adjoint-rule"
    );
    for i in 0..12 {
        println!(
            "{i:>4}  {:>18.12}  {:>18.12}  {:>18.12}",
            fd[i], enzyme[i], adjoint[i]
        );
    }
}
