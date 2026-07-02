// Exact float asserts and tiny index->f64 casts are intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Dynamic LU factor/solve tests (primal only; adjoint rule is next task).

use mercury::{Matrix, Vector, lu_factor, solve};

fn test_matrix() -> Matrix {
    // Needs pivoting (zero at (0,0)); well-conditioned.
    Matrix::from_fn(4, 4, |i, j| match (i, j) {
        (0, 0) => 0.0,
        (i, j) if i == j => 4.0 + i as f64,
        (i, j) => 0.3 * ((2 * i + 3 * j) as f64).sin(),
    })
}

#[test]
fn solve_has_small_residual() {
    let a = test_matrix();
    let b = Vector::from_slice(&[1.0, -2.0, 3.0, 0.5]);
    let x = solve(&a, &b).expect("invertible");
    let r = &(&a * &x) - &b;
    assert!(r.norm_squared().sqrt() < 1e-12, "residual {r:?}");
}

#[test]
fn factors_are_reusable_and_transposed_solve_works() {
    let a = test_matrix();
    let f = lu_factor(&a).expect("invertible");
    assert_eq!(f.dimension(), 4);

    let b1 = Vector::from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let b2 = Vector::from_slice(&[0.0, 1.0, -1.0, 2.0]);
    let x1 = f.solve(&b1).expect("solve");
    let x2 = f.solve(&b2).expect("solve");
    assert!((&(&a * &x1) - &b1).norm_squared().sqrt() < 1e-12);
    assert!((&(&a * &x2) - &b2).norm_squared().sqrt() < 1e-12);

    // A^T z = c via the same factors.
    let c = Vector::from_slice(&[0.5, 1.5, -0.5, 1.0]);
    let z = f.solve_transposed(&c).expect("transposed solve");
    let at = a.transpose();
    assert!((&(&at * &z) - &c).norm_squared().sqrt() < 1e-12);
}

#[test]
fn singular_matrix_errors() {
    let a = Matrix::from_fn(3, 3, |i, _| i as f64); // identical columns
    assert!(lu_factor(&a).is_err());
}

#[test]
fn non_square_errors() {
    let a = Matrix::zeros(3, 4);
    assert!(lu_factor(&a).is_err());
}
