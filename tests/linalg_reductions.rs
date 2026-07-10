// Exact float asserts and tiny index->f64 casts are intentional in tests.
#![allow(clippy::float_cmp, clippy::cast_precision_loss)]

//! Reductions and factorization determinants.

use mercury::linalg::reductions::{norm_l1, norm_l2, norm_max, sum};
use mercury::{Matrix, Vector, llt_factor, lu_factor};

#[test]
fn vector_reductions() {
    let v = Vector::from_slice(&[3.0, -4.0]);
    assert_eq!(sum(v.as_slice()), -1.0);
    assert_eq!(norm_l1(v.as_slice()), 7.0);
    assert_eq!(norm_l2(v.as_slice()), 5.0);
    assert_eq!(norm_max(v.as_slice()), 4.0);
}

#[test]
fn matrix_reductions_use_all_elements() {
    // norm_l2 over a matrix slice is the Frobenius norm.
    let m = Matrix::from_fn(2, 2, |i, j| [[1.0, -2.0], [2.0, 4.0]][i][j]);
    assert_eq!(sum(m.as_slice()), 5.0);
    assert_eq!(norm_l1(m.as_slice()), 9.0);
    assert_eq!(norm_l2(m.as_slice()), 5.0);
    assert_eq!(norm_max(m.as_slice()), 4.0);
}

#[test]
fn empty_reductions_are_zero() {
    assert_eq!(sum(&[]), 0.0);
    assert_eq!(norm_l1(&[]), 0.0);
    assert_eq!(norm_l2(&[]), 0.0);
    assert_eq!(norm_max(&[]), 0.0);
}

#[test]
fn matrix_as_slice_is_row_major() {
    // 2x3 matrix: as_slice must yield row 0 fully before row 1.
    let m = Matrix::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
    assert_eq!(m.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn all_negative_reductions() {
    // norm_max must use absolute value, not raw max, when every element is
    // negative (a naive `x > acc` fold starting at the most-negative value
    // would otherwise report the wrong extreme).
    let v = Vector::from_slice(&[-1.0, -5.0, -2.0]);
    assert_eq!(sum(v.as_slice()), -8.0);
    assert_eq!(norm_l1(v.as_slice()), 8.0);
    assert_eq!(norm_max(v.as_slice()), 5.0);
}

#[test]
fn single_element_reductions() {
    let single = [-3.0];
    assert_eq!(sum(&single), -3.0);
    assert_eq!(norm_l1(&single), 3.0);
    assert_eq!(norm_l2(&single), 3.0);
    assert_eq!(norm_max(&single), 3.0);
}

#[test]
fn lu_determinant_with_permutation_sign() {
    // [[0, 1], [1, 0]] forces a pivot swap; det = -1.
    let a = Matrix::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
    let det = lu_factor(&a).expect("wc").determinant();
    assert!((det - (-1.0)).abs() < 1e-14);
    // [[2, 1], [1, 3]]: det = 5.
    let b = Matrix::from_fn(2, 2, |i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
    let det_b = lu_factor(&b).expect("wc").determinant();
    assert!((det_b - 5.0).abs() < 1e-12);
}

#[test]
fn llt_determinant_matches_lu() {
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let det_llt = llt_factor(&a).expect("spd").determinant();
    let det_lu = lu_factor(&a).expect("wc").determinant();
    assert!((det_llt - 8.0).abs() < 1e-12);
    assert!((det_llt - det_lu).abs() < 1e-12);
}
