// Exact float asserts and tiny index->f64 casts are intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Dynamic Vector/Matrix unit tests (host-side types; no Enzyme leg —
//! kernels receive dynamic data as slices, which Phase 1 already proves).

use mercury::{Matrix, Vector};

#[test]
fn vector_basics() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let w = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());
    assert_eq!(v[1], 2.0);
    assert!((v.dot(&w) - 32.0).abs() < 1e-15);
    assert!((v.norm_squared() - 14.0).abs() < 1e-15);
    assert_eq!((&v + &w).as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!((&w - &v).as_slice(), &[3.0, 3.0, 3.0]);
    assert_eq!((&v * 2.0).as_slice(), &[2.0, 4.0, 6.0]);

    let mut z = Vector::zeros(2);
    z[0] = 7.0;
    assert_eq!(z.as_slice(), &[7.0, 0.0]);
}

#[test]
fn matrix_basics() {
    let a = Matrix::from_fn(2, 3, |i, j| (3 * i + j) as f64); // [[0,1,2],[3,4,5]]
    assert_eq!((a.rows(), a.cols()), (2, 3));
    assert_eq!(a[(1, 2)], 5.0);

    let t = a.transpose();
    assert_eq!((t.rows(), t.cols()), (3, 2));
    assert_eq!(t[(2, 1)], 5.0);

    let v = Vector::from_slice(&[1.0, 1.0, 1.0]);
    assert_eq!((&a * &v).as_slice(), &[3.0, 12.0]);

    let b = Matrix::from_fn(3, 2, |i, j| (2 * i + j) as f64); // [[0,1],[2,3],[4,5]]
    let c = &a * &b;
    assert_eq!((c.rows(), c.cols()), (2, 2));
    // row0 = [0,1,2]·cols -> [0*0+1*2+2*4, 0*1+1*3+2*5] = [10, 13]
    assert_eq!(c[(0, 0)], 10.0);
    assert_eq!(c[(0, 1)], 13.0);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn matvec_dimension_mismatch_panics() {
    let a = Matrix::zeros(2, 3);
    let v = Vector::zeros(2);
    let _ = &a * &v;
}

#[test]
#[should_panic(expected = "matrix index out of bounds")]
fn matrix_column_overflow_does_not_wrap_to_next_row() {
    let a = Matrix::from_fn(2, 3, |i, j| (3 * i + j) as f64);
    // (0, 3) flattens to the same offset as (1, 0) — must panic, not read 3.0.
    let _ = a[(0, 3)];
}
