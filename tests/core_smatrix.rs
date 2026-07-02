#![feature(autodiff)]

//! SMatrix unit tests + Enzyme kernel test (the linalg_compat control kernel
//! rewritten with Mercury types, plus an analytic oracle).

use mercury::{SMatrix, SVector};
use mercury::validation::{central_difference_gradient, compare_gradients};
use std::autodiff::autodiff_reverse;

#[test]
fn constructors_indexing_identity() {
    let a = SMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(a[(0, 1)], 2.0);
    assert_eq!(a, SMatrix::<2, 2>::from_fn(|i, j| (2 * i + j + 1) as f64));

    let eye = SMatrix::<3, 3>::identity();
    assert_eq!(eye[(1, 1)], 1.0);
    assert_eq!(eye[(1, 2)], 0.0);
    assert_eq!(SMatrix::<2, 3>::zeros()[(1, 2)], 0.0);
}

#[test]
fn matmul_matvec_transpose() {
    let a = SMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = SMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let c = a * b;
    assert_eq!(c, SMatrix::new([[19.0, 22.0], [43.0, 50.0]]));

    let v = SVector::new([1.0, 1.0]);
    assert_eq!((a * v).as_slice(), &[3.0, 7.0]);

    let t = a.transpose();
    assert_eq!(t, SMatrix::new([[1.0, 3.0], [2.0, 4.0]]));

    assert_eq!((a + b), SMatrix::new([[6.0, 8.0], [10.0, 12.0]]));
    assert_eq!((b - a), SMatrix::new([[4.0, 4.0], [4.0, 4.0]]));
    assert_eq!((a * 2.0), SMatrix::new([[2.0, 4.0], [6.0, 8.0]]));
}

// --- Enzyme leg: f(x) = sum((A*B) elementwise squared), A,B from x ---
// Same kernel as linalg_compat's control/na_fixed_new; analytic gradient:
// dF/dA = 2 (A B) B^T,  dF/dB = 2 A^T (A B).

#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| x[3 * i + j]);
    let b = SMatrix::<3, 3>::from_fn(|i, j| x[9 + 3 * i + j]);
    let c = a * b;
    let mut acc = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            acc += c[(i, j)] * c[(i, j)];
        }
    }
    *out = acc;
}

fn kernel_value(x: &[f64]) -> f64 {
    let mut out = 0.0;
    kernel(x, &mut out);
    out
}

#[test]
fn enzyme_gradient_matches_fd_and_analytic() {
    let x: Vec<f64> = (0..18).map(|i| 0.3 + 0.1 * (i as f64)).collect();
    let mut grad = vec![0.0; 18];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    // Finite differences.
    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6).expect("fd");
    let fd_check = compare_gradients(&grad, &fd).expect("shape");
    assert!(fd_check.max_abs_error < 1.0e-4, "{fd_check:?}");

    // Analytic: dF/dA = 2 C B^T, dF/dB = 2 A^T C with C = A B.
    let a = SMatrix::<3, 3>::from_fn(|i, j| x[3 * i + j]);
    let b = SMatrix::<3, 3>::from_fn(|i, j| x[9 + 3 * i + j]);
    let c = a * b;
    let da = (c * b.transpose()) * 2.0;
    let db = (a.transpose() * c) * 2.0;
    let mut analytic = vec![0.0; 18];
    for i in 0..3 {
        for j in 0..3 {
            analytic[3 * i + j] = da[(i, j)];
            analytic[9 + 3 * i + j] = db[(i, j)];
        }
    }
    let an_check = compare_gradients(&grad, &analytic).expect("shape");
    assert!(an_check.max_abs_error < 1.0e-9, "{an_check:?}\n ad={grad:?}\n an={analytic:?}");
}
