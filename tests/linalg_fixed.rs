#![feature(autodiff)]

//! solve_fixed unit tests + the differentiate-through-the-solver Enzyme test.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{LinalgError, SMatrix, SVector, solve_fixed, solve_fixed_unchecked};
use std::autodiff::autodiff_reverse;

#[test]
fn solves_known_system() {
    // A = [[4,1],[1,3]], b = [1,2] -> x = [1/11, 7/11]
    let a = SMatrix::new([[4.0, 1.0], [1.0, 3.0]]);
    let b = SVector::new([1.0, 2.0]);
    let x = solve_fixed(&a, &b).expect("well-conditioned");
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);

    // Residual check on a 4x4 that forces pivoting (zero on the diagonal).
    let a4 = SMatrix::new([
        [0.0, 2.0, 1.0, -1.0],
        [3.0, 0.5, -2.0, 1.0],
        [1.0, -1.0, 4.0, 2.0],
        [2.0, 1.0, 1.0, 3.0],
    ]);
    let b4 = SVector::new([1.0, -2.0, 3.0, 0.5]);
    let x4 = solve_fixed(&a4, &b4).expect("invertible");
    let r = a4 * x4 - b4;
    assert!(r.norm() < 1e-12, "residual {r:?}");
}

#[test]
fn singular_matrix_is_an_error() {
    let a = SMatrix::new([[1.0, 2.0], [2.0, 4.0]]); // rank 1
    let b = SVector::new([1.0, 1.0]);
    assert!(matches!(
        solve_fixed(&a, &b),
        Err(LinalgError::Singular { .. })
    ));
}

// --- Enzyme leg: differentiate THROUGH the pivoting solver.
// x[0..9] -> A = M + 5I (well-conditioned), x[9..12] -> b, out = |A^{-1} b|^2.
// This is exactly the kernel nalgebra's LU::solve failed to compile
// (metis-ad-spike/linalg_compat, na_solve_new).
//
// Deviation from the brief's literal kernel body: calling the
// `Result`-returning `solve_fixed(&a, &b).expect(...)` here fails to
// compile with "Enzyme: Cannot deduce type of copy" (confirmed via a
// minimal scratch-kernel bisection: an `SVector`-returning helper compiles
// fine under `#[autodiff_reverse]`, the same helper wrapped in
// `Result<SVector<N>, LinalgError>` does not). Per the brief's documented
// fallback, the kernel calls the infallible `solve_fixed_unchecked`
// instead; `solve_fixed` remains the host-facing, Result-returning API
// exercised by `solves_known_system` and `singular_matrix_is_an_error`
// above.
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| x[3 * i + j] + if i == j { 5.0 } else { 0.0 });
    let b = SVector::<3>::from_fn(|i| x[9 + i]);
    let s = solve_fixed_unchecked(&a, &b);
    *out = s.norm_squared();
}

fn kernel_value(x: &[f64]) -> f64 {
    let mut out = 0.0;
    kernel(x, &mut out);
    out
}

#[test]
fn enzyme_differentiates_through_solve() {
    let x = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];
    let mut grad = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6).expect("fd");
    let check = compare_gradients(&grad, &fd).expect("shape");
    assert!(
        check.max_abs_error < 1.0e-4,
        "{check:?}\n ad={grad:?}\n fd={fd:?}"
    );
}
