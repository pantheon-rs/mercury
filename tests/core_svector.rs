#![feature(autodiff)]
// Exact float asserts and tiny index->f64 casts are intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! `SVector` unit tests + Enzyme kernel-safety test (three-legged law).

use mercury::SVector;
use mercury::validation::{central_difference_gradient, compare_gradients};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

#[test]
fn constructors_and_indexing() {
    let v = SVector::new([1.0, 2.0, 3.0]);
    let w = SVector::<3>::from_fn(|i| (i as f64) + 1.0);
    assert_eq!(v, w);
    assert_eq!(v[2], 3.0);
    assert_eq!(SVector::<4>::zeros().as_slice(), &[0.0; 4]);

    let mut m = v;
    m[0] = 9.0;
    assert_eq!(m[0], 9.0);
}

#[test]
fn arithmetic_ops() {
    let v = SVector::new([1.0, 2.0, 3.0]);
    let w = SVector::new([4.0, 5.0, 6.0]);
    assert_eq!((v + w).as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!((w - v).as_slice(), &[3.0, 3.0, 3.0]);
    assert_eq!((-v).as_slice(), &[-1.0, -2.0, -3.0]);
    assert_eq!((v * 2.0).as_slice(), &[2.0, 4.0, 6.0]);
    assert_eq!((w / 2.0).as_slice(), &[2.0, 2.5, 3.0]);
}

#[test]
fn dot_norm_cross() {
    let v = SVector::new([1.0, 2.0, 3.0]);
    let w = SVector::new([4.0, 5.0, 6.0]);
    assert!((v.dot(&w) - 32.0).abs() < 1e-15);
    assert!((v.norm_squared() - 14.0).abs() < 1e-15);
    assert!((v.norm() - 14.0_f64.sqrt()).abs() < 1e-15);
    // e1 x e2 = e3
    let e1 = SVector::new([1.0, 0.0, 0.0]);
    let e2 = SVector::new([0.0, 1.0, 0.0]);
    assert_eq!(e1.cross(&e2).as_slice(), &[0.0, 0.0, 1.0]);
}

// --- Enzyme leg: differentiate a kernel built from SVector ops ---

// Coverage instrumentation injects atomic profile counters that Enzyme
// cannot differentiate ("Active atomic inst not yet handled") — the Enzyme
// legs are excluded from coverage builds and verified by the normal suite.
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let v = SVector::<3>::from_fn(|i| x[i]);
    let w = SVector::<3>::from_fn(|i| x[3 + i]);
    let c = v.cross(&w);
    *out = v.dot(&w) + c.norm_squared();
}

#[cfg(not(coverage))]
fn kernel_value(x: &[f64]) -> f64 {
    let mut out = 0.0;
    kernel(x, &mut out);
    out
}

#[test]
#[cfg(not(coverage))]
fn enzyme_gradient_matches_finite_differences() {
    let x = [0.7, -1.3, 2.1, 0.4, 1.9, -0.6];
    let mut grad = vec![0.0; 6];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6)
        .expect("finite-difference gradient should compute");
    let check = compare_gradients(&grad, &fd).expect("same shape");
    assert!(
        check.max_abs_error < 1.0e-4,
        "{check:?}\n ad={grad:?}\n fd={fd:?}"
    );
}
