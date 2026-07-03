#![feature(autodiff)]

//! Quaternion unit tests + Enzyme rotation-kernel test.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{Quaternion, SVector};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;
use std::f64::consts::FRAC_PI_2;

#[test]
fn identity_and_hamilton_product() {
    let q = Quaternion::new(0.5, -0.3, 0.8, 0.1).normalized();
    let i = Quaternion::identity();
    let qi = q * i;
    assert!((qi.w - q.w).abs() < 1e-15 && (qi.x - q.x).abs() < 1e-15);

    // i * j = k in Hamilton convention.
    let qi_ = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let qj = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    let qk = qi_ * qj;
    assert!((qk.z - 1.0).abs() < 1e-15 && qk.w.abs() < 1e-15);
}

#[test]
fn rotation_and_dcm_agree() {
    // 90 degrees about z: e1 -> e2.
    let axis = SVector::new([0.0, 0.0, 1.0]);
    let q = Quaternion::from_axis_angle(&axis, FRAC_PI_2);
    let e1 = SVector::new([1.0, 0.0, 0.0]);

    let r = q.rotate(&e1);
    assert!((r[0]).abs() < 1e-12 && (r[1] - 1.0).abs() < 1e-12 && r[2].abs() < 1e-12);

    let dcm = q.to_dcm();
    let rd = dcm * e1;
    assert!((rd[0] - r[0]).abs() < 1e-12 && (rd[1] - r[1]).abs() < 1e-12);

    // Rotation preserves length.
    let v = SVector::new([0.3, -1.2, 2.2]);
    assert!((q.rotate(&v).norm() - v.norm()).abs() < 1e-12);

    // conjugate rotates back.
    let back = q.conjugate().rotate(&r);
    assert!((back[0] - 1.0).abs() < 1e-12 && back[1].abs() < 1e-12);
}

// --- Enzyme leg: rotate a vector by a normalized quaternion, sum squares.
// x[0..4] = raw quaternion (normalized in-kernel), x[4..7] = vector.

// Coverage instrumentation injects atomic profile counters that Enzyme
// cannot differentiate ("Active atomic inst not yet handled") — the Enzyme
// legs are excluded from coverage builds and verified by the normal suite.
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let q = Quaternion::new(x[0], x[1], x[2], x[3]).normalized();
    let v = SVector::new([x[4], x[5], x[6]]);
    let r = q.rotate(&v);
    *out = r.norm_squared() + r[0] * r[1];
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
    let x = [0.9, 0.2, -0.3, 0.1, 0.5, -1.1, 0.8];
    let mut grad = vec![0.0; 7];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6).expect("fd");
    let check = compare_gradients(&grad, &fd).expect("shape");
    assert!(
        check.max_abs_error < 1.0e-4,
        "{check:?}\n ad={grad:?}\n fd={fd:?}"
    );
}
