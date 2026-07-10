//! Element-wise reductions over raw `f64` slices (faer's `reductions/`
//! idea, flattened: `Vector::as_slice()` / `Matrix::as_slice()` feed these,
//! and for matrices `norm_l2` is the Frobenius norm).
//!
//! Host-side conveniences — no kernel-safety claims. `norm_l2` uses the
//! naive sum-of-squares (no overflow rescaling); fine for the magnitudes
//! Mercury works with.

/// Sum of all elements (`0.0` for empty input).
#[must_use]
pub fn sum(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v;
    }
    acc
}

/// Sum of absolute values.
#[must_use]
pub fn norm_l1(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v.abs();
    }
    acc
}

/// Euclidean norm (Frobenius norm for matrix slices).
#[must_use]
pub fn norm_l2(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v * v;
    }
    acc.sqrt()
}

/// Largest absolute value (`0.0` for empty input).
#[must_use]
pub fn norm_max(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        let a = v.abs();
        if a > acc {
            acc = a;
        }
    }
    acc
}
