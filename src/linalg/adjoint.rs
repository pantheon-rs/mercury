//! The adjoint rule for linear solves — Mercury's first owned derivative
//! rule (decision 0003).
//!
//! For `x = A^{-1} b`:
//! - reverse (VJP), given output cotangent `x_bar`:
//!   `b_bar = A^{-T} x_bar`, `A_bar = -b_bar x^T`
//! - forward (JVP), given input tangents `(A_dot, b_dot)`:
//!   `x_dot = A^{-1} (b_dot - A_dot x)`
//!
//! Both reuse the primal [`Factorization`] — no factorization is ever
//! differentiated, and the factorization backend can change freely.
#![allow(clippy::many_single_char_names)]

use crate::core::{Matrix, Vector};

use super::QrFactors;
use super::triangular::{solve_upper, solve_upper_transposed};
use super::{Factorization, LinalgError};

/// Input cotangents produced by [`solve_vjp`].
#[derive(Debug, Clone, PartialEq)]
pub struct SolveGradients {
    /// Cotangent of the matrix: `A_bar = -b_bar x^T`.
    pub a_bar: Matrix,
    /// Cotangent of the right-hand side: `b_bar = A^{-T} x_bar`.
    pub b_bar: Vector,
}

/// Reverse-mode rule for `x = A^{-1} b`.
///
/// `factors` and `x` must come from the primal solve.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when `x` disagrees with the factors;
/// propagates dimension errors from the transposed solve.
pub fn solve_vjp<F: Factorization>(
    factors: &F,
    x: &Vector,
    x_bar: &Vector,
) -> Result<SolveGradients, LinalgError> {
    let n = factors.dimension();
    if x.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: x.len(),
            cols: 1,
        });
    }
    let b_bar = factors.solve_transposed(x_bar)?;
    let a_bar = Matrix::from_fn(n, n, |i, j| -b_bar[i] * x[j]);
    Ok(SolveGradients { a_bar, b_bar })
}

/// Forward-mode rule for `x = A^{-1} b`.
///
/// `factors` and `x` must come from the primal solve.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when tangent shapes disagree with the
/// factors; propagates solve errors.
pub fn solve_jvp<F: Factorization>(
    factors: &F,
    x: &Vector,
    a_dot: &Matrix,
    b_dot: &Vector,
) -> Result<Vector, LinalgError> {
    let n = factors.dimension();
    if x.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: x.len(),
            cols: 1,
        });
    }
    if a_dot.rows() != n || a_dot.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a_dot.rows(),
            cols: a_dot.cols(),
        });
    }
    if b_dot.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: b_dot.len(),
            cols: 1,
        });
    }
    let rhs = b_dot - &(a_dot * x);
    factors.solve(&rhs)
}

/// Validates shapes shared by the least-squares rules; returns `(m, n)`.
const fn check_lstsq_shapes(
    factors: &QrFactors,
    a: &Matrix,
    b: &Vector,
    x: &Vector,
) -> Result<(usize, usize), LinalgError> {
    let m = factors.rows();
    let n = factors.cols();
    if a.rows() != m || a.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a.rows(),
            cols: a.cols(),
        });
    }
    if b.len() != m {
        return Err(LinalgError::DimensionMismatch {
            rows: b.len(),
            cols: 1,
        });
    }
    if x.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: x.len(),
            cols: 1,
        });
    }
    Ok((m, n))
}

/// Solves `(AᵀA) out = w` as `R⁻¹ R⁻ᵀ w` — two triangular solves against
/// the QR factor, never forming `AᵀA`.
fn solve_normal(r: &Matrix, w: &Vector) -> Result<Vector, LinalgError> {
    let t = solve_upper_transposed(r, w, false)?;
    solve_upper(r, &t, false)
}

/// Reverse-mode rule for full-rank least squares `x = argmin ‖A x − b‖₂`.
///
/// With residual `r = b − A x` and `z = (AᵀA)⁻¹ x̄`:
/// `b̄ = A z`, `Ā = −(A z) xᵀ + r zᵀ`.
///
/// `factors`, `a`, `b`, `x` must come from the primal solve.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when any operand disagrees with the
/// factors; propagates triangular-solve errors.
pub fn lstsq_vjp(
    factors: &QrFactors,
    a: &Matrix,
    b: &Vector,
    x: &Vector,
    x_bar: &Vector,
) -> Result<SolveGradients, LinalgError> {
    let (m, n) = check_lstsq_shapes(factors, a, b, x)?;
    if x_bar.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: x_bar.len(),
            cols: 1,
        });
    }
    let r_mat = factors.r();
    let z = solve_normal(&r_mat, x_bar)?;
    let b_bar = a * &z;
    let ax = a * x;
    let resid = b - &ax;
    let a_bar = Matrix::from_fn(m, n, |i, j| -(b_bar[i] * x[j]) + resid[i] * z[j]);
    Ok(SolveGradients { a_bar, b_bar })
}

/// Forward-mode rule for full-rank least squares.
///
/// `ẋ = (AᵀA)⁻¹ (Ȧᵀ r − Aᵀ Ȧ x + Aᵀ ḃ)` with residual `r = b − A x`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when any operand disagrees with the
/// factors; propagates triangular-solve errors.
pub fn lstsq_jvp(
    factors: &QrFactors,
    a: &Matrix,
    b: &Vector,
    x: &Vector,
    a_dot: &Matrix,
    b_dot: &Vector,
) -> Result<Vector, LinalgError> {
    let (m, n) = check_lstsq_shapes(factors, a, b, x)?;
    if a_dot.rows() != m || a_dot.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a_dot.rows(),
            cols: a_dot.cols(),
        });
    }
    if b_dot.len() != m {
        return Err(LinalgError::DimensionMismatch {
            rows: b_dot.len(),
            cols: 1,
        });
    }
    let ax = a * x;
    let resid = b - &ax;
    let at = a.transpose();
    let adot_t = a_dot.transpose();
    // t = Ȧᵀ r − Aᵀ Ȧ x + Aᵀ ḃ
    let term1 = &adot_t * &resid;
    let adot_x = a_dot * x;
    let term2 = &at * &adot_x;
    let term3 = &at * b_dot;
    let t = &(&term1 - &term2) + &term3;
    let r_mat = factors.r();
    solve_normal(&r_mat, &t)
}
