//! The adjoint rule for linear solves — Mercury's first owned derivative
//! rule (decision 0003).
//!
//! For `x = A^{-1} b`:
//! - reverse (VJP), given output cotangent `x_bar`:
//!   `b_bar = A^{-T} x_bar`, `A_bar = -b_bar x^T`
//! - forward (JVP), given input tangents `(A_dot, b_dot)`:
//!   `x_dot = A^{-1} (b_dot - A_dot x)`
//!
//! Both reuse the primal [`LuFactors`] — no factorization is ever
//! differentiated, and the factorization backend can change freely.

use crate::core::{Matrix, Vector};

use super::{LinalgError, LuFactors};

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
/// Propagates dimension errors from the transposed solve.
pub fn solve_vjp(
    factors: &LuFactors,
    x: &Vector,
    x_bar: &Vector,
) -> Result<SolveGradients, LinalgError> {
    let b_bar = factors.solve_transposed(x_bar)?;
    let n = factors.dimension();
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
pub fn solve_jvp(
    factors: &LuFactors,
    x: &Vector,
    a_dot: &Matrix,
    b_dot: &Vector,
) -> Result<Vector, LinalgError> {
    let n = factors.dimension();
    if a_dot.rows() != n || a_dot.cols() != n {
        return Err(LinalgError::DimensionMismatch { rows: a_dot.rows(), cols: a_dot.cols() });
    }
    let rhs = &*b_dot - &(a_dot * x);
    factors.solve(&rhs)
}
