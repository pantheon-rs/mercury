//! Cholesky factorizations for symmetric positive-definite systems.
//!
//! Only the lower triangle of the input is read (faer's `Side::Lower`
//! convention with the side fixed); symmetry is the caller's contract.
//! Callers never differentiate through this code — derivatives come from
//! the generic adjoint rule over [`Factorization`](super::Factorization).

use crate::core::{Matrix, Vector};

use super::triangular::{solve_lower, solve_lower_transposed};
use super::{LinalgError, PIVOT_TOLERANCE};

/// Reusable LLT factors of an SPD matrix (`A = L Lᵀ`, `L` lower).
#[derive(Debug, Clone)]
pub struct LltFactors {
    /// Lower-triangular factor; the strict upper triangle is zero.
    l: Matrix,
}

/// Factors a symmetric positive-definite matrix as `A = L Lᵀ`.
///
/// Reads only the lower triangle of `a`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] for non-square input;
/// [`LinalgError::NotPositiveDefinite`] when a pivot `l_jj²` falls at or
/// below tolerance (the matrix is indefinite or numerically singular).
pub fn llt_factor(a: &Matrix) -> Result<LltFactors, LinalgError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let mut l = Matrix::zeros(n, n);
    for j in 0..n {
        // Diagonal: l_jj = sqrt(a_jj - sum_k l_jk^2).
        let mut sum = a[(j, j)];
        for k in 0..j {
            sum -= l[(j, k)] * l[(j, k)];
        }
        if sum <= PIVOT_TOLERANCE {
            return Err(LinalgError::NotPositiveDefinite { pivot_index: j });
        }
        let ljj = sum.sqrt();
        l[(j, j)] = ljj;
        // Below-diagonal column j.
        for i in (j + 1)..n {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / ljj;
        }
    }
    Ok(LltFactors { l })
}

impl LltFactors {
    /// Side length of the factored matrix.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.l.rows()
    }

    /// The lower-triangular factor `L`.
    #[must_use]
    pub const fn l(&self) -> &Matrix {
        &self.l
    }

    /// Solves `A x = b` via `L y = b`, then `Lᵀ x = y`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let y = solve_lower(&self.l, b, false)?;
        solve_lower_transposed(&self.l, &y, false)
    }
}

impl super::Factorization for LltFactors {
    fn dimension(&self) -> usize {
        Self::dimension(self)
    }

    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve(self, b)
    }

    /// `A` is symmetric, so `Aᵀ x = b` is the same solve.
    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve(self, b)
    }
}
