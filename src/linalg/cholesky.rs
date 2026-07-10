//! Cholesky factorizations for symmetric positive-definite systems.
//!
//! Only the lower triangle of the input is read (faer's `Side::Lower`
//! convention with the side fixed); symmetry is the caller's contract.
//! Callers never differentiate through this code — derivatives come from
//! the generic adjoint rule over [`Factorization`](super::Factorization).
#![allow(clippy::many_single_char_names)]

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

    /// Determinant of the factored matrix: `Π l_ii²`.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let mut det = 1.0;
        for i in 0..self.dimension() {
            det *= self.l[(i, i)] * self.l[(i, i)];
        }
        det
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

/// Reusable LDLT factors of a symmetric matrix (`A = L D Lᵀ`, `L`
/// unit-lower, `D` diagonal).
///
/// Unpivoted: exact for any symmetric matrix whose leading principal minors
/// are nonsingular, and tolerant of indefinite `D`, but can be inaccurate on
/// strongly indefinite matrices — Bunch-Kaufman pivoting is the deferred
/// answer for those (spec non-goal).
#[derive(Debug, Clone)]
pub struct LdltFactors {
    /// Unit-lower factor; multipliers strictly below the diagonal (the
    /// diagonal itself is implied 1 and the stored values are unused).
    l: Matrix,
    /// Diagonal of `D`.
    d: Vector,
}

/// Factors a symmetric matrix as `A = L D Lᵀ` without pivoting.
///
/// Reads only the lower triangle of `a`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] for non-square input;
/// [`LinalgError::NotPositiveDefinite`] when `|d_j|` falls at or below
/// tolerance (breakdown of the unpivoted recurrence).
pub fn ldlt_factor(a: &Matrix) -> Result<LdltFactors, LinalgError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let mut l = Matrix::zeros(n, n);
    let mut d = Vector::zeros(n);
    for j in 0..n {
        // d_j = a_jj - sum_k l_jk^2 d_k.
        let mut dj = a[(j, j)];
        for k in 0..j {
            dj -= l[(j, k)] * l[(j, k)] * d[k];
        }
        if dj.abs() <= PIVOT_TOLERANCE {
            return Err(LinalgError::NotPositiveDefinite { pivot_index: j });
        }
        d[j] = dj;
        // l_ij = (a_ij - sum_k l_ik l_jk d_k) / d_j.
        for i in (j + 1)..n {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)] * d[k];
            }
            l[(i, j)] = s / dj;
        }
    }
    Ok(LdltFactors { l, d })
}

impl LdltFactors {
    /// Side length of the factored matrix.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.l.rows()
    }

    /// The unit-lower factor `L` (diagonal implied 1).
    #[must_use]
    pub const fn l(&self) -> &Matrix {
        &self.l
    }

    /// The diagonal of `D`.
    #[must_use]
    pub const fn d(&self) -> &Vector {
        &self.d
    }

    /// Solves `A x = b` via `L z = b`, `D w = z`, `Lᵀ x = w`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        let z = solve_lower(&self.l, b, true)?;
        let mut w = Vector::zeros(n);
        for i in 0..n {
            w[i] = z[i] / self.d[i];
        }
        solve_lower_transposed(&self.l, &w, true)
    }
}

impl super::Factorization for LdltFactors {
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
