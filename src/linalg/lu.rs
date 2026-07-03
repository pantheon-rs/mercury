//! Dynamic partial-pivot LU factorization — the backend seam.
//!
//! This hand-rolled factorization is Mercury's initial backend; the public
//! contract (`lu_factor`, `LuFactors::solve`) is designed so a faer backend
//! can replace the internals without any caller change. Callers never
//! differentiate through this code — derivatives come from the adjoint rule
//! (`solve_vjp`/`solve_jvp`).

use crate::core::{Matrix, Vector};

use super::{LinalgError, PIVOT_TOLERANCE};

/// Reusable LU factors of a square matrix (`P A = L U`).
#[derive(Debug, Clone)]
pub struct LuFactors {
    /// Combined L (unit lower, below diagonal) and U (upper) storage.
    lu: Matrix,
    /// `perm[i]` = original row of `A` occupying position `i`.
    perm: Vec<usize>,
}

/// Factors a square matrix with partial pivoting.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] for non-square input;
/// [`LinalgError::Singular`] when a pivot falls below tolerance.
pub fn lu_factor(a: &Matrix) -> Result<LuFactors, LinalgError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_mag = lu[(k, k)].abs();
        for i in (k + 1)..n {
            let mag = lu[(i, k)].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_mag < PIVOT_TOLERANCE {
            return Err(LinalgError::Singular { pivot_index: k });
        }
        if pivot_row != k {
            for j in 0..n {
                let tmp = lu[(k, j)];
                lu[(k, j)] = lu[(pivot_row, j)];
                lu[(pivot_row, j)] = tmp;
            }
            perm.swap(k, pivot_row);
        }

        for i in (k + 1)..n {
            let factor = lu[(i, k)] / lu[(k, k)];
            lu[(i, k)] = factor; // store the L multiplier in place
            for j in (k + 1)..n {
                let delta = factor * lu[(k, j)];
                lu[(i, j)] -= delta;
            }
        }
    }

    Ok(LuFactors { lu, perm })
}

impl LuFactors {
    /// Side length of the factored matrix.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.perm.len()
    }

    /// Solves `A x = b` using the stored factors.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch {
                rows: b.len(),
                cols: 1,
            });
        }

        // Forward: L y = P b (L unit lower).
        let mut y = Vector::zeros(n);
        for i in 0..n {
            let mut acc = b[self.perm[i]];
            for j in 0..i {
                acc -= self.lu[(i, j)] * y[j];
            }
            y[i] = acc;
        }
        // Backward: U x = y.
        for i in (0..n).rev() {
            let mut acc = y[i];
            for j in (i + 1)..n {
                acc -= self.lu[(i, j)] * y[j];
            }
            y[i] = acc / self.lu[(i, i)];
        }
        Ok(y)
    }

    /// Solves `A^T z = c` using the same factors
    /// (`A^T = U^T L^T P`, so solve `U^T w = c`, then `L^T v = w`, then
    /// undo the permutation).
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `c` has the wrong length.
    pub fn solve_transposed(&self, c: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        if c.len() != n {
            return Err(LinalgError::DimensionMismatch {
                rows: c.len(),
                cols: 1,
            });
        }

        // Forward: U^T w = c (U^T lower, non-unit diagonal).
        let mut w = Vector::zeros(n);
        for i in 0..n {
            let mut acc = c[i];
            for j in 0..i {
                acc -= self.lu[(j, i)] * w[j];
            }
            w[i] = acc / self.lu[(i, i)];
        }
        // Backward: L^T v = w (L^T upper, unit diagonal).
        for i in (0..n).rev() {
            let mut acc = w[i];
            for j in (i + 1)..n {
                acc -= self.lu[(j, i)] * w[j];
            }
            w[i] = acc;
        }
        // P x = v  =>  x[perm[i]] = v[i].
        let mut x = Vector::zeros(n);
        for i in 0..n {
            x[self.perm[i]] = w[i];
        }
        Ok(x)
    }
}

/// Convenience: factor + solve in one call.
///
/// # Errors
///
/// Propagates [`lu_factor`] and [`LuFactors::solve`] errors.
pub fn solve(a: &Matrix, b: &Vector) -> Result<Vector, LinalgError> {
    lu_factor(a)?.solve(b)
}
