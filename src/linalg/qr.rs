//! Householder QR factorization with least-squares solve.
//!
//! LAPACK-style compact storage: reflectors below the diagonal, `R` on and
//! above it, `tau` coefficients in a side vector. Callers never
//! differentiate through this code — least-squares derivatives come from
//! the dedicated adjoint rule (`lstsq_vjp`/`lstsq_jvp`).
#![allow(clippy::many_single_char_names)]

use crate::core::{Matrix, Vector};

use super::triangular::solve_upper;
use super::{LinalgError, PIVOT_TOLERANCE};

/// Reusable Householder QR factors of an `m x n` matrix, `m >= n`.
#[derive(Debug, Clone)]
pub struct QrFactors {
    /// Compact storage: `R` on/above the diagonal, reflector tails below.
    qr: Matrix,
    /// Householder coefficients, one per column.
    tau: Vector,
}

/// Factors `A = Q R` by Householder reflections. Requires `m >= n`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when `m < n`;
/// [`LinalgError::RankDeficient`] when a diagonal of `R` falls below
/// tolerance (numerically dependent columns).
pub fn qr_factor(a: &Matrix) -> Result<QrFactors, LinalgError> {
    let m = a.rows();
    let n = a.cols();
    if m < n {
        return Err(LinalgError::DimensionMismatch {
            rows: m,
            cols: n,
        });
    }

    let mut qr = a.clone();
    let mut tau = Vector::zeros(n);

    for k in 0..n {
        // Column norm over rows k..m.
        let x0 = qr[(k, k)];
        let mut sigma = 0.0;
        for i in (k + 1)..m {
            sigma += qr[(i, k)] * qr[(i, k)];
        }
        let norm = (x0 * x0 + sigma).sqrt();
        if norm < PIVOT_TOLERANCE {
            return Err(LinalgError::RankDeficient { column: k });
        }
        // beta = -sign(x0) * norm avoids cancellation in x0 - beta.
        let beta = if x0 >= 0.0 { -norm } else { norm };
        let tau_k = (beta - x0) / beta;
        let denom = x0 - beta;
        // Store the reflector tail (v0 = 1 implied) and R's diagonal.
        for i in (k + 1)..m {
            qr[(i, k)] /= denom;
        }
        qr[(k, k)] = beta;
        tau[k] = tau_k;

        // Apply H = I - tau v v^T to the trailing columns.
        for j in (k + 1)..n {
            let mut w = qr[(k, j)];
            for i in (k + 1)..m {
                w += qr[(i, k)] * qr[(i, j)];
            }
            let tw = tau_k * w;
            qr[(k, j)] -= tw;
            for i in (k + 1)..m {
                let delta = tw * qr[(i, k)];
                qr[(i, j)] -= delta;
            }
        }
    }

    Ok(QrFactors { qr, tau })
}

impl QrFactors {
    /// Row count of the factored matrix.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.qr.rows()
    }

    /// Column count of the factored matrix.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.qr.cols()
    }

    /// The `n x n` upper-triangular factor `R` (thin), as a copy.
    #[must_use]
    pub fn r(&self) -> Matrix {
        let n = self.cols();
        Matrix::from_fn(n, n, |i, j| if j >= i { self.qr[(i, j)] } else { 0.0 })
    }

    /// Applies `Qᵀ` to a length-`m` vector via the stored reflectors.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn q_transpose_apply(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let m = self.rows();
        let n = self.cols();
        if b.len() != m {
            return Err(LinalgError::DimensionMismatch {
                rows: b.len(),
                cols: 1,
            });
        }
        let mut y = b.clone();
        for k in 0..n {
            let mut w = y[k];
            for i in (k + 1)..m {
                w += self.qr[(i, k)] * y[i];
            }
            let tw = self.tau[k] * w;
            y[k] -= tw;
            for i in (k + 1)..m {
                let delta = tw * self.qr[(i, k)];
                y[i] -= delta;
            }
        }
        Ok(y)
    }

    /// Solves square `A x = b` via `R x = Qᵀ b`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when the factored matrix is not
    /// square or `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let m = self.rows();
        let n = self.cols();
        if m != n {
            return Err(LinalgError::DimensionMismatch { rows: m, cols: n });
        }
        let y = self.q_transpose_apply(b)?;
        solve_upper(&self.qr, &y, false)
    }

    /// Solves the full-rank least-squares problem `min ‖A x − b‖₂`
    /// via `R x = (Qᵀ b)[0..n]`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve_lstsq(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let n = self.cols();
        let y = self.q_transpose_apply(b)?;
        // Back-substitute R x = y[0..n] directly on the compact storage
        // (R's rows live in the top n rows of qr).
        let mut x = Vector::zeros(n);
        for i in (0..n).rev() {
            let mut acc = y[i];
            for j in (i + 1)..n {
                acc -= self.qr[(i, j)] * x[j];
            }
            x[i] = acc / self.qr[(i, i)];
        }
        Ok(x)
    }
}
