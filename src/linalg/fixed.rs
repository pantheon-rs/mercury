//! Kernel-safe fixed-size dense solve (partial-pivot Gaussian elimination).

use crate::core::{SMatrix, SVector};

use super::{LinalgError, PIVOT_TOLERANCE};

/// Solves `A x = b` for small fixed-size systems on the stack, without a
/// singularity check.
///
/// Written in the Enzyme-safe POD style (element-wise construction, no bulk
/// array copies/swaps), so it is valid to differentiate *through* this
/// function inside a kernel: pivot choices are piecewise-constant in the
/// inputs, and Enzyme differentiates the taken branch. For problem-scale
/// systems use the dynamic [`solve`](crate::linalg::solve) primitive and its
/// adjoint rule instead.
///
/// This is the kernel-facing entry point: unlike [`solve_fixed`], it never
/// returns `Result`. `Result<SVector<N>, LinalgError>` wraps the solution
/// aggregate in an enum, and returning that shape from a function reachable
/// from an `#[autodiff_reverse]` kernel fails to compile — Enzyme cannot
/// deduce a type for the `llvm.memcpy` that materializes the `Ok` payload
/// (empirically confirmed 2026-07-02: a minimal scratch kernel calling a
/// `SVector`-returning helper compiles and differentiates correctly, while
/// the same helper wrapped in `Result<SVector<N>, E>` fails with "Enzyme:
/// Cannot deduce type of copy"). `solve_fixed` below hits exactly that
/// failure when its `Result` return is used inside an autodiff kernel.
///
/// If the pivot column is singular (no entry at or below the diagonal
/// clears `PIVOT_TOLERANCE`), this function does **not** error: the
/// elimination divides by (near-)zero and the affected output components
/// become `NaN`/`±inf`, propagating through back-substitution. Callers that
/// need a hard error (host-side code, not inside a differentiated kernel)
/// should use [`solve_fixed`] instead.
#[must_use]
pub fn solve_fixed_unchecked<const N: usize>(a: &SMatrix<N, N>, b: &SVector<N>) -> SVector<N> {
    // Working copies in the Enzyme-safe shape (Global Constraints rules
    // 1-4): from_fn is kernel-safe, mutation goes through Index/IndexMut.
    let mut m = SMatrix::<N, N>::from_fn(|i, j| a[(i, j)]);
    let mut y = SVector::<N>::from_fn(|i| b[i]);

    for k in 0..N {
        // Partial pivot: largest magnitude in column k at or below row k.
        // No tolerance check here (see doc comment): a degenerate pivot
        // simply produces non-finite outputs downstream.
        let mut pivot_row = k;
        let mut pivot_mag = m[(k, k)].abs();
        for i in (k + 1)..N {
            let mag = m[(i, k)].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_row != k {
            // Element-wise row swap (mem::swap of arrays risks memcpy).
            for j in 0..N {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(pivot_row, j)];
                m[(pivot_row, j)] = tmp;
            }
            let tmp = y[k];
            y[k] = y[pivot_row];
            y[pivot_row] = tmp;
        }

        // Eliminate below the pivot.
        for i in (k + 1)..N {
            let factor = m[(i, k)] / m[(k, k)];
            m[(i, k)] = 0.0;
            for j in (k + 1)..N {
                let delta = factor * m[(k, j)];
                m[(i, j)] -= delta;
            }
            y[i] -= factor * y[k];
        }
    }

    // Back substitution, in place over y.
    for i in (0..N).rev() {
        let mut acc = y[i];
        for j in (i + 1)..N {
            acc -= m[(i, j)] * y[j];
        }
        y[i] = acc / m[(i, i)];
    }

    y
}

/// Solves `A x = b` for small fixed-size systems on the stack.
///
/// Host-facing wrapper around the same partial-pivot Gaussian elimination as
/// [`solve_fixed_unchecked`], with an explicit singularity check. This
/// duplicates rather than calls `solve_fixed_unchecked`, deliberately: the
/// early `Err` return here is exactly the shape that breaks Enzyme's type
/// analysis when used inside a differentiated kernel (see
/// [`solve_fixed_unchecked`]'s doc comment), so this function must not be
/// called on a kernel path. Duplicating the ~40-line elimination keeps the
/// kernel-safe path free of that risk regardless of future edits here.
///
/// # Errors
///
/// [`LinalgError::Singular`] when the best available pivot is below
/// tolerance.
pub fn solve_fixed<const N: usize>(
    a: &SMatrix<N, N>,
    b: &SVector<N>,
) -> Result<SVector<N>, LinalgError> {
    // Working copies in the Enzyme-safe shape (Global Constraints rules
    // 1-4): from_fn is kernel-safe, mutation goes through Index/IndexMut.
    let mut m = SMatrix::<N, N>::from_fn(|i, j| a[(i, j)]);
    let mut y = SVector::<N>::from_fn(|i| b[i]);

    for k in 0..N {
        // Partial pivot: largest magnitude in column k at or below row k.
        let mut pivot_row = k;
        let mut pivot_mag = m[(k, k)].abs();
        for i in (k + 1)..N {
            let mag = m[(i, k)].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_mag < PIVOT_TOLERANCE {
            return Err(LinalgError::Singular { pivot_index: k });
        }
        if pivot_row != k {
            // Element-wise row swap (mem::swap of arrays risks memcpy).
            for j in 0..N {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(pivot_row, j)];
                m[(pivot_row, j)] = tmp;
            }
            let tmp = y[k];
            y[k] = y[pivot_row];
            y[pivot_row] = tmp;
        }

        // Eliminate below the pivot.
        for i in (k + 1)..N {
            let factor = m[(i, k)] / m[(k, k)];
            m[(i, k)] = 0.0;
            for j in (k + 1)..N {
                let delta = factor * m[(k, j)];
                m[(i, j)] -= delta;
            }
            y[i] -= factor * y[k];
        }
    }

    // Back substitution, in place over y.
    for i in (0..N).rev() {
        let mut acc = y[i];
        for j in (i + 1)..N {
            acc -= m[(i, j)] * y[j];
        }
        y[i] = acc / m[(i, i)];
    }

    Ok(y)
}

/// Solves SPD `A x = b` for small fixed-size systems on the stack via
/// unpivoted Cholesky (LLT), without a definiteness check.
///
/// Kernel-facing (same contract family as [`solve_fixed_unchecked`]): no
/// `Result` return, written in the Enzyme-safe POD style. Cheaper than the
/// LU kernel path for SPD systems — no pivot search. Reads only the lower
/// triangle of `a`; symmetry is the caller's contract.
///
/// If `A` is not positive definite, the factorization takes the square
/// root of a non-positive number and the affected outputs become `NaN`,
/// propagating through the solves. Callers needing a hard error should
/// factor host-side with [`llt_factor`](crate::linalg::llt_factor).
#[must_use]
pub fn solve_spd_fixed_unchecked<const N: usize>(
    a: &SMatrix<N, N>,
    b: &SVector<N>,
) -> SVector<N> {
    // Working copy in the Enzyme-safe shape (Global Constraints rules 1-4).
    let mut l = SMatrix::<N, N>::from_fn(|i, j| a[(i, j)]);

    // In-place LLT on the lower triangle.
    for j in 0..N {
        let mut sum = l[(j, j)];
        for k in 0..j {
            sum -= l[(j, k)] * l[(j, k)];
        }
        let ljj = sum.sqrt(); // non-SPD => NaN, propagated by design
        l[(j, j)] = ljj;
        for i in (j + 1)..N {
            let mut s = l[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / ljj;
        }
    }

    // Forward: L y = b.
    let mut y = SVector::<N>::from_fn(|i| b[i]);
    for i in 0..N {
        let mut acc = y[i];
        for j in 0..i {
            acc -= l[(i, j)] * y[j];
        }
        y[i] = acc / l[(i, i)];
    }
    // Backward: Lᵀ x = y, in place over y.
    for i in (0..N).rev() {
        let mut acc = y[i];
        for j in (i + 1)..N {
            acc -= l[(j, i)] * y[j];
        }
        y[i] = acc / l[(i, i)];
    }
    y
}
