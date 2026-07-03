//! The `Factorization` abstraction: any factorization that can solve
//! `A x = b` and `Aᵀ x = b` powers the same adjoint rule
//! (`solve_vjp`/`solve_jvp`) — decision 0003's rule-owning joint,
//! factorization-agnostic by construction.

use crate::core::Vector;

use super::LinalgError;

/// A reusable factorization of a square matrix `A`.
pub trait Factorization {
    /// Side length of the factored matrix.
    fn dimension(&self) -> usize;

    /// Solves `A x = b`.
    ///
    /// # Errors
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length;
    /// [`LinalgError::Singular`] on numerical breakdown.
    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>;

    /// Solves `Aᵀ x = b`. For symmetric factorizations this is `solve`.
    ///
    /// # Errors
    /// Same as [`Factorization::solve`].
    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError>;
}
