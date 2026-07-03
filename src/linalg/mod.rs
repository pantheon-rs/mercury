//! Linear-algebra primitives with Mercury-owned derivative rules.

mod adjoint;
mod cholesky;
mod error;
mod factorization;
mod fixed;
mod lu;
mod triangular;

pub use adjoint::{SolveGradients, solve_jvp, solve_vjp};
pub use cholesky::{LltFactors, llt_factor};
pub use error::LinalgError;
pub use factorization::Factorization;
pub use fixed::{solve_fixed, solve_fixed_unchecked};
pub use lu::{LuFactors, lu_factor, solve};

/// Pivot magnitudes below this are treated as singular.
pub(crate) const PIVOT_TOLERANCE: f64 = 1.0e-12;
