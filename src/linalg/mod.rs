//! Linear-algebra primitives with Mercury-owned derivative rules.

mod adjoint;
mod cholesky;
mod error;
mod factorization;
mod fixed;
mod lu;
mod qr;
pub mod reductions;
mod triangular;

pub use adjoint::{SolveGradients, lstsq_jvp, lstsq_vjp, solve_jvp, solve_vjp};
pub use cholesky::{LdltFactors, LltFactors, ldlt_factor, llt_factor};
pub use error::LinalgError;
pub use factorization::Factorization;
pub use fixed::{solve_fixed, solve_fixed_unchecked, solve_spd_fixed_unchecked};
pub use lu::{LuFactors, lu_factor, solve};
pub use qr::{QrFactors, qr_factor};

/// Pivot magnitudes below this are treated as singular.
pub(crate) const PIVOT_TOLERANCE: f64 = 1.0e-12;
