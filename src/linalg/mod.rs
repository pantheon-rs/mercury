//! Linear-algebra primitives with Mercury-owned derivative rules.

mod error;
mod fixed;
mod lu;

pub use error::LinalgError;
pub use fixed::{solve_fixed, solve_fixed_unchecked};
pub use lu::{LuFactors, lu_factor, solve};
