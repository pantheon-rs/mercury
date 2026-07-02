//! Linear-algebra primitives with Mercury-owned derivative rules.

mod error;
mod fixed;

pub use error::LinalgError;
pub use fixed::{solve_fixed, solve_fixed_unchecked};
