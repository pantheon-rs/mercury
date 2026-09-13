//! Compiled differentiable functions and runtime numerical plans.
#![forbid(unsafe_code)]

mod error;
mod kernel;
mod operator;
mod plan;
mod solve;

pub use error::{Error, Result};
pub use kernel::Kernel;
pub use mercury_macros::{differentiable, function};
pub use operator::{Operator, OperatorWorkspace, Shape};
pub use plan::{Linearization, NodeId, Plan, PlanBuilder, Source, Workspace};
pub use solve::{DenseSolve, ImplicitSolve};

/// Support for generated code; not a user-facing API.
#[doc(hidden)]
pub mod __private {
    pub use crate::error::check_finite;
}
