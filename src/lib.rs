#![doc = include_str!("../docs/api.md")]
#![forbid(unsafe_code)]

mod derivative;
mod error;
mod kernel;
mod operator;
mod plan;
mod solve;
mod sparse;

pub use error::{Error, Result};
pub use mercury_macros::function;
pub use plan::{Gradient, Hessian, Jacobian, NodeId, Plan, PlanBuilder, Source};
pub use solve::{DenseSolve, ImplicitSolve, LinearSolveReport, NewtonReport};

pub mod advanced;

pub(crate) use advanced::{Operator, OperatorWorkspace, PlanExecution, Shape};

/// Support for generated code; not a user-facing API.
#[doc(hidden)]
pub mod __private {
    pub use crate::derivative::derivative_workspace;
    pub use crate::error::check_finite;
}
