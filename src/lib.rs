//! Compiled differentiable operators and runtime numerical plans.
#![forbid(unsafe_code)]

mod error;
mod kernel;
mod operator;
mod plan;
mod solve;

pub use error::{Error, Result};
pub use kernel::Kernel;
pub use mercury_macros::differentiable;
pub use operator::{Operator, OperatorWorkspace, Shape};
pub use plan::{Linearization, NodeId, Plan, PlanBuilder, Source, Workspace};
pub use solve::{DenseSolve, ImplicitSolve};
