#![doc = include_str!("../docs/advanced.md")]

pub use crate::kernel::Kernel;
pub use crate::operator::{Operator, OperatorWorkspace, Shape};
pub use crate::plan::{Linearization, Workspace};
pub use mercury_macros::differentiable;

use crate::Result;

/// Explicit execution and structural inspection of a numerical plan.
///
/// ```
/// use mercury::{Plan, Source};
/// use mercury::advanced::PlanExecution;
/// let plan = Plan::builder(1).build([Source::Input(0)])?;
/// let mut workspace = mercury::advanced::Workspace::new(&plan);
/// let point = [3.0];
/// let mut linearization = plan.linearize(&point, &mut workspace)?;
/// let mut gradient = [0.0];
/// linearization.vjp(&[1.0], &mut gradient)?;
/// assert_eq!(gradient, [1.0]);
/// # Ok::<(), mercury::Error>(())
/// ```
pub trait PlanExecution {
    /// Identify a plan's structure; replay must retain the plan itself too.
    ///
    /// See the [example](crate::advanced#plan-execution).
    fn epoch(&self) -> u64;

    /// Return conservative input dependencies, one sorted list per output.
    ///
    /// See the [example](crate::advanced#plan-execution).
    fn dependencies(&self) -> &[Vec<usize>];

    /// Evaluate into caller-owned output storage.
    ///
    /// See the [example](crate::advanced#plan-execution).
    ///
    /// # Errors
    /// Rejects invalid inputs or a foreign workspace; propagates numerical errors.
    fn evaluate(
        &self,
        point: &[f64],
        workspace: &mut Workspace<'_>,
        output: &mut [f64],
    ) -> Result<()>;

    /// Prepare derivatives at one point, borrowing that point and the workspace.
    ///
    /// See the [example](crate::advanced#plan-execution).
    ///
    /// # Errors
    /// Rejects invalid inputs or a foreign workspace; propagates numerical errors.
    fn linearize<'a, 'p>(
        &'p self,
        point: &'a [f64],
        workspace: &'a mut Workspace<'p>,
    ) -> Result<Linearization<'a, 'p>>;
}
