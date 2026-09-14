//! Numerical implementations and their reusable local storage.

use crate::Result;

/// Dimensions of a numerical map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Number of active scalar inputs.
    pub inputs: usize,
    /// Number of scalar outputs.
    pub outputs: usize,
}

/// An immutable numerical implementation, shared by plan evaluations.
///
/// Calculations must be deterministic and free of external side effects: plans
/// may prune unreachable instances and derivative calls may replay their values.
pub trait Operator: Send + Sync {
    /// Fixed dimensions of this instance.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn shape(&self) -> Shape;

    /// Highest implemented derivative order: 0 (values), 1 (JVP/VJP), or 2
    /// (also weighted curvature). This describes rules, not smoothness at a point.
    /// Custom second-order operators must override the first-order default.
    /// See the [example](crate::advanced#derivative-capabilities).
    fn derivative_order(&self) -> u8 {
        1
    }

    /// Allocate local storage once for a plan workspace.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_>;

    /// Conservative dependency of one output on one input.
    /// Returning false promises independence everywhere in the supported domain,
    /// including all branches. Numerical zeros at one point are insufficient.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn depends_on(&self, _output: usize, _input: usize) -> bool {
        true
    }
}

/// Local buffers and numerical caches for one operator instance.
///
/// Plans supply dimension-checked, finite inputs and preserve the same point
/// between `linearize` and derivative calls. Implementations overwrite outputs
/// and preserve inputs and seeds. Numerical errors invalidate the enclosing
/// linearization. These low-level methods are normally called through a plan.
///
/// # Errors
/// Methods return domain, non-finite, or solver errors. After a numerical
/// failure the caller must invalidate this workspace before reusing it.
#[allow(clippy::missing_errors_doc)] // The shared failure contract applies to every method.
pub trait OperatorWorkspace {
    /// Evaluate without requiring derivative caches.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()>;

    /// Evaluate and prepare any caches needed for derivatives at this point.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn linearize(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.evaluate(input, output)
    }

    /// Write a Jacobian-vector product.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()>;

    /// Write a transpose-Jacobian-vector product.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn vjp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()>;

    /// Write the Hessian-vector product of the fixed weighted output sum.
    /// The result is `D(J(input)^T weights)[direction]`; weights stay constant.
    /// See the [example](crate::advanced#second-order-products).
    /// Operators without this rule return [`crate::Error::UnsupportedDerivative`]; no numerical fallback is used.
    fn curvature(
        &mut self,
        _input: &[f64],
        _weights: &[f64],
        _direction: &[f64],
        _output: &mut [f64],
    ) -> Result<()> {
        Err(crate::Error::UnsupportedDerivative)
    }

    /// Apply contiguous seed rows; implementations may use compiled batch widths.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn jvp_batch(
        &mut self,
        input: &[f64],
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let inputs = input.len();
        let outputs = output.len() / count;
        for row in 0..count {
            self.jvp(
                input,
                &seeds[row * inputs..(row + 1) * inputs],
                &mut output[row * outputs..(row + 1) * outputs],
            )?;
        }
        Ok(())
    }

    /// Apply contiguous cotangent rows, preserving all caller seeds.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn vjp_batch(
        &mut self,
        input: &[f64],
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let inputs = input.len();
        let outputs = seeds.len() / count;
        for row in 0..count {
            self.vjp(
                input,
                &seeds[row * outputs..(row + 1) * outputs],
                &mut output[row * inputs..(row + 1) * inputs],
            )?;
        }
        Ok(())
    }

    /// Discard numerical caches after a failed evaluation or derivative.
    ///
    /// See the [example](crate::advanced#operator-contract).
    fn invalidate(&mut self) {}
}
