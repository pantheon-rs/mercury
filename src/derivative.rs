//! First derivatives as operators. Their products use the original curvature rule.

use crate::error::{check_finite, check_len};
use crate::{Operator, OperatorWorkspace, Result, Shape};

#[doc(hidden)]
pub fn derivative_workspace(
    operator: &dyn Operator,
    gradient: bool,
) -> Box<dyn OperatorWorkspace + '_> {
    Box::new(DerivativeWorkspace {
        base: operator.workspace(),
        shape: operator.shape(),
        gradient,
    })
}

struct DerivativeWorkspace<'a> {
    base: Box<dyn OperatorWorkspace + 'a>,
    shape: Shape,
    gradient: bool,
}

impl DerivativeWorkspace<'_> {
    fn check(&self, input: &[f64], output: &[f64]) -> Result<()> {
        check_len("derivative input", input.len(), self.shape.inputs)?;
        if self.gradient {
            check_len("gradient outputs", self.shape.outputs, 1)?;
        }
        check_len(
            "derivative output",
            output.len(),
            self.shape
                .inputs
                .checked_mul(self.shape.outputs)
                .ok_or(crate::Error::SizeOverflow)?,
        )?;
        check_finite("derivative input", input)
    }
}

impl OperatorWorkspace for DerivativeWorkspace<'_> {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.check(input, output)?;
        let mut value = vec![f64::NAN; self.shape.outputs];
        self.base.linearize(input, &mut value)?;
        check_finite("original function output", &value)?;
        let mut weights = vec![0.0; self.shape.outputs];
        for row in 0..self.shape.outputs {
            weights.fill(0.0);
            weights[row] = 1.0;
            self.base.vjp(
                input,
                &weights,
                &mut output[row * self.shape.inputs..(row + 1) * self.shape.inputs],
            )?;
        }
        check_finite("derivative output", output)
    }

    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        self.check(input, output)?;
        check_len("derivative seed", seed.len(), self.shape.inputs)?;
        check_finite("derivative seed", seed)?;
        let mut weights = vec![0.0; self.shape.outputs];
        for row in 0..self.shape.outputs {
            weights.fill(0.0);
            weights[row] = 1.0;
            self.base.curvature(
                input,
                &weights,
                seed,
                &mut output[row * self.shape.inputs..(row + 1) * self.shape.inputs],
            )?;
        }
        check_finite("derivative tangent", output)
    }

    fn vjp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        self.check(input, seed)?;
        check_len(
            "derivative cotangent output",
            output.len(),
            self.shape.inputs,
        )?;
        check_finite("derivative seed", seed)?;
        output.fill(0.0);
        let mut weights = vec![0.0; self.shape.outputs];
        let mut product = vec![0.0; self.shape.inputs];
        for row in 0..self.shape.outputs {
            weights.fill(0.0);
            weights[row] = 1.0;
            // Each output Hessian is symmetric on the kernel's smooth domain.
            self.base.curvature(
                input,
                &weights,
                &seed[row * self.shape.inputs..(row + 1) * self.shape.inputs],
                &mut product,
            )?;
            for (entry, &value) in output.iter_mut().zip(&product) {
                *entry += value;
            }
        }
        check_finite("derivative cotangent", output)
    }

    fn invalidate(&mut self) {
        self.base.invalidate();
    }
}
