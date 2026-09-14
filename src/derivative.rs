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
        value: vec![f64::NAN; operator.shape().outputs],
        weights: vec![0.0; operator.shape().outputs],
        product: vec![0.0; operator.shape().inputs],
    })
}

struct DerivativeWorkspace<'a> {
    base: Box<dyn OperatorWorkspace + 'a>,
    shape: Shape,
    gradient: bool,
    value: Vec<f64>,
    weights: Vec<f64>,
    product: Vec<f64>,
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
        self.value.fill(f64::NAN);
        self.base.linearize(input, &mut self.value)?;
        check_finite("original function output", &self.value)?;
        if self.shape.inputs < self.shape.outputs {
            for column in 0..self.shape.inputs {
                self.product.fill(0.0);
                self.product[column] = 1.0;
                self.value.fill(f64::NAN);
                self.base.jvp(input, &self.product, &mut self.value)?;
                check_finite("derivative column", &self.value)?;
                for (row, value) in self.value.iter().enumerate() {
                    output[row * self.shape.inputs + column] = *value;
                }
            }
        } else {
            for row in 0..self.shape.outputs {
                self.weights.fill(0.0);
                self.weights[row] = 1.0;
                self.base.vjp(
                    input,
                    &self.weights,
                    &mut output[row * self.shape.inputs..(row + 1) * self.shape.inputs],
                )?;
            }
        }
        check_finite("derivative output", output)
    }

    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        self.check(input, output)?;
        check_len("derivative seed", seed.len(), self.shape.inputs)?;
        check_finite("derivative seed", seed)?;
        for row in 0..self.shape.outputs {
            self.weights.fill(0.0);
            self.weights[row] = 1.0;
            self.base.curvature(
                input,
                &self.weights,
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
        for row in 0..self.shape.outputs {
            self.weights.fill(0.0);
            self.weights[row] = 1.0;
            self.product.fill(f64::NAN);
            // Each output Hessian is symmetric on the kernel's smooth domain.
            self.base.curvature(
                input,
                &self.weights,
                &seed[row * self.shape.inputs..(row + 1) * self.shape.inputs],
                &mut self.product,
            )?;
            for (entry, &value) in output.iter_mut().zip(&self.product) {
                *entry += value;
            }
        }
        check_finite("derivative cotangent", output)
    }

    fn invalidate(&mut self) {
        self.base.invalidate();
    }
}
