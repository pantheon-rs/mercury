//! Checked adapters around concrete compiler-generated derivatives.

use crate::error::{check_finite, check_len};
use crate::{Error, Operator, OperatorWorkspace, Result, Shape};

type Primal<C> = fn(&C, &[f64], &mut [f64]);
type Forward<C> = fn(&C, &[f64], &[f64], &mut [f64], &mut [f64]);
type Reverse<C> = fn(&C, &[f64], &mut [f64], &mut [f64], &mut [f64]);
type Domain<C> = fn(&C, &[f64]) -> Result<()>;

/// A compiled numerical kernel with owned, immutable inactive configuration.
///
/// Usually constructed by [`crate::differentiable`]. Each workspace owns its
/// scratch buffers; generated reverse functions never consume caller seeds.
pub struct Kernel<C> {
    config: C,
    shape: Shape,
    primal: Primal<C>,
    forward: Forward<C>,
    reverse: Reverse<C>,
    domain: Option<Domain<C>>,
}

impl<C> Kernel<C> {
    /// Construct from primal and derivative callbacks in the Enzyme argument order.
    ///
    /// Callbacks must overwrite all primal outputs and implement the declared
    /// dimensions. Reverse callbacks accumulate into the supplied input shadow.
    /// Prefer the generated constructor from [`crate::differentiable`].
    pub const fn new(
        config: C,
        shape: Shape,
        primal: Primal<C>,
        forward: Forward<C>,
        reverse: Reverse<C>,
    ) -> Self {
        Self {
            config,
            shape,
            primal,
            forward,
            reverse,
            domain: None,
        }
    }

    /// Check a kernel's domain outside differentiated code before each call.
    #[must_use]
    pub fn with_domain(mut self, domain: Domain<C>) -> Self {
        self.domain = Some(domain);
        self
    }
}

impl<C: Send + Sync> Operator for Kernel<C> {
    fn shape(&self) -> Shape {
        self.shape
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(KernelWorkspace {
            kernel: self,
            value: vec![0.0; self.shape.outputs],
            reverse_seed: vec![0.0; self.shape.outputs],
        })
    }
}

struct KernelWorkspace<'a, C> {
    kernel: &'a Kernel<C>,
    value: Vec<f64>,
    reverse_seed: Vec<f64>,
}

impl<C> KernelWorkspace<'_, C> {
    fn check_input(&self, input: &[f64]) -> Result<()> {
        check_len("input", input.len(), self.kernel.shape.inputs)?;
        check_finite("input", input)?;
        if let Some(domain) = self.kernel.domain {
            domain(&self.kernel.config, input)?;
        }
        Ok(())
    }

    fn check_batch(
        &self,
        input: &[f64],
        count: usize,
        seeds: &[f64],
        output: &[f64],
        reverse: bool,
    ) -> Result<()> {
        let shape = self.kernel.shape;
        let (seed_size, output_size) = if reverse {
            (shape.outputs, shape.inputs)
        } else {
            (shape.inputs, shape.outputs)
        };
        check_len(
            "batch seeds",
            seeds.len(),
            count.checked_mul(seed_size).ok_or(Error::SizeOverflow)?,
        )?;
        check_len(
            "batch output",
            output.len(),
            count.checked_mul(output_size).ok_or(Error::SizeOverflow)?,
        )?;
        self.check_input(input)?;
        check_finite("batch seeds", seeds)
    }
}

impl<C> OperatorWorkspace for KernelWorkspace<'_, C> {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        check_len("output", output.len(), self.kernel.shape.outputs)?;
        self.check_input(input)?;
        output.fill(f64::NAN);
        (self.kernel.primal)(&self.kernel.config, input, output);
        check_finite("kernel output", output)
    }

    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        check_len("seed", seed.len(), self.kernel.shape.inputs)?;
        check_len("output", output.len(), self.kernel.shape.outputs)?;
        self.check_input(input)?;
        check_finite("seed", seed)?;
        self.value.fill(f64::NAN);
        output.fill(0.0);
        (self.kernel.forward)(&self.kernel.config, input, seed, &mut self.value, output);
        check_finite("kernel output", &self.value)?;
        check_finite("kernel JVP", output)
    }

    fn vjp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        check_len("seed", seed.len(), self.kernel.shape.outputs)?;
        check_len("output", output.len(), self.kernel.shape.inputs)?;
        self.check_input(input)?;
        check_finite("seed", seed)?;
        self.value.fill(f64::NAN);
        self.reverse_seed.copy_from_slice(seed);
        output.fill(0.0);
        (self.kernel.reverse)(
            &self.kernel.config,
            input,
            output,
            &mut self.value,
            &mut self.reverse_seed,
        );
        check_finite("kernel output", &self.value)?;
        check_finite("kernel VJP", output)
    }

    fn jvp_batch(
        &mut self,
        input: &[f64],
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        self.check_batch(input, count, seeds, output, false)?;
        let Shape { inputs, outputs } = self.kernel.shape;
        for row in 0..count {
            self.jvp(
                input,
                &seeds[row * inputs..(row + 1) * inputs],
                &mut output[row * outputs..(row + 1) * outputs],
            )?;
        }
        Ok(())
    }

    fn vjp_batch(
        &mut self,
        input: &[f64],
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        self.check_batch(input, count, seeds, output, true)?;
        let Shape { inputs, outputs } = self.kernel.shape;
        for row in 0..count {
            self.vjp(
                input,
                &seeds[row * outputs..(row + 1) * outputs],
                &mut output[row * inputs..(row + 1) * inputs],
            )?;
        }
        Ok(())
    }
}
