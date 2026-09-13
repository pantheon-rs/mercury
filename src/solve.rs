//! Dense linear solves and residual-defined roots with explicit derivatives.

use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::{MatMut, MatRef};

use crate::error::{check_finite, check_len};
use crate::{Error, Operator, OperatorWorkspace, Result, Shape};

/// Solve `A x = b`, with active inputs `[A, b]` and output `x`.
///
/// The square matrix `A` is flattened in row-major order. Derivatives use
/// ordinary and transpose solves with the factors prepared at the input point.
#[derive(Debug)]
pub struct DenseSolve {
    dimension: usize,
    inputs: usize,
}

impl DenseSolve {
    /// Construct a square solve of positive dimension.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    ///
    /// # Errors
    /// Rejects zero dimensions and overflowing input sizes.
    pub fn new(dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::Domain("a solve needs at least one unknown"));
        }
        let inputs = dimension
            .checked_mul(dimension)
            .and_then(|size| size.checked_add(dimension))
            .ok_or(Error::SizeOverflow)?;
        Ok(Self { dimension, inputs })
    }
}

impl Operator for DenseSolve {
    fn shape(&self) -> Shape {
        Shape {
            inputs: self.inputs,
            outputs: self.dimension,
        }
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(DenseWorkspace {
            operator: self,
            factors: None,
            solution: vec![0.0; self.dimension],
            rhs: vec![0.0; self.dimension],
        })
    }
}

struct DenseWorkspace<'a> {
    operator: &'a DenseSolve,
    factors: Option<PartialPivLu<f64>>,
    solution: Vec<f64>,
    rhs: Vec<f64>,
}

impl DenseWorkspace<'_> {
    fn solve_value(&mut self, input: &[f64], output: &mut [f64], prepare: bool) -> Result<()> {
        self.invalidate();
        let n = self.operator.dimension;
        check_len("solve input", input.len(), self.operator.inputs)?;
        check_len("solve output", output.len(), n)?;
        check_finite("solve input", input)?;
        let (matrix, rhs) = input.split_at(n * n);
        let factors = factor(MatRef::from_row_major_slice(matrix, n, n))?;
        self.solution.copy_from_slice(rhs);
        solve_factored(&factors, &mut self.solution, false)?;
        output.copy_from_slice(&self.solution);
        if prepare {
            self.factors = Some(factors);
        }
        Ok(())
    }
}

impl OperatorWorkspace for DenseWorkspace<'_> {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.solve_value(input, output, false)
    }

    fn linearize(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.solve_value(input, output, true)
    }

    fn jvp(&mut self, _input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        let n = self.operator.dimension;
        check_len("solve tangent", seed.len(), self.operator.inputs)?;
        check_len("solve tangent output", output.len(), n)?;
        check_finite("solve tangent", seed)?;
        let factors = self.factors.as_ref().ok_or(Error::InvalidLinearization)?;
        let (matrix_seed, rhs_seed) = seed.split_at(n * n);
        for (row, rhs) in self.rhs.iter_mut().enumerate() {
            let product: f64 = matrix_seed[row * n..(row + 1) * n]
                .iter()
                .zip(&self.solution)
                .map(|(a, x)| a * x)
                .sum();
            *rhs = rhs_seed[row] - product;
        }
        solve_factored(factors, &mut self.rhs, false)?;
        output.copy_from_slice(&self.rhs);
        Ok(())
    }

    fn vjp(&mut self, _input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        let n = self.operator.dimension;
        check_len("solve cotangent", seed.len(), n)?;
        check_len("solve cotangent output", output.len(), self.operator.inputs)?;
        check_finite("solve cotangent", seed)?;
        let factors = self.factors.as_ref().ok_or(Error::InvalidLinearization)?;
        self.rhs.copy_from_slice(seed);
        solve_factored(factors, &mut self.rhs, true)?;
        let (matrix_bar, rhs_bar) = output.split_at_mut(n * n);
        for (row, lambda) in self.rhs.iter().enumerate() {
            for (col, x) in self.solution.iter().enumerate() {
                matrix_bar[row * n + col] = -lambda * x;
            }
        }
        rhs_bar.copy_from_slice(&self.rhs);
        check_finite("solve cotangent output", output)
    }

    fn invalidate(&mut self) {
        self.factors = None;
    }
}

/// Solve `R(z, q) = 0` using Newton iterations, returning `z` as a function of `q`.
///
/// The residual operator takes `[z, q]` and returns one residual per unknown.
/// The initial guess, absolute residual tolerance and iteration limit are
/// inactive configuration. This undamped solver requires a suitable initial
/// guess; it neither chooses among roots nor guarantees global convergence.
/// Derivatives use the implicit function theorem at the returned root, not the
/// finite iteration sequence. Their accuracy depends on convergence and conditioning.
pub struct ImplicitSolve {
    residual: Box<dyn Operator>,
    initial: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
    shape: Shape,
}

impl ImplicitSolve {
    /// Construct a residual solve with an explicit initial guess.
    ///
    /// See the [example](crate#implicit-roots).
    ///
    /// # Errors
    /// Rejects inconsistent residual dimensions, non-finite initial values,
    /// non-positive or non-finite tolerances, zero iteration limits, and
    /// overflowing Jacobian sizes.
    pub fn new(
        residual: impl Operator + 'static,
        initial: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self> {
        let residual_shape = residual.shape();
        let n = initial.len();
        if n == 0 || residual_shape.outputs != n || residual_shape.inputs < n {
            return Err(Error::Domain(
                "residual dimensions must describe R([z, q]) with one output per unknown",
            ));
        }
        check_finite("initial guess", &initial)?;
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(Error::Domain(
                "Newton needs a finite positive tolerance and a positive iteration limit",
            ));
        }
        n.checked_mul(residual_shape.inputs)
            .ok_or(Error::SizeOverflow)?;
        Ok(Self {
            residual: Box::new(residual),
            initial,
            tolerance,
            max_iterations,
            shape: Shape {
                inputs: residual_shape.inputs - n,
                outputs: n,
            },
        })
    }
}

impl Operator for ImplicitSolve {
    fn shape(&self) -> Shape {
        self.shape
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        let n = self.shape.outputs;
        let columns = self.residual.shape().inputs;
        Box::new(ImplicitWorkspace {
            operator: self,
            residual: self.residual.workspace(),
            point: vec![0.0; columns],
            value: vec![0.0; n],
            seed: vec![0.0; columns],
            rhs: vec![0.0; n],
            jacobian: vec![0.0; n * columns],
            factors: None,
        })
    }
}

struct ImplicitWorkspace<'a> {
    operator: &'a ImplicitSolve,
    residual: Box<dyn OperatorWorkspace + 'a>,
    point: Vec<f64>,
    value: Vec<f64>,
    seed: Vec<f64>,
    rhs: Vec<f64>,
    jacobian: Vec<f64>,
    factors: Option<PartialPivLu<f64>>,
}

impl ImplicitWorkspace<'_> {
    fn build_jacobian(&mut self, include_parameters: bool) -> Result<()> {
        let columns = self.point.len();
        let count = if include_parameters {
            columns
        } else {
            self.value.len()
        };
        for col in 0..count {
            self.seed.fill(0.0);
            self.seed[col] = 1.0;
            self.rhs.fill(f64::NAN);
            self.residual.jvp(&self.point, &self.seed, &mut self.rhs)?;
            check_finite("residual Jacobian", &self.rhs)?;
            for (row, derivative) in self.rhs.iter().enumerate() {
                self.jacobian[row * columns + col] = *derivative;
            }
        }
        Ok(())
    }

    fn state_factors(&self) -> Result<PartialPivLu<f64>> {
        let n = self.value.len();
        let jacobian = MatRef::from_row_major_slice(&self.jacobian, n, self.point.len());
        factor(jacobian.subcols(0, n))
    }

    fn solve_root(&mut self, input: &[f64], output: &mut [f64], prepare: bool) -> Result<()> {
        self.invalidate();
        let n = self.operator.shape.outputs;
        check_len("root input", input.len(), self.operator.shape.inputs)?;
        check_len("root output", output.len(), n)?;
        check_finite("root input", input)?;
        self.point[..n].copy_from_slice(&self.operator.initial);
        self.point[n..].copy_from_slice(input);

        for iteration in 0..=self.operator.max_iterations {
            self.value.fill(f64::NAN);
            self.residual.linearize(&self.point, &mut self.value)?;
            check_finite("residual", &self.value)?;
            let norm = self
                .value
                .iter()
                .fold(0.0_f64, |norm, value| norm.max(value.abs()));
            if norm <= self.operator.tolerance {
                if prepare {
                    // The final accepted point may differ from the point of the
                    // last Newton factorization. Its derivatives need fresh factors.
                    self.build_jacobian(true)?;
                    self.factors = Some(self.state_factors()?);
                }
                output.copy_from_slice(&self.point[..n]);
                self.residual.invalidate();
                return Ok(());
            }
            if iteration == self.operator.max_iterations {
                return Err(Error::NonConvergence);
            }
            self.build_jacobian(false)?;
            let factors = self.state_factors()?;
            for (rhs, residual) in self.rhs.iter_mut().zip(&self.value) {
                *rhs = -residual;
            }
            solve_factored(&factors, &mut self.rhs, false)?;
            for (state, step) in self.point[..n].iter_mut().zip(&self.rhs) {
                *state += step;
            }
            check_finite("Newton iterate", &self.point)?;
        }
        Err(Error::NonConvergence)
    }
}

impl OperatorWorkspace for ImplicitWorkspace<'_> {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.solve_root(input, output, false)
    }

    fn linearize(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.solve_root(input, output, true)
    }

    fn jvp(&mut self, _input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        let n = self.operator.shape.outputs;
        let p = self.operator.shape.inputs;
        check_len("root tangent", seed.len(), p)?;
        check_len("root tangent output", output.len(), n)?;
        check_finite("root tangent", seed)?;
        let factors = self.factors.as_ref().ok_or(Error::InvalidLinearization)?;
        for (row, rhs) in self.rhs.iter_mut().enumerate() {
            let start = row * (n + p) + n;
            *rhs = -self.jacobian[start..start + p]
                .iter()
                .zip(seed)
                .map(|(a, v)| a * v)
                .sum::<f64>();
        }
        solve_factored(factors, &mut self.rhs, false)?;
        output.copy_from_slice(&self.rhs);
        Ok(())
    }

    fn vjp(&mut self, _input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        let n = self.operator.shape.outputs;
        let p = self.operator.shape.inputs;
        check_len("root cotangent", seed.len(), n)?;
        check_len("root cotangent output", output.len(), p)?;
        check_finite("root cotangent", seed)?;
        let factors = self.factors.as_ref().ok_or(Error::InvalidLinearization)?;
        self.rhs.copy_from_slice(seed);
        solve_factored(factors, &mut self.rhs, true)?;
        for (col, derivative) in output.iter_mut().enumerate() {
            *derivative = -self
                .rhs
                .iter()
                .enumerate()
                .map(|(row, lambda)| self.jacobian[row * (n + p) + n + col] * lambda)
                .sum::<f64>();
        }
        check_finite("root cotangent output", output)
    }

    fn invalidate(&mut self) {
        self.factors = None;
        self.residual.invalidate();
    }
}

fn factor(matrix: MatRef<'_, f64>) -> Result<PartialPivLu<f64>> {
    let factors = PartialPivLu::new(matrix);
    let n = matrix.nrows();
    for row in 0..n {
        if factors.U()[(row, row)] == 0.0 {
            return Err(Error::Singular);
        }
        for col in 0..n {
            if !factors.L()[(row, col)].is_finite() || !factors.U()[(row, col)].is_finite() {
                return Err(Error::NonFinite("LU factors"));
            }
        }
    }
    Ok(factors)
}

fn solve_factored(factors: &PartialPivLu<f64>, rhs: &mut [f64], transpose: bool) -> Result<()> {
    let n = rhs.len();
    let view = MatMut::from_column_major_slice_mut(rhs, n, 1);
    if transpose {
        factors.solve_transpose_in_place(view);
    } else {
        factors.solve_in_place(view);
    }
    check_finite("solve result", rhs)
}
