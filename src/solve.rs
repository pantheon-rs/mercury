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

    /// Solve once and report normwise backward error and reciprocal condition.
    /// Diagnostics perform `n` additional solves to measure the inverse infinity
    /// norm. Ordinary plan evaluation does not pay this cost or enforce a
    /// conditioning threshold. See the [example](crate#linear-solve-reports).
    ///
    /// # Errors
    /// Rejects invalid inputs, singular factors and nonfinite solves or diagnostics.
    pub fn solve_with_report(&self, input: &[f64]) -> Result<(Vec<f64>, LinearSolveReport)> {
        check_len("solve input", input.len(), self.inputs)?;
        check_finite("solve input", input)?;
        let n = self.dimension;
        let (matrix, rhs) = input.split_at(n * n);
        let factors = factor(MatRef::from_row_major_slice(matrix, n, n))?;
        let mut solution = rhs.to_vec();
        solve_factored(&factors, &mut solution, false)?;
        let matrix_norm = matrix
            .chunks_exact(n)
            .map(|row| row.iter().map(|entry| entry.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let solution_norm = solution
            .iter()
            .map(|entry| entry.abs())
            .fold(0.0_f64, f64::max);
        let rhs_norm = rhs.iter().map(|entry| entry.abs()).fold(0.0_f64, f64::max);
        let mut residual_norm = 0.0_f64;
        for (row, rhs) in matrix.chunks_exact(n).zip(rhs) {
            let residual = row.iter().zip(&solution).map(|(a, x)| a * x).sum::<f64>() - rhs;
            check_finite("linear residual", &[residual])?;
            residual_norm = residual_norm.max(residual.abs());
        }
        let mut column = vec![0.0; n];
        let mut inverse_rows = vec![0.0; n];
        for index in 0..n {
            column.fill(0.0);
            column[index] = 1.0;
            solve_factored(&factors, &mut column, false)?;
            for (sum, entry) in inverse_rows.iter_mut().zip(&column) {
                *sum += entry.abs();
            }
        }
        let inverse_norm = inverse_rows.into_iter().fold(0.0_f64, f64::max);
        let denominator = matrix_norm * solution_norm + rhs_norm;
        check_finite(
            "linear diagnostics",
            &[matrix_norm, inverse_norm, denominator],
        )?;
        let report = LinearSolveReport {
            backward_error: if denominator == 0.0 {
                0.0
            } else {
                residual_norm / denominator
            },
            reciprocal_condition: (1.0 / matrix_norm) / inverse_norm,
        };
        check_finite(
            "linear diagnostics",
            &[report.backward_error, report.reciprocal_condition],
        )?;
        Ok((solution, report))
    }
}

/// Floating-point diagnostics for a dense solve, using infinity norms.
///
/// A small backward error does not imply a small forward error when the
/// reciprocal condition is small. Neither quantity is a certified error bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearSolveReport {
    /// `norm(Ax-b) / (norm(A) norm(x) + norm(b))`; zero for an exactly zero RHS/solution.
    pub backward_error: f64,
    /// `1 / (norm(A) norm(inverse(A)))`, computed using additional factored solves.
    pub reciprocal_condition: f64,
}

impl Operator for DenseSolve {
    fn derivative_order(&self) -> u8 {
        2
    }

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
            tangent: vec![0.0; self.dimension],
            lambda: vec![0.0; self.dimension],
            delta_lambda: vec![0.0; self.dimension],
        })
    }
}

struct DenseWorkspace<'a> {
    operator: &'a DenseSolve,
    factors: Option<PartialPivLu<f64>>,
    solution: Vec<f64>,
    rhs: Vec<f64>,
    tangent: Vec<f64>,
    lambda: Vec<f64>,
    delta_lambda: Vec<f64>,
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

    fn curvature(
        &mut self,
        _input: &[f64],
        weights: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        let n = self.operator.dimension;
        check_len("solve curvature", output.len(), self.operator.inputs)?;
        check_len("solve curvature weights", weights.len(), n)?;
        check_finite("solve curvature weights", weights)?;
        check_len(
            "solve curvature direction",
            direction.len(),
            self.operator.inputs,
        )?;
        check_finite("solve curvature direction", direction)?;
        let factors = self.factors.as_ref().ok_or(Error::InvalidLinearization)?;
        for row in 0..n {
            self.tangent[row] = direction[n * n + row]
                - (0..n)
                    .map(|column| direction[row * n + column] * self.solution[column])
                    .sum::<f64>();
        }
        solve_factored(factors, &mut self.tangent, false)?;
        self.lambda.copy_from_slice(weights);
        solve_factored(factors, &mut self.lambda, true)?;
        for (column, value) in self.delta_lambda.iter_mut().enumerate() {
            *value = -(0..n)
                .map(|row| direction[row * n + column] * self.lambda[row])
                .sum::<f64>();
        }
        solve_factored(factors, &mut self.delta_lambda, true)?;
        for row in 0..n {
            for column in 0..n {
                output[row * n + column] = -self.delta_lambda[row] * self.solution[column]
                    - self.lambda[row] * self.tangent[column];
            }
        }
        output[n * n..].copy_from_slice(&self.delta_lambda);
        check_finite("solve curvature", output)
    }

    fn invalidate(&mut self) {
        self.factors = None;
    }
}

/// Solve `R(z, q) = 0` using Newton iterations, returning `z` as a function of `q`.
///
/// The residual operator takes `[z, q]` and returns one residual per unknown.
/// The initial guess, scaled residual/correction tolerance and iteration limit are
/// inactive configuration. This undamped solver requires a suitable initial
/// guess; it neither chooses among roots nor guarantees global convergence.
/// Derivatives use the implicit function theorem at the returned root, not the
/// finite iteration sequence. Their accuracy depends on convergence and conditioning.
pub struct ImplicitSolve {
    residual: Box<dyn Operator>,
    initial: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
    residual_scales: Vec<f64>,
    state_scales: Vec<f64>,
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
        let entries = n.checked_mul(n).ok_or(Error::SizeOverflow)?;
        std::alloc::Layout::array::<f64>(entries).map_err(|_| Error::SizeOverflow)?;
        Ok(Self {
            residual: Box::new(residual),
            initial,
            tolerance,
            max_iterations,
            residual_scales: vec![1.0; n],
            state_scales: vec![1.0; n],
            shape: Shape {
                inputs: residual_shape.inputs - n,
                outputs: n,
            },
        })
    }
}

impl ImplicitSolve {
    /// Set positive characteristic magnitudes for residuals and unknowns.
    /// Residuals are tested as `abs(R[i]) / residual_scales[i]`; Newton
    /// corrections as `abs(dz[i]) / max(state_scales[i], abs(z[i]))`.
    /// Both infinity norms must meet the constructor's tolerance.
    /// See the [example](crate#scaled-roots-and-reports).
    ///
    /// # Errors
    /// Rejects wrong lengths and nonpositive or nonfinite scales.
    pub fn with_scaling(
        mut self,
        residual_scales: Vec<f64>,
        state_scales: Vec<f64>,
    ) -> Result<Self> {
        for (name, scales) in [
            ("residual scales", &residual_scales),
            ("state scales", &state_scales),
        ] {
            check_len(name, scales.len(), self.shape.outputs)?;
            if scales
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
            {
                return Err(Error::Domain("solve scales must be finite and positive"));
            }
        }
        self.residual_scales = residual_scales;
        self.state_scales = state_scales;
        Ok(self)
    }

    /// Solve once and return the accepted point's convergence diagnostics.
    /// Plan execution uses the same acceptance rule and returns only values.
    /// See the [example](crate#scaled-roots-and-reports).
    ///
    /// # Errors
    /// Rejects invalid inputs and propagates residual, factorization, or convergence errors.
    pub fn solve_with_report(&self, input: &[f64]) -> Result<(Vec<f64>, NewtonReport)> {
        let mut workspace = self.make_workspace();
        let mut output = vec![f64::NAN; self.shape.outputs];
        let report = workspace.solve_root(input, &mut output, false)?;
        Ok((output, report))
    }

    fn make_workspace(&self) -> ImplicitWorkspace<'_> {
        let n = self.shape.outputs;
        let columns = self.residual.shape().inputs;
        ImplicitWorkspace {
            operator: self,
            residual: self.residual.workspace(),
            point: vec![0.0; columns],
            value: vec![0.0; n],
            seed: vec![0.0; columns],
            rhs: vec![0.0; n],
            jacobian: vec![0.0; n * n],
            pullback: vec![0.0; columns],
            tangent: vec![0.0; columns],
            curved: vec![0.0; columns],
            lambda: vec![0.0; n],
            delta_lambda: vec![0.0; n],
            factors: None,
        }
    }
}

/// Convergence at a returned root, or at the last point on iteration exhaustion.
/// Norms use the scales documented by [`ImplicitSolve::with_scaling`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewtonReport {
    /// Number of Newton updates already applied.
    pub iterations: usize,
    /// Infinity norm of the scaled residual at this point.
    pub residual_norm: f64,
    /// Infinity norm of the scaled Newton correction at this point.
    pub correction_norm: f64,
}

impl Operator for ImplicitSolve {
    fn derivative_order(&self) -> u8 {
        self.residual.derivative_order()
    }

    fn shape(&self) -> Shape {
        self.shape
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(self.make_workspace())
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
    pullback: Vec<f64>,
    tangent: Vec<f64>,
    curved: Vec<f64>,
    lambda: Vec<f64>,
    delta_lambda: Vec<f64>,
    factors: Option<PartialPivLu<f64>>,
}

impl ImplicitWorkspace<'_> {
    fn build_jacobian(&mut self) -> Result<PartialPivLu<f64>> {
        let n = self.value.len();
        for col in 0..n {
            self.seed.fill(0.0);
            self.seed[col] = 1.0;
            self.rhs.fill(f64::NAN);
            self.residual.jvp(&self.point, &self.seed, &mut self.rhs)?;
            check_finite("residual Jacobian", &self.rhs)?;
            for (row, derivative) in self.rhs.iter().enumerate() {
                self.jacobian[row * n + col] = *derivative;
            }
        }
        factor(MatRef::from_row_major_slice(&self.jacobian, n, n))
    }

    // Apply only the requested parameter direction at the accepted root.
    fn parameter_tangent(&mut self, direction: &[f64]) -> Result<()> {
        let n = self.value.len();
        self.seed[..n].fill(0.0);
        self.seed[n..].copy_from_slice(direction);
        self.rhs.fill(f64::NAN);
        self.residual.jvp(&self.point, &self.seed, &mut self.rhs)?;
        check_finite("residual tangent", &self.rhs)?;
        for entry in &mut self.rhs {
            *entry = -*entry;
        }
        let factors = self.factors.as_ref().ok_or(Error::InvalidLinearization)?;
        solve_factored(factors, &mut self.rhs, false)
    }

    fn solve_root(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        prepare: bool,
    ) -> Result<NewtonReport> {
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
            let residual_norm = self
                .value
                .iter()
                .zip(&self.operator.residual_scales)
                .fold(0.0_f64, |norm, (value, scale)| {
                    norm.max(value.abs() / scale)
                });
            // An exact residual has a valid value even at a singular root.
            // Preparing implicit derivatives still requires invertible R_z.
            if self.value.iter().all(|value| *value == 0.0) && !prepare {
                output.copy_from_slice(&self.point[..n]);
                self.residual.invalidate();
                return Ok(NewtonReport {
                    iterations: iteration,
                    residual_norm,
                    correction_norm: 0.0,
                });
            }
            let factors = self.build_jacobian()?;
            for (rhs, residual) in self.rhs.iter_mut().zip(&self.value) {
                *rhs = -residual;
            }
            solve_factored(&factors, &mut self.rhs, false)?;
            let correction_norm = self
                .rhs
                .iter()
                .zip(&self.point[..n])
                .zip(&self.operator.state_scales)
                .fold(0.0_f64, |norm, ((step, state), scale)| {
                    norm.max(step.abs() / scale.max(state.abs()))
                });
            let report = NewtonReport {
                iterations: iteration,
                residual_norm,
                correction_norm,
            };
            if residual_norm <= self.operator.tolerance
                && correction_norm <= self.operator.tolerance
            {
                // These factors and the residual workspace belong to this accepted point.
                if prepare {
                    self.factors = Some(factors);
                } else {
                    self.residual.invalidate();
                }
                output.copy_from_slice(&self.point[..n]);
                return Ok(report);
            }
            if iteration == self.operator.max_iterations {
                return Err(Error::NonConvergence { report });
            }
            for (state, step) in self.point[..n].iter_mut().zip(&self.rhs) {
                *state += step;
            }
            check_finite("Newton iterate", &self.point)?;
        }
        unreachable!("the final iteration either accepts the root or reports nonconvergence")
    }
}

impl OperatorWorkspace for ImplicitWorkspace<'_> {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.solve_root(input, output, false).map(|_| ())
    }

    fn linearize(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.solve_root(input, output, true).map(|_| ())
    }

    fn jvp(&mut self, _input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        let n = self.operator.shape.outputs;
        let p = self.operator.shape.inputs;
        check_len("root tangent", seed.len(), p)?;
        check_len("root tangent output", output.len(), n)?;
        check_finite("root tangent", seed)?;
        if self.factors.is_none() {
            return Err(Error::InvalidLinearization);
        }
        self.parameter_tangent(seed)?;
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
        self.pullback.fill(f64::NAN);
        self.residual
            .vjp(&self.point, &self.rhs, &mut self.pullback)?;
        check_finite("residual cotangent", &self.pullback)?;
        for (entry, value) in output.iter_mut().zip(&self.pullback[n..]) {
            *entry = -*value;
        }
        check_finite("root cotangent output", output)
    }

    fn curvature(
        &mut self,
        _input: &[f64],
        weights: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        let n = self.operator.shape.outputs;
        let p = self.operator.shape.inputs;
        check_len("root curvature", output.len(), p)?;
        check_len("root curvature weights", weights.len(), n)?;
        check_finite("root curvature weights", weights)?;
        check_len("root curvature direction", direction.len(), p)?;
        check_finite("root curvature direction", direction)?;
        if self.factors.is_none() {
            return Err(Error::InvalidLinearization);
        }
        self.parameter_tangent(direction)?;
        self.tangent[..n].copy_from_slice(&self.rhs);
        self.tangent[n..].copy_from_slice(direction);
        let factors = self.factors.as_ref().ok_or(Error::InvalidLinearization)?;
        self.lambda.copy_from_slice(weights);
        solve_factored(factors, &mut self.lambda, true)?;
        self.curved.fill(f64::NAN);
        self.residual
            .curvature(&self.point, &self.lambda, &self.tangent, &mut self.curved)?;
        check_finite("residual curvature", &self.curved)?;
        for (entry, value) in self.delta_lambda.iter_mut().zip(&self.curved[..n]) {
            *entry = -*value;
        }
        solve_factored(factors, &mut self.delta_lambda, true)?;
        self.pullback.fill(f64::NAN);
        self.residual
            .vjp(&self.point, &self.delta_lambda, &mut self.pullback)?;
        check_finite("residual cotangent", &self.pullback)?;
        for (column, value) in output.iter_mut().enumerate() {
            *value = -self.curved[n + column] - self.pullback[n + column];
        }
        check_finite("root curvature", output)
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
