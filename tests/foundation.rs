#![feature(autodiff)]
#![allow(
    clippy::float_cmp,
    reason = "Exact binary reference cases; other comparisons use tolerances."
)]
//! Independent checks of sparse structure, typed arguments and second derivatives.

use mercury::advanced::{Operator, OperatorWorkspace, PlanExecution, Shape, Workspace};
use mercury::{DenseSolve, ImplicitSolve, Plan, Source, function};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

#[function(Energy)]
fn energy(velocity: [f64; 2], mass: f64) -> f64 {
    0.5 * mass * (velocity[0] * velocity[0] + velocity[1] * velocity[1])
}

#[function(Transform)]
fn transform(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

#[function(MatrixOutput)]
fn matrix_output(x: f64) -> [[f64; 2]; 2] {
    [[x, x * x], [2.0 * x, 1.0]]
}

#[test]
fn typed_arguments_preserve_names_and_shapes() -> mercury::Result<()> {
    let function = Energy::new();
    let (value, gradient) = function.value_and_gradient([3.0, 4.0], 2.0)?;
    close(value, 25.0);
    assert_eq!(gradient.velocity, [6.0, 8.0]);
    close(gradient.mass, 12.5);
    assert_eq!(function.gradient().eval([3.0, 4.0], 2.0)?, gradient);
    let hessian = function.gradient().jacobian().eval([3.0, 4.0], 2.0)?;
    assert_eq!(hessian, [[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [3.0, 4.0, 0.0]]);
    let transform = Transform::new();
    let matrix = [[2.0, 3.0], [5.0, 7.0]];
    assert_eq!(transform.eval(matrix, [11.0, 13.0])?, [61.0, 146.0]);
    let jacobian = transform.jacobian().eval(matrix, [11.0, 13.0])?;
    assert_eq!(jacobian.vector, matrix);
    assert_eq!(
        jacobian.matrix,
        [[[11.0, 13.0], [0.0, 0.0]], [[0.0, 0.0], [11.0, 13.0]]]
    );
    assert_eq!(MatrixOutput::new().eval(3.0)?, [[3.0, 9.0], [6.0, 1.0]]);
    assert_eq!(
        MatrixOutput::new().jacobian().eval(3.0)?,
        [[1.0], [6.0], [2.0], [0.0]]
    );
    Ok(())
}

struct Diagonal {
    calls: Arc<AtomicUsize>,
}
impl Operator for Diagonal {
    fn shape(&self) -> Shape {
        Shape {
            inputs: 2,
            outputs: 2,
        }
    }
    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(self)
    }
    fn depends_on(&self, output: usize, input: usize) -> bool {
        output == input
    }
}
impl OperatorWorkspace for &Diagonal {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> mercury::Result<()> {
        output.copy_from_slice(&[input[0] * input[0], input[1] * input[1]]);
        Ok(())
    }
    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> mercury::Result<()> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        output.copy_from_slice(&[2.0 * input[0] * seed[0], 2.0 * input[1] * seed[1]]);
        Ok(())
    }
    fn vjp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> mercury::Result<()> {
        output.copy_from_slice(&[2.0 * input[0] * seed[0], 2.0 * input[1] * seed[1]]);
        Ok(())
    }
}

#[test]
fn sparse_coloring_reuses_structure_and_preserves_zero_entries() -> mercury::Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let plan = Plan::from_operator(Diagonal {
        calls: calls.clone(),
    })?;
    let jacobian = plan.jacobian();
    let first = jacobian.eval_sparse(&[0.0, 3.0])?;
    assert_eq!(first.val(), &[0.0, 6.0]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let second = jacobian.eval_sparse(&[2.0, 0.0])?;
    assert_eq!(second.val(), &[4.0, 0.0]);
    assert_eq!(first.symbolic().col_ptr(), second.symbolic().col_ptr());
    assert_eq!(first.symbolic().row_idx(), &[0, 1]);
    assert_eq!(jacobian.sparsity().row_idx(), &[0, 1]);
    let dense = jacobian.eval(&[2.0, 3.0])?;
    assert_eq!(dense, faer::mat![[4.0, 0.0], [0.0, 6.0]]);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    let empty = Plan::builder(2)
        .build([])?
        .jacobian()
        .eval_sparse(&[1.0, 2.0])?;
    assert_eq!((empty.nrows(), empty.ncols(), empty.val().len()), (0, 2, 0));
    Ok(())
}

#[function(Square)]
fn square(x: f64) -> f64 {
    x * x
}
#[function(Add)]
fn add(x: f64, y: f64) -> f64 {
    x + y
}
#[function(Cube)]
fn cube(x: f64) -> f64 {
    x * x * x
}

#[test]
fn composed_hessian_and_gradient_operator_follow_second_order_chain_rule() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let squared = builder.add(Square::new(), [Source::Input(0)]);
    let sum = builder.add(Add::new(), [squared.output(0), Source::Input(1)]);
    let cubed = builder.add(Cube::new(), [sum.output(0)]);
    let plan = builder.build([cubed.output(0)])?;
    let point = [2.0, 3.0];
    let expected = faer::mat![[966.0, 168.0], [168.0, 42.0]];
    assert_eq!(plan.gradient().jacobian().eval(&point)?, expected);
    let gradient = plan.gradient();
    drop(plan); // A derived handle owns a shared copy of its immutable plan.
    let derived = Plan::from_operator(gradient)?;
    assert_eq!(derived.eval(&point)?, [588.0, 147.0]);
    assert_eq!(derived.jacobian().eval(&point)?, expected);
    let energy_gradient = Plan::from_operator(Energy::new().gradient())?;
    assert_eq!(
        energy_gradient.jacobian().eval(&[3.0, 4.0, 2.0])?,
        faer::mat![[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [3.0, 4.0, 0.0]]
    );
    Ok(())
}

#[test]
fn vector_jacobian_is_a_differentiable_operator() -> mercury::Result<()> {
    let derived = Plan::from_operator(MatrixOutput::new().jacobian())?;
    assert_eq!(derived.eval(&[3.0])?, [1.0, 6.0, 2.0, 0.0]);
    assert_eq!(
        derived.jacobian().eval(&[3.0])?,
        faer::mat![[0.0], [2.0], [0.0], [0.0]]
    );
    let plan = Plan::from_operator(MatrixOutput::new())?;
    let derived = Plan::from_operator(plan.jacobian())?;
    assert_eq!(
        derived.jacobian().eval(&[3.0])?,
        faer::mat![[0.0], [2.0], [0.0], [0.0]]
    );
    Ok(())
}

#[function(Residual)]
fn residual(z: f64, q: f64) -> f64 {
    z * z - q
}

#[test]
fn solve_hessians_match_analytic_rules() -> mercury::Result<()> {
    let solve = Plan::from_operator(DenseSolve::new(1)?)?;
    let hessian = solve.gradient().jacobian().eval(&[2.0, 6.0])?;
    assert_eq!(hessian, faer::mat![[1.5, -0.25], [-0.25, 0.0]]);
    // Nested residual plan must retain its accepted-point linearization.
    let residual = Plan::from_operator(Residual::new())?;
    let root = Plan::from_operator(ImplicitSolve::new(residual, vec![1.0], 1e-13, 30)?)?;
    close(
        root.gradient().jacobian().eval(&[4.0])?[(0, 0)],
        -1.0 / 32.0,
    );
    close(
        Plan::from_operator(root.gradient())?
            .jacobian()
            .eval(&[4.0])?[(0, 0)],
        -1.0 / 32.0,
    );
    Ok(())
}

#[test]
fn curvature_failure_invalidates_and_recovery_is_explicit() -> mercury::Result<()> {
    let plan = Plan::from_operator(Diagonal {
        calls: Arc::new(AtomicUsize::new(0)),
    })?;
    let mut workspace = Workspace::new(&plan);
    let mut linearization = plan.linearize(&[2.0, 3.0], &mut workspace)?;
    let mut output = [0.0; 2];
    assert!(matches!(
        linearization.curvature(&[1.0], &[1.0, 0.0], &mut output),
        Err(mercury::Error::Dimension { .. })
    ));
    assert!(linearization.value().is_ok());
    assert!(
        matches!(linearization.curvature(&[1.0, 0.0], &[1.0, 0.0], &mut output), Err(mercury::Error::Operator { source, .. }) if *source == mercury::Error::UnsupportedDerivative)
    );
    assert!(output.iter().all(|value| value.is_nan()));
    assert!(matches!(
        linearization.value(),
        Err(mercury::Error::InvalidLinearization)
    ));
    assert_eq!(
        plan.linearize(&[2.0, 3.0], &mut workspace)?.value()?,
        &[4.0, 9.0]
    );
    Ok(())
}

#[function(CoupledResidual)]
fn coupled_residual(z0: f64, z1: f64, q0: f64, q1: f64) -> [f64; 2] {
    [z0 * z0 - q0, z1 * z1 * z1 - z0 - q1]
}

fn check_curvature_difference(
    plan: &Plan,
    point: &[f64],
    weights: &[f64],
    direction: &[f64],
) -> mercury::Result<()> {
    let mut workspace = Workspace::new(plan);
    let mut linearization = plan.linearize(point, &mut workspace)?;
    let mut actual = vec![0.0; point.len()];
    linearization.curvature(weights, direction, &mut actual)?;
    let step = 1e-5;
    let plus: Vec<_> = point
        .iter()
        .zip(direction)
        .map(|(x, v)| x + step * v)
        .collect();
    let minus: Vec<_> = point
        .iter()
        .zip(direction)
        .map(|(x, v)| x - step * v)
        .collect();
    let plus = plan.jacobian().eval(&plus)?;
    let minus = plan.jacobian().eval(&minus)?;
    for column in 0..point.len() {
        let expected: f64 = weights
            .iter()
            .enumerate()
            .map(|(row, weight)| {
                weight * (plus[(row, column)] - minus[(row, column)]) / (2.0 * step)
            })
            .sum();
        assert!(
            (actual[column] - expected).abs() < 1e-7,
            "{} != {expected}",
            actual[column]
        );
    }
    Ok(())
}

#[test]
fn weighted_curvature_matches_perturb_and_resolve() -> mercury::Result<()> {
    let linear = Plan::from_operator(DenseSolve::new(2)?)?;
    check_curvature_difference(
        &linear,
        &[3.0, 1.0, -0.5, 2.0, 5.0, 5.0],
        &[0.7, -0.4],
        &[0.2, -0.1, 0.3, 0.4, -0.5, 0.6],
    )?;
    let residual = Plan::from_operator(CoupledResidual::new())?;
    let root = Plan::from_operator(ImplicitSolve::new(residual, vec![1.0, 1.0], 1e-13, 30)?)?;
    check_curvature_difference(&root, &[4.0, 6.0], &[0.7, -0.4], &[0.3, -0.2])?;
    Ok(())
}

#[test]
fn fanout_reordered_inputs_and_duplicate_outputs_accumulate_curvature() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let square = builder.add(Square::new(), [Source::Input(1)]);
    let cube = builder.add(Cube::new(), [square.output(0)]);
    let sum = builder.add(Add::new(), [cube.output(0), square.output(0)]);
    let plan = builder.build([
        sum.output(0),
        cube.output(0),
        sum.output(0),
        Source::Input(0),
    ])?;
    check_curvature_difference(&plan, &[2.0, 0.7], &[1.0, -0.3, 0.2, 0.4], &[0.1, 0.3])?;
    let sparse = plan.jacobian().eval_sparse(&[2.0, 0.7])?;
    assert_eq!(sparse.symbolic().col_ptr(), &[0, 1, 4]);
    assert_eq!(sparse.symbolic().row_idx(), &[3, 0, 1, 2]);
    Ok(())
}

#[test]
fn derivative_operator_products_are_adjoint_and_third_order_is_rejected() -> mercury::Result<()> {
    let gradient = Plan::from_operator(Energy::new().gradient())?;
    let point = [3.0, 4.0, 2.0];
    let direction = [0.2, -0.4, 0.7];
    let weights = [0.3, 0.8, -0.5];
    let mut workspace = Workspace::new(&gradient);
    let mut linearization = gradient.linearize(&point, &mut workspace)?;
    let mut tangent = [0.0; 3];
    let mut cotangent = [0.0; 3];
    linearization.jvp(&direction, &mut tangent)?;
    linearization.vjp(&weights, &mut cotangent)?;
    close(
        weights.iter().zip(tangent).map(|(w, v)| w * v).sum(),
        direction.iter().zip(cotangent).map(|(v, w)| v * w).sum(),
    );
    assert!(
        matches!(linearization.curvature(&weights, &direction, &mut tangent), Err(mercury::Error::Operator { source, .. }) if *source == mercury::Error::UnsupportedDerivative)
    );
    Ok(())
}
