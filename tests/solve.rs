//! Independent numerical and lifecycle checks for explicit solve derivatives.

use mercury::advanced::{Operator, OperatorWorkspace, PlanExecution, Shape};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mercury::{DenseSolve, Error, ImplicitSolve, Plan, Result, Source};

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn check_direction(operator: &impl Operator, point: &[f64], direction: &[f64], tolerance: f64) {
    let mut workspace = operator.workspace();
    let mut value = vec![0.0; operator.shape().outputs];
    let mut tangent = value.clone();
    workspace.linearize(point, &mut value).unwrap();
    workspace.jvp(point, direction, &mut tangent).unwrap();

    let h = 1e-5;
    let plus: Vec<_> = point
        .iter()
        .zip(direction)
        .map(|(x, v)| x + h * v)
        .collect();
    let minus: Vec<_> = point
        .iter()
        .zip(direction)
        .map(|(x, v)| x - h * v)
        .collect();
    let mut high = value.clone();
    let mut low = value;
    workspace.evaluate(&plus, &mut high).unwrap();
    workspace.evaluate(&minus, &mut low).unwrap();
    let difference: Vec<_> = high
        .iter()
        .zip(&low)
        .map(|(a, b)| (a - b) / (2.0 * h))
        .collect();
    close(&tangent, &difference, tolerance);
}

#[test]
fn pivoted_linear_solve_and_both_derivative_rules() {
    let operator = DenseSolve::new(2).unwrap();
    let point = [0.0, 2.0, 1.0, 3.0, -2.0, -1.0];
    let tangent = [1.0, 0.5, -0.25, 2.0, 0.3, -0.2];
    let cotangent = [0.7, -1.1];
    let mut workspace = operator.workspace();
    let mut value = [f64::NAN; 2];
    let mut jvp = [f64::NAN; 2];
    let mut vjp = [f64::NAN; 6];

    workspace.linearize(&point, &mut value).unwrap();
    workspace.jvp(&point, &tangent, &mut jvp).unwrap();
    workspace.vjp(&point, &cotangent, &mut vjp).unwrap();
    close(&value, &[2.0, -1.0], 1e-14);
    close(&jvp, &[4.1, -0.6], 1e-14);
    close(&vjp, &[3.2, -1.6, -1.4, 0.7, -1.6, 0.7], 1e-14);
    assert!((dot(&jvp, &cotangent) - dot(&tangent, &vjp)).abs() < 1e-13);

    workspace.jvp(&point, &tangent, &mut jvp).unwrap();
    close(&jvp, &[4.1, -0.6], 1e-14);
    close(&value, &[2.0, -1.0], 1e-14);
    check_direction(&operator, &point, &tangent, 1e-8);
}

#[test]
fn linear_batches_preserve_seeds_and_workspace_can_prepare_a_new_point() {
    let operator = DenseSolve::new(2).unwrap();
    let point = [0.0, 2.0, 1.0, 3.0, -2.0, -1.0];
    let seeds = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let saved = seeds;
    let mut workspace = operator.workspace();
    let mut value = [0.0; 2];
    let mut results = [f64::NAN; 4];
    workspace.linearize(&point, &mut value).unwrap();
    workspace
        .jvp_batch(&point, 2, &seeds, &mut results)
        .unwrap();
    close(&results, &[-1.5, 0.5, 1.0, 0.0], 1e-14);
    assert_eq!(seeds.map(f64::to_bits), saved.map(f64::to_bits));
    workspace.jvp_batch(&point, 0, &[], &mut []).unwrap();

    let point = [2.0, 0.0, 0.0, 4.0, 6.0, 8.0];
    workspace.linearize(&point, &mut value).unwrap();
    workspace
        .jvp_batch(&point, 2, &seeds, &mut results)
        .unwrap();
    close(&value, &[3.0, 2.0], 1e-14);
    close(&results, &[0.5, 0.0, 0.0, 0.25], 1e-14);
    workspace.vjp_batch(&point, 0, &[], &mut []).unwrap();
}

#[test]
fn singular_or_nonfinite_linear_input_cannot_leave_usable_factors() {
    let operator = DenseSolve::new(2).unwrap();
    let mut workspace = operator.workspace();
    let valid = [1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
    let singular = [1.0, 2.0, 2.0, 4.0, 1.0, 2.0];
    let mut value = [0.0; 2];
    workspace.linearize(&valid, &mut value).unwrap();
    assert_eq!(
        workspace.linearize(&singular, &mut value),
        Err(Error::Singular)
    );
    assert_eq!(
        workspace.jvp(&singular, &[0.0; 6], &mut value),
        Err(Error::InvalidLinearization)
    );

    let mut invalid = valid;
    invalid[0] = f64::INFINITY;
    assert!(matches!(
        workspace.linearize(&invalid, &mut value),
        Err(Error::NonFinite(_))
    ));
    workspace.linearize(&valid, &mut value).unwrap();
    workspace.invalidate();
    assert_eq!(
        workspace.jvp(&valid, &[0.0; 6], &mut value),
        Err(Error::InvalidLinearization)
    );
}

#[test]
fn derivative_overflow_invalidates_the_plan_and_a_new_point_recovers() {
    let mut builder = Plan::builder(2);
    let solve = builder.add(
        DenseSolve::new(1).unwrap(),
        [Source::Input(0), Source::Input(1)],
    );
    let plan = builder.build([solve.output(0)]).unwrap();
    let mut workspace = mercury::advanced::Workspace::new(&plan);
    let point = [1e-308, 1e-308];
    {
        let mut linearization = plan.linearize(&point, &mut workspace).unwrap();
        let mut derivatives = [0.0; 2];
        assert!(
            linearization
                .jvp_batch(2, &[0.0, 0.0, 0.0, f64::MAX], &mut derivatives)
                .is_err()
        );
        assert!(derivatives.iter().all(|value| value.is_nan()));
        assert_eq!(linearization.value(), Err(Error::InvalidLinearization));
        assert_eq!(
            linearization.vjp(&[1.0], &mut [0.0; 2]),
            Err(Error::InvalidLinearization)
        );
    }
    let point = [2.0, 6.0];
    let mut linearization = plan.linearize(&point, &mut workspace).unwrap();
    close(linearization.value().unwrap(), &[3.0], 1e-14);
    let mut cotangent = [f64::NAN; 2];
    linearization.vjp(&[2.0], &mut cotangent).unwrap();
    close(&cotangent, &[-3.0, 1.0], 1e-14);
}

#[derive(Clone)]
struct CoupledResidual {
    evaluations: Arc<AtomicUsize>,
}

impl Operator for CoupledResidual {
    fn shape(&self) -> Shape {
        Shape {
            inputs: 4,
            outputs: 2,
        }
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(self.clone())
    }
}

impl OperatorWorkspace for CoupledResidual {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        output[0] = input[0] * input[0] + input[1] - input[2];
        output[1] = 3.0 * input[0] + 2.0 * input[1] * input[1] - input[3];
        Ok(())
    }

    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        output[0] = 2.0 * input[0] * seed[0] + seed[1] - seed[2];
        output[1] = 3.0 * seed[0] + 4.0 * input[1] * seed[1] - seed[3];
        Ok(())
    }

    fn vjp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        output[0] = 2.0 * input[0] * seed[0] + 3.0 * seed[1];
        output[1] = seed[0] + 4.0 * input[1] * seed[1];
        output[2] = -seed[0];
        output[3] = -seed[1];
        Ok(())
    }
}

#[test]
fn coupled_root_derivatives_match_analytic_solution_sensitivity_and_resolving() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let residual = CoupledResidual {
        evaluations: evaluations.clone(),
    };
    let operator = ImplicitSolve::new(residual, vec![1.5, 0.8], 1e-13, 20).unwrap();
    let point = [5.0, 8.0];
    let tangent = [0.7, -0.2];
    let cotangent = [0.4, -0.6];
    let mut workspace = operator.workspace();
    let mut value = [f64::NAN; 2];
    let mut jvp = [f64::NAN; 2];
    let mut vjp = [f64::NAN; 2];
    workspace.linearize(&point, &mut value).unwrap();
    let prepared_evaluations = evaluations.load(Ordering::Relaxed);
    workspace.jvp(&point, &tangent, &mut jvp).unwrap();
    workspace.vjp(&point, &cotangent, &mut vjp).unwrap();
    close(&value, &[2.0, 1.0], 1e-12);
    close(&jvp, &[3.0 / 13.0, -2.9 / 13.0], 1e-12);
    close(&vjp, &[3.4 / 13.0, -2.8 / 13.0], 1e-12);
    assert!((dot(&jvp, &cotangent) - dot(&tangent, &vjp)).abs() < 1e-12);
    workspace.jvp(&point, &tangent, &mut jvp).unwrap();
    close(&jvp, &[3.0 / 13.0, -2.9 / 13.0], 1e-12);
    assert_eq!(evaluations.load(Ordering::Relaxed), prepared_evaluations);
    check_direction(&operator, &point, &tangent, 1e-8);
}

struct SquareRootResidual;

impl Operator for SquareRootResidual {
    fn shape(&self) -> Shape {
        Shape {
            inputs: 2,
            outputs: 1,
        }
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(Self)
    }
}

impl OperatorWorkspace for SquareRootResidual {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        output[0] = input[0] * input[0] - input[1];
        Ok(())
    }

    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        output[0] = 2.0 * input[0] * seed[0] - seed[1];
        Ok(())
    }

    fn vjp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        output[0] = 2.0 * input[0] * seed[0];
        output[1] = -seed[0];
        Ok(())
    }
}

#[test]
fn runtime_composed_residual_can_be_solved_inside_another_plan() {
    // R(z,q) = (z²-q)²-q has the selected root sqrt(q + sqrt(q)).
    let mut builder = Plan::builder(2);
    let inner = builder.add(SquareRootResidual, [Source::Input(0), Source::Input(1)]);
    let outer = builder.add(SquareRootResidual, [inner.output(0), Source::Input(1)]);
    let residual = builder.build([outer.output(0)]).unwrap();
    let solve = ImplicitSolve::new(residual, vec![2.4], 1e-13, 20).unwrap();
    check_direction(&solve, &[4.0], &[0.7], 1e-8);

    let mut builder = Plan::builder(1);
    let root = builder.add(solve, [Source::Input(0)]);
    let plan = builder.build([root.output(0)]).unwrap();
    let mut workspace = mercury::advanced::Workspace::new(&plan);
    let point = [4.0];
    let mut linearization = plan.linearize(&point, &mut workspace).unwrap();
    let mut derivative = [0.0];
    close(linearization.value().unwrap(), &[6.0_f64.sqrt()], 1e-12);
    linearization.jvp(&[1.0], &mut derivative).unwrap();
    close(&derivative, &[0.625 / 6.0_f64.sqrt()], 1e-12);
    linearization.vjp(&[2.0], &mut derivative).unwrap();
    close(&derivative, &[1.25 / 6.0_f64.sqrt()], 1e-12);
}

#[test]
fn root_linearization_rebuilds_factors_at_the_returned_point() {
    // Accept the second Newton update. The preceding Jacobian is 3, whereas
    // the Jacobian at the returned point is 17/6, large enough to distinguish.
    let operator = ImplicitSolve::new(SquareRootResidual, vec![1.0], 0.01, 2).unwrap();
    let mut workspace = operator.workspace();
    let mut value = [0.0];
    let mut derivative = [0.0];
    workspace.linearize(&[2.0], &mut value).unwrap();
    workspace.jvp(&[2.0], &[1.0], &mut derivative).unwrap();
    close(&value, &[17.0 / 12.0], 1e-14);
    close(&derivative, &[1.0 / (2.0 * value[0])], 1e-14);
    assert!((derivative[0] - 1.0 / 3.0).abs() > 0.01);
}

#[test]
fn roots_report_nonconvergence_and_singular_jacobians() {
    let operator = ImplicitSolve::new(SquareRootResidual, vec![1.0], 1e-13, 1).unwrap();
    let mut workspace = operator.workspace();
    let mut output = [0.0];
    let Err(Error::NonConvergence { report }) = workspace.linearize(&[2.0], &mut output) else {
        panic!("expected iteration exhaustion");
    };
    assert_eq!(report.iterations, 1);
    assert!(report.residual_norm > 0.0);
    assert!(report.correction_norm > 0.0);
    assert_eq!(
        workspace.jvp(&[2.0], &[1.0], &mut output),
        Err(Error::InvalidLinearization)
    );

    let operator = ImplicitSolve::new(SquareRootResidual, vec![0.0], 1e-13, 5).unwrap();
    let mut workspace = operator.workspace();
    assert_eq!(
        workspace.linearize(&[1.0], &mut output),
        Err(Error::Singular)
    );
    // A root can have a valid value while lacking an implicit derivative.
    workspace.evaluate(&[0.0], &mut output).unwrap();
    assert_eq!(
        workspace.linearize(&[0.0], &mut output),
        Err(Error::Singular)
    );
}

#[test]
fn solve_constructors_reject_invalid_shapes_and_settings() {
    assert!(matches!(DenseSolve::new(0), Err(Error::Domain(_))));
    assert!(matches!(
        DenseSolve::new(usize::MAX),
        Err(Error::SizeOverflow)
    ));
    assert!(matches!(
        ImplicitSolve::new(SquareRootResidual, vec![], 1e-12, 10),
        Err(Error::Domain(_))
    ));
    assert!(matches!(
        ImplicitSolve::new(SquareRootResidual, vec![1.0], 0.0, 10),
        Err(Error::Domain(_))
    ));
    assert!(matches!(
        ImplicitSolve::new(SquareRootResidual, vec![1.0], 1e-12, 0),
        Err(Error::Domain(_))
    ));
    assert!(matches!(
        ImplicitSolve::new(SquareRootResidual, vec![f64::NAN], 1e-12, 10),
        Err(Error::NonFinite(_))
    ));
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "The kernel ABI borrows inactive configuration."
)]
fn scaled_root(scale: &f64, input: &[f64], output: &mut [f64]) {
    output[0] = scale * (input[0] * input[0] - input[1]);
}
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "The kernel ABI borrows inactive configuration."
)]
fn scaled_forward(scale: &f64, input: &[f64], seed: &[f64], value: &mut [f64], output: &mut [f64]) {
    scaled_root(scale, input, value);
    output[0] = scale * (2.0 * input[0] * seed[0] - seed[1]);
}
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "The kernel ABI borrows inactive configuration."
)]
fn scaled_reverse(
    scale: &f64,
    input: &[f64],
    output: &mut [f64],
    value: &mut [f64],
    seed: &mut [f64],
) {
    scaled_root(scale, input, value);
    output[0] += scale * 2.0 * input[0] * seed[0];
    output[1] -= scale * seed[0];
}
fn scaled_operator(scale: f64) -> mercury::advanced::Kernel<f64> {
    mercury::advanced::Kernel::new(
        scale,
        Shape {
            inputs: 2,
            outputs: 1,
        },
        scaled_root,
        scaled_forward,
        scaled_reverse,
    )
}

#[test]
fn root_acceptance_checks_corrections_and_explicit_scales() -> Result<()> {
    for scale in [1.0, 1e-12] {
        for explicit in [false, true] {
            let mut solve = ImplicitSolve::new(scaled_operator(scale), vec![1.0], 1e-8, 20)?;
            if explicit {
                solve = solve.with_scaling(vec![scale], vec![2.0])?;
            }
            let (root, report) = solve.solve_with_report(&[4.0])?;
            close(&root, &[2.0], 1e-8);
            assert!(report.iterations > 0);
            assert!(report.residual_norm <= 1e-8);
            assert!(report.correction_norm <= 1e-8);
            let plan = Plan::from_operator(solve)?;
            close(&plan.gradient().eval(&[4.0])?, &[0.25], 1e-8);
        }
    }
    for scales in [
        vec![],
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        assert!(
            ImplicitSolve::new(scaled_operator(1.0), vec![1.0], 1e-8, 20)?
                .with_scaling(scales.clone(), vec![1.0])
                .is_err()
        );
        assert!(
            ImplicitSolve::new(scaled_operator(1.0), vec![1.0], 1e-8, 20)?
                .with_scaling(vec![1.0], scales)
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn linear_reports_distinguish_conditioning_from_backward_error() -> Result<()> {
    let solve = DenseSolve::new(2)?;
    for diagonal in [1.0, 1e-12] {
        let (value, report) = solve.solve_with_report(&[1.0, 0.0, 0.0, diagonal, 1.0, diagonal])?;
        close(&value, &[1.0, 1.0], 1e-14);
        assert!(report.backward_error <= 1e-14);
        close(&[report.reciprocal_condition], &[diagonal], 1e-14);
    }
    let (_, report) = solve.solve_with_report(&[2.0, 1.0, 0.0, 3.0, 0.0, 0.0])?;
    close(&[report.backward_error], &[0.0], 0.0);
    // ||A||_inf = 3, ||A^-1||_inf = 2/3.
    close(&[report.reciprocal_condition], &[0.5], 1e-14);
    assert!(matches!(
        solve.solve_with_report(&[0.0; 6]),
        Err(Error::Singular)
    ));
    Ok(())
}

#[derive(Default)]
struct ResidualCalls {
    forward: AtomicUsize,
    reverse: AtomicUsize,
    fail: std::sync::atomic::AtomicBool,
}
fn wide_value(_calls: &Arc<ResidualCalls>, point: &[f64], value: &mut [f64]) {
    value[0] = point[0] - point[1..].iter().sum::<f64>();
}
fn wide_forward(
    calls: &Arc<ResidualCalls>,
    point: &[f64],
    seed: &[f64],
    value: &mut [f64],
    output: &mut [f64],
) {
    calls.forward.fetch_add(1, Ordering::Relaxed);
    wide_value(calls, point, value);
    output[0] = seed[0] - seed[1..].iter().sum::<f64>();
}
fn wide_reverse(
    calls: &Arc<ResidualCalls>,
    point: &[f64],
    output: &mut [f64],
    value: &mut [f64],
    seed: &mut [f64],
) {
    calls.reverse.fetch_add(1, Ordering::Relaxed);
    wide_value(calls, point, value);
    output[0] += seed[0];
    for entry in &mut output[1..] {
        *entry -= seed[0];
    }
}
fn wide_domain(calls: &Arc<ResidualCalls>, _point: &[f64]) -> Result<()> {
    if calls.fail.load(Ordering::Relaxed) {
        Err(Error::Domain("residual failure"))
    } else {
        Ok(())
    }
}

#[test]
fn implicit_parameter_products_are_directional_and_failures_invalidate() -> Result<()> {
    let calls = Arc::new(ResidualCalls::default());
    let residual = mercury::advanced::Kernel::new(
        calls.clone(),
        Shape {
            inputs: 129,
            outputs: 1,
        },
        wide_value,
        wide_forward,
        wide_reverse,
    )
    .with_domain(wide_domain);
    let plan = Plan::from_operator(ImplicitSolve::new(residual, vec![0.0], 1e-10, 10)?)?;
    let mut workspace = mercury::advanced::Workspace::new(&plan);
    let point = [0.0; 128];
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    // Only R_z is assembled, even with 128 parameters.
    assert_eq!(calls.forward.load(Ordering::Relaxed), 1);
    assert_eq!(calls.reverse.load(Ordering::Relaxed), 0);
    let mut tangent = [0.0];
    linearization.jvp(&[1.0; 128], &mut tangent)?;
    close(&tangent, &[128.0], 0.0);
    assert_eq!(calls.forward.load(Ordering::Relaxed), 2);
    let mut gradient = [0.0; 128];
    linearization.vjp(&[2.0], &mut gradient)?;
    close(&gradient, &[2.0; 128], 0.0);
    assert_eq!(calls.reverse.load(Ordering::Relaxed), 1);
    calls.fail.store(true, Ordering::Relaxed);
    assert!(linearization.vjp(&[1.0], &mut gradient).is_err());
    assert!(gradient.iter().all(|entry| entry.is_nan()));
    assert_eq!(linearization.value(), Err(Error::InvalidLinearization));
    calls.fail.store(false, Ordering::Relaxed);
    plan.linearize(&point, &mut workspace)?
        .vjp(&[1.0], &mut gradient)?;
    close(&gradient, &[1.0; 128], 0.0);
    Ok(())
}
