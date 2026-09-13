#![feature(autodiff)]

//! Typed calls agree with analytic derivatives and the runtime graph boundary.

use mercury::advanced::PlanExecution;
use mercury::{Error, Plan, Source, function};

#[function(Rosenbrock)]
fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0 - x;
    let b = y - x * x;
    a * a + 100.0 * b * b
}

#[function(Polynomial)]
fn polynomial(x: f64, y: f64) -> [f64; 3] {
    [x * x + y, x * y, x - 3.0 * y * y]
}

#[function(Root)]
fn root(x: f64) -> f64 {
    x.sqrt()
}

#[function(RootVector)]
fn root_vector(x: f64) -> [f64; 1] {
    [x.sqrt()]
}

#[function(Overflow)]
fn overflow(x: f64) -> f64 {
    (x * f64::MAX) * 2.0
}

#[function(Logarithm)]
fn logarithm(x: f64) -> [f64; 1] {
    [x.ln()]
}

mod public_api {
    #[mercury::function(Exported)]
    pub fn r#type(value_and_gradient: f64, jacobian: f64, input: f64) -> f64 {
        value_and_gradient * jacobian + input
    }

    #[mercury::function(Vector)]
    pub fn vector(jacobian: f64) -> [f64; 1] {
        [jacobian * jacobian]
    }
}

#[test]
fn public_types_and_parameter_names_work_across_modules() {
    let function = public_api::Exported::new();
    close(
        &function.gradient().eval(2.0, 3.0, 4.0).unwrap(),
        &[3.0, 2.0, 1.0],
        1e-12,
    );
    close(&[function.eval(2.0, 3.0, 4.0).unwrap()], &[10.0], 1e-12);
    close(
        &public_api::Vector::new().jacobian().eval(3.0).unwrap()[0],
        &[6.0],
        1e-12,
    );
}

#[function(Disabled)]
#[cfg(any())]
fn disabled(x: Unavailable) -> f64 {
    x
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn scalar_value_gradient_and_combined_call_agree_at_multiple_points() {
    let function = Rosenbrock::new();
    let gradient = function.gradient();
    for [x, y] in [[-1.2, 1.0], [1.0, 1.0], [0.5, -0.3]] {
        let expected = [
            2.0 * (x - 1.0) - 400.0 * x * (y - x * x),
            200.0 * (y - x * x),
        ];
        close(&gradient.eval(x, y).unwrap(), &expected, 1e-10);
        let (value, combined_gradient) = function.value_and_gradient(x, y).unwrap();
        close(&[value], &[function.eval(x, y).unwrap()], 1e-12);
        close(&combined_gradient, &expected, 1e-10);
        close(&[value], &[rosenbrock(x, y)], 1e-12);
    }
    close(&[function.eval(-1.2, 1.0).unwrap()], &[24.2], 1e-12);
}

#[test]
fn rectangular_jacobian_has_output_rows_and_argument_columns() {
    let function = Polynomial::new();
    close(&function.eval(2.0, 3.0).unwrap(), &[7.0, 6.0, -25.0], 1e-12);
    let jacobian = function.jacobian();
    for [x, y] in [[2.0, 3.0], [-0.7, 1.2]] {
        let actual = jacobian.eval(x, y).unwrap();
        let analytic = [[2.0 * x, 1.0], [y, x], [1.0, -6.0 * y]];
        for (actual, expected) in actual.iter().zip(analytic) {
            close(actual, &expected, 1e-12);
        }
        let h = 1e-6;
        for column in 0..2 {
            let mut plus = [x, y];
            let mut minus = [x, y];
            plus[column] += h;
            minus[column] -= h;
            let plus = function.eval(plus[0], plus[1]).unwrap();
            let minus = function.eval(minus[0], minus[1]).unwrap();
            for row in 0..3 {
                close(
                    &[actual[row][column]],
                    &[(plus[row] - minus[row]) / (2.0 * h)],
                    1e-8,
                );
            }
        }
    }
}

#[test]
fn numerical_errors_do_not_leave_state_in_derivative_handles() {
    let function = Root::new();
    let gradient = function.gradient();
    assert_eq!(function.eval(f64::NAN), Err(Error::NonFinite("input")));
    assert_eq!(gradient.eval(f64::INFINITY), Err(Error::NonFinite("input")));
    assert_eq!(
        function.eval(-1.0),
        Err(Error::NonFinite("function output"))
    );
    assert_eq!(
        gradient.eval(-1.0),
        Err(Error::NonFinite("function output"))
    );
    close(&[function.eval(0.0).unwrap()], &[0.0], 1e-12);
    assert_eq!(
        Overflow::new().gradient().eval(0.0),
        Err(Error::NonFinite("gradient"))
    );
    close(&gradient.eval(4.0).unwrap(), &[0.25], 1e-12);

    let vector = RootVector::new();
    assert_eq!(vector.eval(f64::INFINITY), Err(Error::NonFinite("input")));
    let jacobian = vector.jacobian();
    assert_eq!(jacobian.eval(f64::NAN), Err(Error::NonFinite("input")));
    assert_eq!(
        jacobian.eval(-1.0),
        Err(Error::NonFinite("function output"))
    );
    close(&jacobian.eval(4.0).unwrap()[0], &[0.25], 1e-12);
    assert_eq!(
        Logarithm::new().jacobian().eval(1e-320),
        Err(Error::NonFinite("Jacobian"))
    );
}

#[test]
fn typed_functions_compose_with_shared_and_reordered_graph_inputs() {
    let mut builder = Plan::builder(2);
    let polynomial = builder.add(Polynomial::new(), [Source::Input(1), Source::Input(0)]);
    let objective = builder.add(
        Rosenbrock::new(),
        [polynomial.output(1), polynomial.output(0)],
    );
    let plan = builder
        .build([objective.output(0), polynomial.output(1)])
        .unwrap();
    let point = [0.7, -0.3];
    let mut workspace = mercury::advanced::Workspace::new(&plan);
    let mut linearization = plan.linearize(&point, &mut workspace).unwrap();
    let values = Polynomial::new().eval(point[1], point[0]).unwrap();
    let (objective_value, gradient) = Rosenbrock::new()
        .value_and_gradient(values[1], values[0])
        .unwrap();
    close(
        linearization.value().unwrap(),
        &[objective_value, values[1]],
        1e-12,
    );
    let mut reverse = [0.0; 2];
    linearization.vjp(&[1.0, 2.0], &mut reverse).unwrap();
    close(
        &reverse,
        &[
            (gradient[0] + 2.0) * point[1] + gradient[1],
            (gradient[0] + 2.0) * point[0] + gradient[1] * 2.0 * point[1],
        ],
        1e-10,
    );
    let mut forward = [0.0; 2];
    linearization.jvp(&[0.2, -0.4], &mut forward).unwrap();
    close(
        &[forward[0] + 2.0 * forward[1]],
        &[reverse[0] * 0.2 - reverse[1] * 0.4],
        1e-10,
    );
}

#[test]
fn simple_plan_calls_return_owned_values_and_mathematical_derivatives() {
    let scalar = Plan::from_operator(Rosenbrock::new()).unwrap();
    let gradient = scalar.gradient();
    let value = scalar.eval(&[-1.2, 1.0]).unwrap();
    let derivatives = gradient.eval(&[-1.2, 1.0]).unwrap();
    close(&value, &[24.2], 1e-12);
    close(&derivatives, &[-215.6, -88.0], 1e-10);
    let (value, derivatives) = scalar.value_and_gradient(&[1.0, 1.0]).unwrap();
    close(&[value], &[0.0], 1e-12);
    close(&derivatives, &[0.0, 0.0], 1e-12);

    let vector = Plan::from_operator(Polynomial::new()).unwrap();
    let jacobian = vector.jacobian().eval(&[2.0, 3.0]).unwrap();
    assert_eq!((jacobian.nrows(), jacobian.ncols()), (3, 2));
    for (row, expected) in [[4.0, 1.0], [3.0, 2.0], [1.0, -18.0]].iter().enumerate() {
        for (column, expected) in expected.iter().enumerate() {
            close(&[jacobian[(row, column)]], &[*expected], 1e-12);
        }
    }
    assert!(matches!(
        vector.gradient().eval(&[2.0, 3.0]),
        Err(Error::Dimension {
            buffer: "gradient outputs",
            expected: 1,
            actual: 3
        })
    ));

    let solve = Plan::from_operator(mercury::DenseSolve::new(1).unwrap()).unwrap();
    let (solution, gradient) = solve.value_and_gradient(&[2.0, 6.0]).unwrap();
    close(&[solution], &[3.0], 1e-12);
    close(&gradient, &[-1.5, 0.5], 1e-12);
}

#[test]
fn simple_plan_failures_are_independent_and_empty_jacobians_keep_their_shape() {
    let plan = Plan::from_operator(Root::new()).unwrap();
    assert!(matches!(plan.eval(&[]), Err(Error::Dimension { .. })));
    assert_eq!(
        plan.gradient().eval(&[f64::NAN]),
        Err(Error::NonFinite("point"))
    );
    assert!(plan.jacobian().eval(&[f64::INFINITY]).is_err());
    assert!(matches!(
        plan.value_and_gradient(&[-1.0]),
        Err(Error::Operator { .. })
    ));
    let (value, gradient) = plan.value_and_gradient(&[4.0]).unwrap();
    close(&[value], &[2.0], 1e-12);
    close(&gradient, &[0.25], 1e-12);
    let owned_value = plan.eval(&[9.0]).unwrap();
    assert!(plan.eval(&[-1.0]).is_err());
    close(&owned_value, &[3.0], 1e-12);

    let empty = Plan::builder(2).build([]).unwrap();
    let jacobian = empty.jacobian().eval(&[1.0, 2.0]).unwrap();
    assert_eq!((jacobian.nrows(), jacobian.ncols()), (0, 2));
    assert!(empty.eval(&[1.0, 2.0]).unwrap().is_empty());
    assert!(empty.gradient().eval(&[1.0, 2.0]).is_err());
}
