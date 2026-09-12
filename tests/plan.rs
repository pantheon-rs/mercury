//! Numerical and lifecycle checks independent of the compiler adapter.

use mercury::{Error, Operator, OperatorWorkspace, Plan, Result, Shape, Source};

struct Polynomial;

impl Operator for Polynomial {
    fn shape(&self) -> Shape {
        Shape {
            inputs: 2,
            outputs: 3,
        }
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(Self)
    }
}

impl OperatorWorkspace for Polynomial {
    fn evaluate(&mut self, q: &[f64], out: &mut [f64]) -> Result<()> {
        out.copy_from_slice(&[q[0] * q[0] + q[1], q[0] * q[1], q[0] - q[1] * q[1]]);
        Ok(())
    }

    fn jvp(&mut self, q: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        out.copy_from_slice(&[
            2.0 * q[0] * v[0] + v[1],
            q[1] * v[0] + q[0] * v[1],
            v[0] - 2.0 * q[1] * v[1],
        ]);
        Ok(())
    }

    fn vjp(&mut self, q: &[f64], w: &[f64], out: &mut [f64]) -> Result<()> {
        out.copy_from_slice(&[
            2.0 * q[0] * w[0] + q[1] * w[1] + w[2],
            w[0] + q[0] * w[1] - 2.0 * q[1] * w[2],
        ]);
        Ok(())
    }
}

struct Sum;

impl Operator for Sum {
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

impl OperatorWorkspace for Sum {
    fn evaluate(&mut self, q: &[f64], out: &mut [f64]) -> Result<()> {
        out[0] = q[0] + q[1];
        Ok(())
    }

    fn jvp(&mut self, _q: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        out[0] = v[0] + v[1];
        Ok(())
    }

    fn vjp(&mut self, _q: &[f64], w: &[f64], out: &mut [f64]) -> Result<()> {
        out.fill(w[0]);
        Ok(())
    }
}

fn composed_plan() -> Plan {
    let mut builder = Plan::builder(3);
    let first = builder.add(Polynomial, [Source::Input(0), Source::Input(1)]);
    let second = builder.add(Polynomial, [first.output(0), Source::Input(2)]);
    let shared = builder.add(Sum, [first.output(0), second.output(1)]);
    let repeated = builder.add(Sum, [first.output(2), first.output(2)]);
    builder
        .build([
            shared.output(0),
            repeated.output(0),
            second.output(2),
            shared.output(0),
            Source::Input(2),
        ])
        .unwrap()
}

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

#[test]
fn composition_matches_directional_difference_and_adjoint_identity() {
    let plan = composed_plan();
    let point = [1.2, -0.7, 0.4];
    let direction = [0.3, -0.4, 0.8];
    let weights = [0.5, -1.1, 0.8, 0.3, 0.6];
    let mut workspace = plan.workspace();
    let mut tangent = [0.0; 5];
    let mut cotangent = [0.0; 3];
    let mut jacobian = [0.0; 15];
    {
        let mut linearization = plan.linearize(&point, &mut workspace).unwrap();
        close(
            linearization.value().unwrap(),
            &[1.036, 1.42, 0.58, 1.036, 0.4],
            1e-12,
        );
        linearization.jvp(&direction, &mut tangent).unwrap();
        linearization.vjp(&weights, &mut cotangent).unwrap();
        linearization.jacobian(&mut jacobian).unwrap();
    }
    let h = 1e-6;
    let plus: [f64; 3] = std::array::from_fn(|i| point[i] + h * direction[i]);
    let minus: [f64; 3] = std::array::from_fn(|i| point[i] - h * direction[i]);
    let mut upper = [0.0; 5];
    let mut lower = [0.0; 5];
    plan.evaluate(&plus, &mut workspace, &mut upper).unwrap();
    plan.evaluate(&minus, &mut workspace, &mut lower).unwrap();
    let difference: [f64; 5] = std::array::from_fn(|i| (upper[i] - lower[i]) / (2.0 * h));
    close(&tangent, &difference, 1e-8);
    close(
        &[dot(&weights, &tangent)],
        &[dot(&direction, &cotangent)],
        1e-12,
    );
    let assembled: Vec<_> = jacobian
        .chunks_exact(3)
        .map(|row| dot(row, &direction))
        .collect();
    close(&assembled, &tangent, 1e-12);
    assert_eq!(
        plan.dependencies(),
        &[
            vec![0, 1, 2],
            vec![0, 1],
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![2]
        ]
    );
}

#[test]
fn batches_preserve_seeds_and_match_independent_calls() {
    let plan = composed_plan();
    let point = [0.8, 0.2, -0.4];
    let mut workspace = plan.workspace();
    let mut linearization = plan.linearize(&point, &mut workspace).unwrap();
    let seeds = [
        0.2, -0.3, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -0.1, 0.7, 0.4,
    ];
    let saved_seeds = seeds;
    let mut tangents = [9.0; 25];
    linearization.jvp_batch(5, &seeds, &mut tangents).unwrap();
    for (seed, actual) in seeds.chunks_exact(3).zip(tangents.chunks_exact(5)) {
        let mut expected = [0.0; 5];
        linearization.jvp(seed, &mut expected).unwrap();
        close(actual, &expected, 1e-12);
    }
    let saved_tangents = tangents;
    let mut cotangents = [9.0; 15];
    linearization
        .vjp_batch(5, &tangents, &mut cotangents)
        .unwrap();
    for (seed, actual) in tangents.chunks_exact(5).zip(cotangents.chunks_exact(3)) {
        let mut expected = [0.0; 3];
        linearization.vjp(seed, &mut expected).unwrap();
        close(actual, &expected, 1e-12);
    }
    close(&seeds, &saved_seeds, 0.0);
    close(&tangents, &saved_tangents, 0.0);
    linearization.jvp_batch(0, &[], &mut []).unwrap();
    linearization.vjp_batch(0, &[], &mut []).unwrap();
}

#[test]
fn wiring_checks_forward_references_cycles_and_foreign_nodes() {
    let mut builder = Plan::builder(2);
    let first = builder.add(Sum, [Source::Input(0), Source::Input(1)]);
    let second = builder.add(Sum, [Source::Input(0), Source::Input(1)]);
    builder
        .connect(first, [second.output(0), Source::Input(1)])
        .unwrap();
    let plan = builder.build([first.output(0)]).unwrap();
    let mut workspace = plan.workspace();
    let mut value = [0.0];
    plan.evaluate(&[2.0, 3.0], &mut workspace, &mut value)
        .unwrap();
    close(&value, &[8.0], 0.0);

    let mut cyclic = Plan::builder(1);
    let a = cyclic.add(Sum, [Source::Input(0), Source::Input(0)]);
    let b = cyclic.add(Sum, [a.output(0), Source::Input(0)]);
    cyclic.connect(a, [b.output(0), Source::Input(0)]).unwrap();
    assert!(matches!(cyclic.build([b.output(0)]), Err(Error::Cycle)));

    let mut foreign = Plan::builder(1);
    assert_eq!(foreign.connect(first, []), Err(Error::ForeignNode));
    assert!(matches!(
        foreign.build([first.output(0)]),
        Err(Error::ForeignNode)
    ));
    assert!(matches!(
        Plan::builder(1).build([Source::Input(1)]),
        Err(Error::InvalidSource)
    ));
    let mut bad_shape = Plan::builder(1);
    let node = bad_shape.add(Sum, [Source::Input(0)]);
    assert!(matches!(
        bad_shape.build([node.output(0)]),
        Err(Error::Dimension { .. })
    ));
}

#[test]
fn argument_errors_leave_a_valid_linearization_usable() {
    let plan = composed_plan();
    let other = composed_plan();
    let mut workspace = plan.workspace();
    assert!(matches!(
        other.linearize(&[0.0; 3], &mut workspace),
        Err(Error::ForeignWorkspace)
    ));
    let mut linearization = plan.linearize(&[0.0; 3], &mut workspace).unwrap();
    let mut output = [42.0; 5];
    assert!(matches!(
        linearization.jvp(&[1.0], &mut output),
        Err(Error::Dimension { .. })
    ));
    assert!(matches!(
        linearization.jvp_batch(usize::MAX, &[], &mut []),
        Err(Error::SizeOverflow)
    ));
    assert!(matches!(
        linearization.jvp(&[f64::NAN; 3], &mut output),
        Err(Error::NonFinite(_))
    ));
    close(&output, &[42.0; 5], 0.0);
    assert!(linearization.value().is_ok());
    linearization.jvp(&[1.0; 3], &mut output).unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
}

struct Fails;

impl Operator for Fails {
    fn shape(&self) -> Shape {
        Shape {
            inputs: 1,
            outputs: 1,
        }
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(Self)
    }
}

impl OperatorWorkspace for Fails {
    fn evaluate(&mut self, q: &[f64], out: &mut [f64]) -> Result<()> {
        if q[0] < 0.0 {
            return Ok(()); // Deliberately incomplete write must be detected by the plan.
        }
        out[0] = q[0];
        Ok(())
    }

    fn jvp(&mut self, _q: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        if v[0] < 0.0 {
            return Err(Error::Domain("deliberate derivative failure"));
        }
        out.copy_from_slice(v);
        Ok(())
    }

    fn vjp(&mut self, q: &[f64], w: &[f64], out: &mut [f64]) -> Result<()> {
        self.jvp(q, w, out)
    }
}

#[test]
fn numerical_failures_invalidate_entire_batches_and_allow_fresh_preparation() {
    let mut builder = Plan::builder(1);
    let node = builder.add(Fails, [Source::Input(0)]);
    let plan = builder.build([node.output(0)]).unwrap();
    let mut workspace = plan.workspace();
    {
        let mut linearization = plan.linearize(&[1.0], &mut workspace).unwrap();
        let mut out = [0.0; 2];
        assert!(linearization.jvp_batch(2, &[1.0, -1.0], &mut out).is_err());
        assert!(out.iter().all(|x| x.is_nan()));
        assert_eq!(linearization.value(), Err(Error::InvalidLinearization));
        assert_eq!(
            linearization.vjp(&[1.0], &mut [0.0]),
            Err(Error::InvalidLinearization)
        );
    }
    assert!(plan.linearize(&[-1.0], &mut workspace).is_err());
    let linearization = plan.linearize(&[2.0], &mut workspace).unwrap();
    close(linearization.value().unwrap(), &[2.0], 0.0);
}

#[test]
fn empty_plans_and_repeated_published_outputs_have_correct_products() {
    let plan = Plan::builder(1)
        .build([Source::Input(0), Source::Input(0)])
        .unwrap();
    let mut workspace = plan.workspace();
    let mut linearization = plan.linearize(&[3.0], &mut workspace).unwrap();
    let mut output = [0.0];
    linearization.vjp(&[2.0, 4.0], &mut output).unwrap();
    close(&output, &[6.0], 0.0);

    let empty = Plan::builder(0).build([]).unwrap();
    let mut workspace = empty.workspace();
    let mut linearization = empty.linearize(&[], &mut workspace).unwrap();
    linearization.jvp_batch(3, &[], &mut []).unwrap();
    linearization.vjp_batch(3, &[], &mut []).unwrap();
    linearization.jvp_batch(usize::MAX, &[], &mut []).unwrap();
    linearization.vjp_batch(usize::MAX, &[], &mut []).unwrap();
    linearization.jacobian(&mut []).unwrap();
}

struct Constant;

impl Operator for Constant {
    fn shape(&self) -> Shape {
        Shape {
            inputs: 0,
            outputs: 1,
        }
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(Self)
    }
}

impl OperatorWorkspace for Constant {
    fn evaluate(&mut self, _q: &[f64], out: &mut [f64]) -> Result<()> {
        out[0] = 1.0;
        Ok(())
    }

    fn jvp(&mut self, _q: &[f64], _seed: &[f64], out: &mut [f64]) -> Result<()> {
        out.fill(0.0);
        Ok(())
    }

    fn vjp(&mut self, _q: &[f64], _seed: &[f64], _out: &mut [f64]) -> Result<()> {
        Ok(())
    }
}

#[test]
fn allocation_layout_overflow_is_rejected_before_dispatch() {
    assert!(matches!(
        Plan::builder(usize::MAX).build([]),
        Err(Error::SizeOverflow)
    ));
    let mut builder = Plan::builder(0);
    builder.add(Constant, []);
    let plan = builder.build([]).unwrap();
    let mut workspace = plan.workspace();
    let mut linearization = plan.linearize(&[], &mut workspace).unwrap();
    assert_eq!(
        linearization.jvp_batch(usize::MAX, &[], &mut []),
        Err(Error::SizeOverflow)
    );
    assert_eq!(
        linearization.vjp_batch(usize::MAX, &[], &mut []),
        Err(Error::SizeOverflow)
    );
    assert!(linearization.value().is_ok());
    linearization.jvp_batch(2, &[], &mut []).unwrap();
}
