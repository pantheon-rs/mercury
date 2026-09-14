//! Preparation costs and derivative direction selection are observable contracts.

use mercury::advanced::{Kernel, PlanExecution, Shape, Workspace};
use mercury::{Plan, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct Calls {
    dependencies: AtomicUsize,
    forward: AtomicUsize,
    reverse: AtomicUsize,
    fail_after: AtomicUsize,
}

fn value(_calls: &Arc<Calls>, point: &[f64], output: &mut [f64]) {
    output[0] = point.iter().sum();
}
fn forward(
    calls: &Arc<Calls>,
    point: &[f64],
    seed: &[f64],
    primal: &mut [f64],
    output: &mut [f64],
) {
    calls.forward.fetch_add(1, Ordering::Relaxed);
    value(calls, point, primal);
    output[0] = seed.iter().sum();
}
fn reverse(
    calls: &Arc<Calls>,
    point: &[f64],
    output: &mut [f64],
    primal: &mut [f64],
    seed: &mut [f64],
) {
    calls.reverse.fetch_add(1, Ordering::Relaxed);
    value(calls, point, primal);
    for entry in output {
        *entry += seed[0];
    }
}
fn dependency(calls: &Arc<Calls>, _output: usize, _input: usize) -> bool {
    calls.dependencies.fetch_add(1, Ordering::Relaxed);
    true
}

#[test]
fn values_and_products_do_not_prepare_sparsity_and_wide_assembly_uses_reverse() -> Result<()> {
    let calls = Arc::new(Calls::default());
    let kernel = Kernel::new(
        calls.clone(),
        Shape {
            inputs: 64,
            outputs: 1,
        },
        value,
        forward,
        reverse,
    )
    .with_dependencies(dependency);
    let plan = Plan::from_operator(kernel)?;
    let point = [1.0; 64];
    let mut workspace = Workspace::new(&plan);
    let mut output = [0.0];
    plan.evaluate(&point, &mut workspace, &mut output)?;
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    linearization.jvp(&point, &mut output)?;
    assert_eq!(calls.dependencies.load(Ordering::Relaxed), 0);
    assert_eq!(calls.forward.load(Ordering::Relaxed), 1);

    let mut jacobian = [0.0; 64];
    linearization.jacobian(&mut jacobian)?;
    assert!(jacobian.iter().all(|entry| (*entry - 1.0).abs() < 1e-14));
    assert_eq!(calls.dependencies.load(Ordering::Relaxed), 64);
    assert_eq!(calls.forward.load(Ordering::Relaxed), 1);
    assert_eq!(calls.reverse.load(Ordering::Relaxed), 1);
    linearization.sparse_jacobian(&mut jacobian)?;
    let clone = plan.clone();
    clone.jacobian().eval_sparse(&point)?;
    assert_eq!(calls.dependencies.load(Ordering::Relaxed), 64);
    assert_eq!(calls.reverse.load(Ordering::Relaxed), 3);
    Ok(())
}

fn sparse_value(_calls: &Arc<Calls>, point: &[f64], output: &mut [f64]) {
    output[0] = point[0] + 2.0 * point[2];
    output[1] = point[1] + 3.0 * point[2] + 4.0 * point[3];
}
fn sparse_forward(
    calls: &Arc<Calls>,
    point: &[f64],
    seed: &[f64],
    primal: &mut [f64],
    output: &mut [f64],
) {
    calls.forward.fetch_add(1, Ordering::Relaxed);
    sparse_value(calls, point, primal);
    sparse_value(calls, seed, output);
}
fn sparse_reverse(
    calls: &Arc<Calls>,
    point: &[f64],
    output: &mut [f64],
    primal: &mut [f64],
    seed: &mut [f64],
) {
    calls.reverse.fetch_add(1, Ordering::Relaxed);
    sparse_value(calls, point, primal);
    output[0] += seed[0];
    output[1] += seed[1];
    output[2] += 2.0 * seed[0] + 3.0 * seed[1];
    output[3] += 4.0 * seed[1];
}
const fn sparse_dependency(_calls: &Arc<Calls>, output: usize, input: usize) -> bool {
    if output == 0 {
        input == 0 || input == 2
    } else {
        input != 0
    }
}
fn fail_after_reverse(calls: &Arc<Calls>, _point: &[f64]) -> Result<()> {
    let limit = calls.fail_after.load(Ordering::Relaxed);
    if limit != 0 && calls.reverse.load(Ordering::Relaxed) >= limit {
        Err(mercury::Error::Domain("injected reverse failure"))
    } else {
        Ok(())
    }
}

#[test]
fn reverse_sparse_scatter_preserves_zeros_and_recovers_after_partial_failure() -> Result<()> {
    let calls = Arc::new(Calls::default());
    let kernel = Kernel::new(
        calls.clone(),
        Shape {
            inputs: 4,
            outputs: 2,
        },
        sparse_value,
        sparse_forward,
        sparse_reverse,
    )
    .with_dependencies(sparse_dependency)
    .with_domain(fail_after_reverse);
    let plan = Plan::from_operator(kernel)?;
    let mut workspace = Workspace::new(&plan);
    let point = [1.0; 4];
    let mut linearization = plan.linearize(&point, &mut workspace)?;
    let mut sparse = [0.0; 5];
    linearization.sparse_jacobian(&mut sparse)?;
    for (actual, expected) in sparse.iter().zip([1.0, 1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-14);
    }
    assert_eq!(calls.forward.load(Ordering::Relaxed), 0);
    assert_eq!(calls.reverse.load(Ordering::Relaxed), 2);
    calls.fail_after.store(3, Ordering::Relaxed);
    assert!(linearization.sparse_jacobian(&mut sparse).is_err());
    assert!(sparse.iter().all(|entry| entry.is_nan()));
    assert!(matches!(
        linearization.value(),
        Err(mercury::Error::InvalidLinearization)
    ));
    calls.fail_after.store(0, Ordering::Relaxed);
    let mut dense = [0.0; 8];
    plan.linearize(&point, &mut workspace)?
        .jacobian(&mut dense)?;
    for (actual, expected) in dense.iter().zip([1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-14);
    }
    Ok(())
}
