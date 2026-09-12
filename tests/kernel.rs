#![feature(autodiff)]

//! Compiled kernel adapters: derivatives, buffers, configuration, and failures.

use mercury::{Error, Operator, Shape, differentiable};

struct Config {
    scale: f64,
}

#[differentiable(inputs = 2, outputs = 3)]
fn polynomial(config: &Config, input: &[f64], output: &mut [f64]) {
    output[0] = config.scale * (input[0] * input[0] + input[1]);
    output[1] = input[0] * input[1];
    output[2] = input[0] - 3.0 * input[1] * input[1];
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn configured_kernel_preserves_seeds_and_overwrites_repeated_products() {
    let kernel = polynomial_operator(Config { scale: 2.0 });
    assert_eq!(
        kernel.shape(),
        Shape {
            inputs: 2,
            outputs: 3
        }
    );
    let mut workspace = kernel.workspace();
    let point = [2.0, 3.0];
    let mut value = [0.0; 3];
    workspace.evaluate(&point, &mut value).unwrap();
    assert_close(&value, &[14.0, 6.0, -25.0]);

    let direction = [0.5, -0.25];
    workspace.jvp(&point, &direction, &mut value).unwrap();
    assert_close(&value, &[3.5, 1.0, 5.0]);

    let seed = [1.0, 2.0, -1.0];
    let mut gradient = [99.0; 2];
    for _ in 0..2 {
        workspace.vjp(&point, &seed, &mut gradient).unwrap();
        assert_close(&gradient, &[13.0, 24.0]);
        assert_close(&seed, &[1.0, 2.0, -1.0]);
        assert_close(&point, &[2.0, 3.0]);
        assert_close(&direction, &[0.5, -0.25]);
    }
}

#[test]
fn batches_preserve_direction_layout_and_handle_empty_and_partial_widths() {
    let kernel = polynomial_operator(Config { scale: 2.0 });
    let mut workspace = kernel.workspace();
    let seeds = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 2.0, 0.5, -0.25];
    let mut output = [99.0; 15];
    workspace
        .jvp_batch(&[2.0, 3.0], 5, &seeds, &mut output)
        .unwrap();
    assert_close(
        &output,
        &[
            8.0, 3.0, 1.0, 2.0, 2.0, -18.0, 10.0, 5.0, -17.0, -4.0, 1.0, -37.0, 3.5, 1.0, 5.0,
        ],
    );

    let weights = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, -1.0, 0.0, 0.0, 0.0,
    ];
    let mut gradient = [99.0; 10];
    workspace
        .vjp_batch(&[2.0, 3.0], 5, &weights, &mut gradient)
        .unwrap();
    assert_close(
        &gradient,
        &[8.0, 2.0, 3.0, 2.0, 1.0, -18.0, 13.0, 24.0, 0.0, 0.0],
    );

    workspace.jvp_batch(&[2.0, 3.0], 0, &[], &mut []).unwrap();
    workspace.vjp_batch(&[2.0, 3.0], 0, &[], &mut []).unwrap();
    assert!(matches!(
        workspace.jvp_batch(&[2.0, 3.0], 0, &[1.0], &mut []),
        Err(Error::Dimension { .. })
    ));
    assert_eq!(
        workspace.jvp_batch(&[2.0, 3.0], usize::MAX, &[], &mut []),
        Err(Error::SizeOverflow)
    );
}

#[test]
fn argument_errors_are_rejected_before_writing_outputs() {
    let kernel = polynomial_operator(Config { scale: 2.0 });
    let mut workspace = kernel.workspace();
    let mut output = [7.0; 3];
    assert!(matches!(
        workspace.evaluate(&[2.0], &mut output),
        Err(Error::Dimension { .. })
    ));
    assert!(matches!(
        workspace.jvp(&[2.0, 3.0], &[1.0], &mut output),
        Err(Error::Dimension { .. })
    ));
    assert_eq!(
        workspace.jvp(&[2.0, 3.0], &[1.0, f64::NAN], &mut output),
        Err(Error::NonFinite("seed"))
    );
    assert_close(&output, &[7.0; 3]);

    let mut batch_output = [7.0; 6];
    assert_eq!(
        workspace.jvp_batch(
            &[2.0, 3.0],
            2,
            &[1.0, 0.0, 1.0, f64::NAN],
            &mut batch_output,
        ),
        Err(Error::NonFinite("batch seeds"))
    );
    assert_close(&batch_output, &[7.0; 6]);
}

fn positive_inputs(config: &Config, input: &[f64]) -> mercury::Result<()> {
    if config.scale <= 0.0 || input.iter().any(|value| *value <= 0.0) {
        return Err(Error::Domain("positive scale and inputs required"));
    }
    Ok(())
}

#[test]
fn domains_and_nonfinite_kernel_results_are_reported() {
    let kernel = polynomial_operator(Config { scale: 2.0 }).with_domain(positive_inputs);
    let mut workspace = kernel.workspace();
    let mut output = [7.0; 3];
    assert_eq!(
        workspace.evaluate(&[2.0, -1.0], &mut output),
        Err(Error::Domain("positive scale and inputs required"))
    );
    assert_close(&output, &[7.0; 3]);

    let invalid = polynomial_operator(Config {
        scale: f64::INFINITY,
    });
    assert_eq!(
        invalid.workspace().evaluate(&[2.0, 3.0], &mut output),
        Err(Error::NonFinite("kernel output"))
    );
}

struct VectorConfig {
    dimension: usize,
}

#[differentiable(inputs = config.dimension, outputs = config.dimension)]
fn elementwise(config: &VectorConfig, input: &[f64], output: &mut [f64]) {
    for i in 0..config.dimension {
        output[i] = input[i] * input[i];
    }
}

#[differentiable(inputs = __mercury_shape.dimension, outputs = __mercury_shape.dimension)]
fn named_like_generated_local(__mercury_shape: &VectorConfig, input: &[f64], output: &mut [f64]) {
    elementwise(__mercury_shape, input, output);
}

#[differentiable(inputs = 1, outputs = 1)]
#[cfg(any())]
fn disabled(_config: &UnavailableType, _input: &[f64], _output: &mut [f64]) {}

#[test]
fn one_compiled_slice_kernel_accepts_runtime_instance_dimensions() {
    for dimension in [0, 1, 4] {
        let kernel = elementwise_operator(VectorConfig { dimension });
        let mut workspace = kernel.workspace();
        let mut output = vec![99.0; dimension];
        workspace
            .jvp(&vec![2.0; dimension], &vec![3.0; dimension], &mut output)
            .unwrap();
        assert_close(&output, &vec![12.0; dimension]);
        workspace
            .vjp(&vec![2.0; dimension], &vec![3.0; dimension], &mut output)
            .unwrap();
        assert_close(&output, &vec![12.0; dimension]);
    }

    let kernel = named_like_generated_local_operator(VectorConfig { dimension: 1 });
    let mut output = [0.0];
    kernel.workspace().jvp(&[2.0], &[3.0], &mut output).unwrap();
    assert_close(&output, &[12.0]);
}

#[differentiable(inputs = 1, outputs = 2)]
fn incomplete(_config: &(), input: &[f64], output: &mut [f64]) {
    output[0] = input[0] * input[0];
}

#[test]
fn incomplete_primal_writes_are_detected_in_value_and_derivative_calls() {
    let kernel = incomplete_operator(());
    let mut workspace = kernel.workspace();
    assert_eq!(
        workspace.evaluate(&[2.0], &mut [0.0; 2]),
        Err(Error::NonFinite("kernel output"))
    );
    assert_eq!(
        workspace.jvp(&[2.0], &[1.0], &mut [0.0; 2]),
        Err(Error::NonFinite("kernel output"))
    );
    assert_eq!(
        workspace.vjp(&[2.0], &[1.0, 0.0], &mut [0.0; 1]),
        Err(Error::NonFinite("kernel output"))
    );
}

#[differentiable(inputs = 1, outputs = 1)]
fn scaled(config: &Config, input: &[f64], output: &mut [f64]) {
    output[0] = config.scale * input[0];
}

#[test]
fn finite_primal_does_not_hide_nonfinite_derivatives() {
    let kernel = scaled_operator(Config { scale: f64::MAX });
    let mut workspace = kernel.workspace();
    workspace.evaluate(&[0.0], &mut [0.0]).unwrap();
    assert_eq!(
        workspace.jvp(&[0.0], &[2.0], &mut [0.0]),
        Err(Error::NonFinite("kernel JVP"))
    );
    assert_eq!(
        workspace.vjp(&[0.0], &[2.0], &mut [0.0]),
        Err(Error::NonFinite("kernel VJP"))
    );
}
