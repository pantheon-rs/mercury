#![feature(autodiff)]
//! 4. One compiled slice kernel, with an inactive runtime iteration count.

use mercury::Plan;

struct Config {
    steps: usize,
}

#[mercury::advanced::differentiable(inputs = 2, outputs = 1)]
fn decay(config: &Config, input: &[f64], output: &mut [f64]) {
    // Starting the retention accumulator from a constant avoids the pinned
    // compiler's type-inference failure for some slice-initialized loop states.
    let mut retention = 1.0;
    for _ in 0..config.steps {
        retention *= input[1];
    }
    output[0] = input[0] * retention;
}

fn main() -> mercury::Result<()> {
    // state = initial * factor^steps. The count is configuration, not an
    // active input: derivatives are with respect to initial and factor only.
    for (steps, expected_value, expected_gradient) in [
        (0, 3.0, [1.0, 0.0]),
        (1, 1.5, [0.5, 3.0]),
        (4, 0.1875, [0.0625, 1.5]),
    ] {
        let function = Plan::from_operator(decay_operator(Config { steps }))?;
        let (value, gradient) = function.value_and_gradient(&[3.0, 0.5])?;
        assert!((value - expected_value).abs() < 1e-12);
        for (actual, expected) in gradient.iter().zip(expected_gradient) {
            assert!((actual - expected).abs() < 1e-12);
        }
        println!("steps={steps}: state={value}, [d_initial, d_factor]={gradient:?}");
    }
    Ok(())
}
