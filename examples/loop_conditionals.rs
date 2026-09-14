#![feature(autodiff)]
//! 3. Combine a loop and a conditional to accumulate an array-valued input's loss.

#[mercury::function(PositivePenalty, first_order)]
fn positive_penalty(errors: [f64; 4]) -> f64 {
    let mut penalty = 0.0;
    for error in errors {
        if error > 0.0 {
            penalty += error * error;
        }
    }
    penalty
}

fn main() -> mercury::Result<()> {
    let function = PositivePenalty::new();
    for (errors, expected_value, expected_gradient) in [
        ([-2.0, -1.0, 1.0, 3.0], 10.0, [0.0, 0.0, 2.0, 6.0]),
        ([2.0, 0.0, -1.0, 1.0], 5.0, [4.0, 0.0, 0.0, 2.0]),
    ] {
        let (value, gradient) = function.value_and_gradient(errors)?;
        assert!((value - expected_value).abs() < 1e-12);
        for (actual, expected) in gradient.errors.iter().zip(expected_gradient) {
            assert!((actual - expected).abs() < 1e-12);
        }
        println!(
            "errors={errors:?}: penalty={value}, gradient={:?}",
            gradient.errors
        );
    }
    // Unlike abs(x), max(x,0)^2 has a first derivative at zero: both sides
    // approach zero slope. Its second derivative does not exist there.
    Ok(())
}
