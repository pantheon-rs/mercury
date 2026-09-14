#![feature(autodiff)]
//! 5. An active stopping condition: differentiate the executed iteration sequence.

#[mercury::function(Accumulation, first_order)]
fn accumulate(increment: f64) -> f64 {
    let mut total = 0.0;
    let mut iterations = 0;
    // The explicit cap keeps the calculation bounded even for zero/negative input.
    while total < 1.0 && iterations < 16 {
        total += increment;
        iterations += 1;
    }
    total
}

fn main() -> mercury::Result<()> {
    let function = Accumulation::new();
    for (increment, expected_value, expected_slope) in [
        (0.375, 1.125, 3.0),
        (0.625, 1.25, 2.0),
        (0.03125, 0.5, 16.0),
    ] {
        let (value, [slope]) = function.value_and_gradient(increment)?;
        assert!((value - expected_value).abs() < 1e-12);
        assert!((slope - expected_slope).abs() < 1e-12);
        // Both perturbations keep the same stopping decisions in these cases.
        let step = 1e-6;
        let difference =
            (function.eval(increment + step)? - function.eval(increment - step)?) / (2.0 * step);
        assert!((slope - difference).abs() < 1e-8);
        println!("increment={increment}: total={value}, derivative={slope}");
    }
    // At 0.25, four increments hit the threshold exactly. Just below it, five
    // increments are taken. The returned total is discontinuous at this boundary.
    let boundary = function.eval(0.25)?;
    let below = function.eval(0.25 - 1e-6)?;
    assert!((boundary - 1.0).abs() < 1e-12);
    assert!(below > 1.24);
    println!(
        "at increment=0.25: total={boundary}, just below: {below}; no derivative at this jump"
    );
    Ok(())
}
