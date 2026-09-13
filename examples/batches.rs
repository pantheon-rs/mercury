#![feature(autodiff)]

//! Apply several derivative seeds at one point. Each seed and result is a row.

use mercury::{Plan, Source, differentiable};

#[differentiable(inputs = 2, outputs = 2)]
fn function(_config: &(), input: &[f64], output: &mut [f64]) {
    let x = input[0];
    let y = input[1];
    output[0] = x * x;
    output[1] = x * y;
}

fn main() -> mercury::Result<()> {
    let mut builder = Plan::builder(2);
    let function = builder.add(function_operator(()), [Source::Input(0), Source::Input(1)]);
    let plan = builder.build([function.output(0), function.output(1)])?;
    let mut workspace = plan.workspace();
    let point = [2.0, 3.0];
    let mut linearization = plan.linearize(&point, &mut workspace)?;

    // Three rows: [1, 0], [0, 1], [1, 2]. Batches currently use scalar loops.
    let seeds = [1.0, 0.0, 0.0, 1.0, 1.0, 2.0];
    let mut tangents = [0.0; 6];
    let mut gradients = [0.0; 6];
    linearization.jvp_batch(3, &seeds, &mut tangents)?;
    linearization.vjp_batch(3, &seeds, &mut gradients)?;

    println!("f(x, y) = [x^2, x*y], at (2, 3); J = [[4, 0], [3, 2]]");
    for row in 0..3 {
        let entries = 2 * row..2 * row + 2;
        println!(
            "seed {:?}: JVP {:?}, VJP {:?}",
            &seeds[entries.clone()],
            &tangents[entries.clone()],
            &gradients[entries]
        );
    }
    println!("Expected JVP rows: [4, 3], [0, 2], [4, 7]");
    println!("Expected VJP rows: [4, 0], [3, 2], [10, 4]");
    for (actual, expected) in tangents.into_iter().zip([4.0, 3.0, 0.0, 2.0, 4.0, 7.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in gradients.into_iter().zip([4.0, 0.0, 3.0, 2.0, 10.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    Ok(())
}
