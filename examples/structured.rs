#![feature(autodiff)]
//! Vector arguments keep their names and shapes in the gradient.

use mercury::function;

#[function(Energy)]
fn energy(velocity: [f64; 2], mass: f64) -> f64 {
    0.5 * mass * (velocity[0] * velocity[0] + velocity[1] * velocity[1])
}

#[allow(
    clippy::float_cmp,
    reason = "These reference values are exactly representable."
)]
fn main() -> mercury::Result<()> {
    let function = Energy::new();
    let (value, gradient) = function.value_and_gradient([3.0, 4.0], 2.0)?;
    println!("energy = {value}");
    println!("d_energy/d_velocity = {:?}", gradient.velocity);
    println!("d_energy/d_mass = {}", gradient.mass);
    assert_eq!(gradient.velocity, [6.0, 8.0]);
    assert_eq!(gradient.mass, 12.5);
    Ok(())
}
