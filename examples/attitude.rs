#![feature(autodiff)]
//! Body-to-world force rotation. Forces use newtons; local angles use radians.
//! The host owns frames and units. Mercury sees explicit Euclidean coordinates.

/// Scalar-first quaternion storage, normalized inside the calculation.
/// Requires a finite nonzero quaternion with representable squared norm.
#[mercury::function(RotatedForce)]
pub fn rotated_force(quaternion: [f64; 4], body: [f64; 3]) -> [f64; 3] {
    let norm = (quaternion[0] * quaternion[0]
        + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3])
        .sqrt();
    let scalar = quaternion[0] / norm;
    let x = quaternion[1] / norm;
    let y = quaternion[2] / norm;
    let z = quaternion[3] / norm;
    let tx = 2.0 * (y * body[2] - z * body[1]);
    let ty = 2.0 * (z * body[0] - x * body[2]);
    let tz = 2.0 * (x * body[1] - y * body[0]);
    [
        body[0] + scalar * tx + y * tz - z * ty,
        body[1] + scalar * ty + z * tx - x * tz,
        body[2] + scalar * tz + x * ty - y * tx,
    ]
}

/// A local chart about identity: normalize `[1, delta/2]`.
///
/// Its derivative at delta=0 is the three-coordinate infinitesimal attitude perturbation.
/// This is a retraction, not a global rotation-vector parameterization.
#[mercury::function(LocalForce)]
pub fn local_force(delta: [f64; 3], body: [f64; 3]) -> [f64; 3] {
    rotated_force([1.0, 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2]], body)
}

fn main() -> mercury::Result<()> {
    let jacobian = LocalForce::new()
        .jacobian()
        .eval([0.0; 3], [10.0, 0.0, 0.0])?;
    assert_eq!(
        jacobian.delta,
        [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, -10.0, 0.0]]
    );
    println!(
        "world force sensitivity to three local attitude coordinates: {:?}",
        jacobian.delta
    );
    Ok(())
}
