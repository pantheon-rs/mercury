//! Unit quaternion for attitude representation (scalar-first, Hamilton).

use std::ops::Mul;

use crate::core::{SMatrix, SVector};

/// Quaternion `w + xi + yj + zk` (scalar-first, Hamilton convention).
///
/// Rotation methods ([`Quaternion::rotate`], [`Quaternion::to_dcm`]) assume
/// a unit quaternion; call [`Quaternion::normalized`] first when in doubt.
/// All operations are analytic and kernel-safe; `normalized` inherits the
/// sqrt kink at zero norm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    /// Scalar part.
    pub w: f64,
    /// First vector component (i).
    pub x: f64,
    /// Second vector component (j).
    pub y: f64,
    /// Third vector component (k).
    pub z: f64,
}

impl Quaternion {
    /// Builds a quaternion from components (scalar first).
    #[must_use]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Identity rotation.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Rotation of `angle` radians about a **unit** `axis`.
    #[must_use]
    pub fn from_axis_angle(axis: &SVector<3>, angle: f64) -> Self {
        let (s, c) = (angle / 2.0).sin_cos();
        Self::new(c, s * axis[0], s * axis[1], s * axis[2])
    }

    /// Conjugate (inverse for unit quaternions).
    #[must_use]
    pub const fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Euclidean norm of the 4-tuple.
    #[must_use]
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Unit-norm copy. Kink at zero norm (division by `norm()`).
    #[must_use]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        Self::new(self.w / n, self.x / n, self.y / n, self.z / n)
    }

    /// Rotates a vector: `q v q*` for unit `q`, via the two-cross expansion
    /// `v + w t + qv x t` with `t = 2 qv x v`.
    #[must_use]
    pub fn rotate(&self, v: &SVector<3>) -> SVector<3> {
        let qv = SVector::new([self.x, self.y, self.z]);
        let t = qv.cross(v) * 2.0;
        *v + t * self.w + qv.cross(&t)
    }

    /// Direction-cosine (rotation) matrix equivalent, for unit `q`.
    #[must_use]
    pub fn to_dcm(&self) -> SMatrix<3, 3> {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        SMatrix::new([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ])
    }
}

impl Mul for Quaternion {
    type Output = Self;
    /// Hamilton product `self * rhs` (applies `rhs` first, then `self`).
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        )
    }
}
