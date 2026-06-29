//! Generic differentiable math substrate for `pantheon-rs`.
//!
//! Mercury owns the core math contracts that higher layers build on: numeric
//! execution, automatic differentiation execution, symbolic tracing, linear
//! algebra facades, sparsity, derivative evaluators, and optimization-facing
//! derivative interfaces.
//!
//! The first implementation is intentionally small. Backends should be
//! introduced behind Mercury-owned traits and types rather than leaking backend
//! crates directly through the architecture.
#![forbid(unsafe_code)]

/// Scalar behavior required by Mercury's first-pass math surface.
///
/// This trait is deliberately small. It lets the crate establish a stable
/// Pantheon-owned scalar entry point before committing to specific AD, symbolic,
/// or linear algebra backends.
pub trait Scalar:
    Copy
    + PartialOrd
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    /// Additive identity.
    const ZERO: Self;

    /// Multiplicative identity.
    const ONE: Self;

    /// Sine.
    #[must_use]
    fn sin(self) -> Self;

    /// Cosine.
    #[must_use]
    fn cos(self) -> Self;

    /// Square root.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Absolute value.
    #[must_use]
    fn abs(self) -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn abs(self) -> Self {
        self.abs()
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn abs(self) -> Self {
        self.abs()
    }
}

/// Value-selection primitive that keeps branch intent explicit.
///
/// Symbolic backends will eventually implement this as an expression node. For
/// numeric scalars, it is just a normal boolean selection.
pub fn where_<T>(condition: bool, then_value: T, else_value: T) -> T {
    if condition { then_value } else { else_value }
}

/// Returns `x * x`.
pub fn square<T: Scalar>(x: T) -> T {
    x * x
}

#[cfg(test)]
mod tests {
    use super::{Scalar, square, where_};

    #[test]
    fn square_works_for_f64() {
        assert!((square(3.0_f64) - 9.0).abs() < 1.0e-12);
    }

    #[test]
    fn scalar_trig_uses_numeric_backend() {
        let x = core::f64::consts::FRAC_PI_2;
        assert!((x.sin() - <f64 as Scalar>::ONE).abs() < 1.0e-12);
    }

    #[test]
    fn where_selects_expected_branch() {
        assert_eq!(where_(true, 1, 2), 1);
        assert_eq!(where_(false, 1, 2), 2);
    }
}
