//! Differentiable math substrate for `pantheon-rs`.
//!
//! Mercury Phase 1 is a plain-`f64` core differentiated with Rust nightly
//! `std::autodiff` / Enzyme. The current crate is still a small scaffold; these
//! helpers are not the final automatic differentiation contract.
#![forbid(unsafe_code)]

/// Temporary scalar behavior from Mercury's initial scaffold.
///
/// Phase 1 should not expand this into a generic AD or symbolic model contract.
/// The planned core model path is ordinary `f64` code differentiated by Enzyme.
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

/// Value-selection primitive retained from the initial scaffold.
///
/// Phase 1 model code may use ordinary Rust control flow. This helper remains
/// useful when piecewise behavior should be visually explicit.
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
