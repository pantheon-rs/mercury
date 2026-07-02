//! Derivative validation utilities for Mercury's Phase 1 `f64` core.

use std::fmt;

/// Error returned by derivative validation helpers.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Finite-difference step must be finite and positive.
    InvalidStep {
        /// Rejected finite-difference step.
        step: f64,
    },
    /// Gradient vectors must have identical lengths.
    LengthMismatch {
        /// Length of the actual gradient vector.
        actual: usize,
        /// Length of the expected gradient vector.
        expected: usize,
    },
    /// Gradient comparison needs at least one element.
    EmptyGradient,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStep { step } => {
                write!(
                    f,
                    "finite-difference step must be finite and positive, got {step}"
                )
            }
            Self::LengthMismatch { actual, expected } => {
                write!(
                    f,
                    "gradient length mismatch: actual has {actual} entries, expected has {expected}"
                )
            }
            Self::EmptyGradient => write!(f, "gradient comparison needs at least one element"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Summary of an element-wise gradient comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientCheck {
    /// Largest absolute element-wise error.
    pub max_abs_error: f64,
    /// Largest relative element-wise error.
    pub max_rel_error: f64,
    /// Index where the largest absolute error occurred.
    pub worst_index: usize,
}

/// Computes a dense central-difference gradient for a scalar-output function.
///
/// Perturbations are scaled as `step * max(1, abs(x[i]))`, which keeps the probe
/// useful for both small and large input magnitudes.
///
/// # Errors
///
/// Returns [`ValidationError::InvalidStep`] when `step` is not finite and
/// positive.
pub fn central_difference_gradient<F>(
    f: F,
    x: &[f64],
    step: f64,
) -> Result<Vec<f64>, ValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    if !step.is_finite() || step <= 0.0 {
        return Err(ValidationError::InvalidStep { step });
    }

    let mut gradient = vec![0.0; x.len()];
    let mut probe = x.to_vec();

    for (index, &x_i) in x.iter().enumerate() {
        let perturbation = step * x_i.abs().max(1.0);

        probe[index] = x_i + perturbation;
        let f_plus = f(&probe);

        probe[index] = x_i - perturbation;
        let f_minus = f(&probe);

        probe[index] = x_i;
        gradient[index] = (f_plus - f_minus) / (2.0 * perturbation);
    }

    Ok(gradient)
}

/// Compares two dense gradients and reports the worst absolute and relative
/// errors.
///
/// The `actual` vector is usually the automatic-differentiation result. The
/// `expected` vector is usually finite-difference or analytic reference data.
///
/// # Errors
///
/// Returns [`ValidationError::LengthMismatch`] when vector lengths differ, or
/// [`ValidationError::EmptyGradient`] when both vectors are empty.
pub fn compare_gradients(
    actual: &[f64],
    expected: &[f64],
) -> Result<GradientCheck, ValidationError> {
    if actual.len() != expected.len() {
        return Err(ValidationError::LengthMismatch {
            actual: actual.len(),
            expected: expected.len(),
        });
    }
    if actual.is_empty() {
        return Err(ValidationError::EmptyGradient);
    }

    let mut check = GradientCheck {
        max_abs_error: 0.0,
        max_rel_error: 0.0,
        worst_index: 0,
    };

    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let abs_error = (actual_value - expected_value).abs();
        let scale = actual_value
            .abs()
            .max(expected_value.abs())
            .max(f64::EPSILON);
        let rel_error = abs_error / scale;

        if abs_error > check.max_abs_error {
            check.max_abs_error = abs_error;
            check.worst_index = index;
        }
        if rel_error > check.max_rel_error {
            check.max_rel_error = rel_error;
        }
    }

    Ok(check)
}
