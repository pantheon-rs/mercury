//! Scalar objective API built on Enzyme reverse-mode automatic differentiation.

/// Scalar objective value and dense gradient evaluated at one input point.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueGradient {
    /// Objective value.
    pub value: f64,
    /// Dense gradient with respect to the input slice.
    pub gradient: Vec<f64>,
}

impl ValueGradient {
    /// Creates a value-gradient pair.
    #[must_use]
    pub fn new(value: f64, gradient: Vec<f64>) -> Self {
        Self { value, gradient }
    }

    /// Returns the input dimension represented by this gradient.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.gradient.len()
    }
}

/// Defines an Enzyme-backed scalar objective module.
///
/// The generated module contains:
///
/// - `value(x: &[f64]) -> f64`
/// - `gradient(x: &[f64]) -> Vec<f64>`
/// - `value_and_gradient(x: &[f64]) -> ValueGradient`
///
/// Raw Enzyme activity markers, output buffers, and shadow buffers stay inside
/// the generated module.
///
/// # Example
///
/// ```ignore
/// mercury::scalar_objective! {
///     pub mod quadratic(x) {
///         x[0] * x[0] + 3.0 * x[1] * x[1]
///     }
/// }
///
/// let result = quadratic::value_and_gradient(&[2.0, -1.0]);
/// assert_eq!(result.value, 7.0);
/// assert_eq!(result.gradient, vec![4.0, -6.0]);
/// ```
#[macro_export]
macro_rules! scalar_objective {
    (
        $(#[$meta:meta])*
        $vis:vis mod $name:ident($x:ident) $body:block
    ) => {
        $(#[$meta])*
        $vis mod $name {
            use std::autodiff::autodiff_reverse;

            #[autodiff_reverse(__mercury_adjoint, Duplicated, Duplicated)]
            fn __mercury_primal($x: &[f64], out: &mut f64) {
                *out = $body;
            }

            /// Evaluates the scalar objective value.
            #[must_use]
            pub fn value($x: &[f64]) -> f64 {
                let mut out = 0.0;
                __mercury_primal($x, &mut out);
                out
            }

            /// Evaluates the dense reverse-mode gradient.
            #[must_use]
            pub fn gradient($x: &[f64]) -> ::std::vec::Vec<f64> {
                value_and_gradient($x).gradient
            }

            /// Evaluates the scalar objective value and dense reverse-mode
            /// gradient together.
            #[must_use]
            pub fn value_and_gradient($x: &[f64]) -> $crate::ValueGradient {
                let mut gradient = ::std::vec![0.0; $x.len()];
                let mut out = 0.0;
                let mut dout = 1.0;

                __mercury_adjoint($x, &mut gradient, &mut out, &mut dout);

                $crate::ValueGradient::new(out, gradient)
            }
        }
    };
}
