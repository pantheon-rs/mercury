//! Linear-algebra error type.

use std::fmt;

/// Error returned by Mercury linear-algebra primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinalgError {
    /// Factorization hit a pivot too small to divide by.
    Singular {
        /// Elimination column where the zero pivot occurred.
        pivot_index: usize,
    },
    /// Operands have incompatible shapes.
    DimensionMismatch {
        /// Rows of the offending operand.
        rows: usize,
        /// Columns of the offending operand.
        cols: usize,
    },
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Singular { pivot_index } => {
                write!(f, "matrix is singular (zero pivot at column {pivot_index})")
            }
            Self::DimensionMismatch { rows, cols } => {
                write!(f, "dimension mismatch: operand is {rows}x{cols}")
            }
        }
    }
}

impl std::error::Error for LinalgError {}
