//! Errors at numerical and graph boundaries.

use std::fmt;

/// A failed validation, evaluation, or derivative application.
/// See the [example](crate#errors-and-ownership).
#[derive(Debug, PartialEq)]
pub enum Error {
    /// A buffer has the wrong length.
    Dimension {
        /// Buffer being checked.
        buffer: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// Dimension arithmetic overflowed.
    SizeOverflow,
    /// A source refers to a missing input, node, or output.
    InvalidSource,
    /// A node belongs to another builder.
    ForeignNode,
    /// Same-evaluation connections contain a cycle.
    Cycle,
    /// A workspace belongs to another plan.
    ForeignWorkspace,
    /// A failed linearization must be prepared again.
    InvalidLinearization,
    /// A numerical buffer contains NaN or infinity.
    NonFinite(&'static str),
    /// A kernel's documented domain was violated.
    Domain(&'static str),
    /// A solve matrix is singular.
    Singular,
    /// A nonlinear solve did not converge.
    NonConvergence,
    /// An operator failed during plan execution.
    Operator {
        /// Node index in insertion order.
        node: usize,
        /// Original numerical error.
        source: Box<Self>,
    },
}

/// A checked numerical result.
/// See the [example](crate#errors-and-ownership).
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension {
                buffer,
                expected,
                actual,
            } => {
                write!(f, "{buffer}: expected {expected} entries, got {actual}")
            }
            Self::SizeOverflow => f.write_str("dimension arithmetic overflowed"),
            Self::InvalidSource => {
                f.write_str("connection refers to a missing input, node, or output")
            }
            Self::ForeignNode => f.write_str("node belongs to another plan builder"),
            Self::Cycle => f.write_str("same-evaluation connections contain a cycle"),
            Self::ForeignWorkspace => f.write_str("workspace belongs to another plan"),
            Self::InvalidLinearization => f.write_str("linearization is invalid; prepare it again"),
            Self::NonFinite(buffer) => write!(f, "{buffer} contains a non-finite value"),
            Self::Domain(message) => f.write_str(message),
            Self::Singular => f.write_str("solve matrix is singular"),
            Self::NonConvergence => f.write_str("nonlinear solve did not converge"),
            Self::Operator { node, source } => write!(f, "operator {node}: {source}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operator { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

pub const fn check_len(buffer: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::Dimension {
            buffer,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Reject NaN and infinity at a numerical boundary.
///
/// # Errors
/// Returns [`Error::NonFinite`] if any value is not finite.
pub fn check_finite(buffer: &'static str, values: &[f64]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFinite(buffer));
    }
    Ok(())
}
