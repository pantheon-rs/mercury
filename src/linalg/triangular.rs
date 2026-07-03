//! Shared triangular-substitution substrate (faer's `triangular_solve` idea).
//!
//! Every function reads ONLY its named triangle of `m`, so callers may pass
//! combined-storage matrices (e.g. LU's packed `L\U`, LDLT's unit-lower `l`)
//! whose other triangle holds unrelated data.

use crate::core::{Matrix, Vector};

use super::{LinalgError, PIVOT_TOLERANCE};

/// Validates a square system: `m` is `n x n` and `b` has length `n`.
const fn check_square_system(m: &Matrix, b: &Vector) -> Result<usize, LinalgError> {
    let n = m.rows();
    if m.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: m.rows(),
            cols: m.cols(),
        });
    }
    if b.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: b.len(),
            cols: 1,
        });
    }
    Ok(n)
}

/// Divides by the diagonal entry, or errors when it is below tolerance.
fn div_diag(acc: f64, diag: f64, i: usize) -> Result<f64, LinalgError> {
    if diag.abs() < PIVOT_TOLERANCE {
        return Err(LinalgError::Singular { pivot_index: i });
    }
    Ok(acc / diag)
}

/// Solves `L x = b` by forward substitution, reading only the lower
/// triangle of `m` (diagonal implied 1 when `unit_diag`).
pub fn solve_lower(m: &Matrix, b: &Vector, unit_diag: bool) -> Result<Vector, LinalgError> {
    let n = check_square_system(m, b)?;
    let mut x = Vector::zeros(n);
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= m[(i, j)] * x[j];
        }
        x[i] = if unit_diag {
            acc
        } else {
            div_diag(acc, m[(i, i)], i)?
        };
    }
    Ok(x)
}

/// Solves `U x = b` by backward substitution, reading only the upper
/// triangle of `m`.
pub fn solve_upper(m: &Matrix, b: &Vector, unit_diag: bool) -> Result<Vector, LinalgError> {
    let n = check_square_system(m, b)?;
    let mut x = Vector::zeros(n);
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= m[(i, j)] * x[j];
        }
        x[i] = if unit_diag {
            acc
        } else {
            div_diag(acc, m[(i, i)], i)?
        };
    }
    Ok(x)
}

/// Solves `Lᵀ x = b` (an upper-triangular system) by backward substitution,
/// reading only the lower triangle of `m` via transposed indices.
pub fn solve_lower_transposed(
    m: &Matrix,
    b: &Vector,
    unit_diag: bool,
) -> Result<Vector, LinalgError> {
    let n = check_square_system(m, b)?;
    let mut x = Vector::zeros(n);
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= m[(j, i)] * x[j];
        }
        x[i] = if unit_diag {
            acc
        } else {
            div_diag(acc, m[(i, i)], i)?
        };
    }
    Ok(x)
}

/// Solves `Uᵀ x = b` (a lower-triangular system) by forward substitution,
/// reading only the upper triangle of `m` via transposed indices.
pub fn solve_upper_transposed(
    m: &Matrix,
    b: &Vector,
    unit_diag: bool,
) -> Result<Vector, LinalgError> {
    let n = check_square_system(m, b)?;
    let mut x = Vector::zeros(n);
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= m[(j, i)] * x[j];
        }
        x[i] = if unit_diag {
            acc
        } else {
            div_diag(acc, m[(i, i)], i)?
        };
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower() -> Matrix {
        // L = [[2, *], [1, 3]] — upper triangle poisoned to prove it is
        // never read.
        Matrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (1, 0) => 1.0,
            (1, 1) => 3.0,
            _ => f64::NAN, // poison
        })
    }

    fn upper() -> Matrix {
        // U = [[2, 1], [*, 4]] — lower triangle poisoned.
        Matrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (0, 1) => 1.0,
            (1, 1) => 4.0,
            _ => f64::NAN, // poison
        })
    }

    #[test]
    fn solve_lower_non_unit() {
        // 2x0 = 4; 1*x0 + 3*x1 = 8 => x = [2, 2]
        let b = Vector::from_slice(&[4.0, 8.0]);
        let x = solve_lower(&lower(), &b, false).expect("solve");
        assert!((x[0] - 2.0).abs() < 1e-14);
        assert!((x[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn solve_lower_unit_ignores_diagonal() {
        // Unit diagonal: x0 = 4; 1*x0 + x1 = 8 => x = [4, 4].
        // Diagonal values 2 and 3 must be ignored.
        let b = Vector::from_slice(&[4.0, 8.0]);
        let x = solve_lower(&lower(), &b, true).expect("solve");
        assert!((x[0] - 4.0).abs() < 1e-14);
        assert!((x[1] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn solve_upper_non_unit() {
        // 2*x0 + 1*x1 = 8; 4*x1 = 8 => x = [3, 2]
        let b = Vector::from_slice(&[8.0, 8.0]);
        let x = solve_upper(&upper(), &b, false).expect("solve");
        assert!((x[0] - 3.0).abs() < 1e-14);
        assert!((x[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn solve_lower_transposed_matches_explicit_transpose() {
        // Lᵀ = [[2, 1], [0, 3]]: 2*x0 + 1*x1 = 8; 3*x1 = 6 => x = [3, 2]
        let b = Vector::from_slice(&[8.0, 6.0]);
        let x = solve_lower_transposed(&lower(), &b, false).expect("solve");
        assert!((x[0] - 3.0).abs() < 1e-14);
        assert!((x[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn solve_upper_transposed_matches_explicit_transpose() {
        // Uᵀ = [[2, 0], [1, 4]]: 2*x0 = 4; 1*x0 + 4*x1 = 10 => x = [2, 2]
        let b = Vector::from_slice(&[4.0, 10.0]);
        let x = solve_upper_transposed(&upper(), &b, false).expect("solve");
        assert!((x[0] - 2.0).abs() < 1e-14);
        assert!((x[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn zero_diagonal_errors_singular() {
        let m = Matrix::from_fn(2, 2, |_, _| 0.0);
        let b = Vector::zeros(2);
        assert_eq!(
            solve_lower(&m, &b, false),
            Err(LinalgError::Singular { pivot_index: 0 })
        );
        // Unit-diagonal variant never divides, so it succeeds.
        assert!(solve_lower(&m, &b, true).is_ok());
    }

    #[test]
    fn dimension_mismatches_error() {
        let m = Matrix::zeros(2, 3); // non-square
        let b = Vector::zeros(2);
        assert!(solve_lower(&m, &b, false).is_err());
        let sq = Matrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let short = Vector::zeros(1);
        assert!(solve_upper(&sq, &short, false).is_err());
    }

    #[test]
    fn solve_upper_unit_ignores_diagonal() {
        // Unit diagonal: x1 = 8; x0 + 1*x1 = 8 => x0 = 0. Diagonal values
        // 2 and 4 must be ignored.
        let b = Vector::from_slice(&[8.0, 8.0]);
        let x = solve_upper(&upper(), &b, true).expect("solve");
        assert!((x[1] - 8.0).abs() < 1e-14);
        assert!((x[0] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn solve_lower_transposed_unit_diag() {
        // Unit-diagonal Lᵀ = [[1, 1], [0, 1]]: x1 = 6; x0 + x1 = 8 => x0 = 2.
        let b = Vector::from_slice(&[8.0, 6.0]);
        let x = solve_lower_transposed(&lower(), &b, true).expect("solve");
        assert!((x[0] - 2.0).abs() < 1e-14);
        assert!((x[1] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn solve_upper_transposed_unit_diag() {
        // Unit-diagonal Uᵀ = [[1, 0], [1, 1]]: x0 = 4; x0 + x1 = 10 => x1 = 6.
        let b = Vector::from_slice(&[4.0, 10.0]);
        let x = solve_upper_transposed(&upper(), &b, true).expect("solve");
        assert!((x[0] - 4.0).abs() < 1e-14);
        assert!((x[1] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn zero_diagonal_errors_singular_for_every_variant() {
        // Only solve_lower's error path is exercised above; check the other
        // three non-unit-diag variants also surface Singular, each at the
        // pivot where the zero diagonal is first read.
        let m = Matrix::from_fn(2, 2, |_, _| 0.0);
        let b = Vector::zeros(2);
        assert_eq!(
            solve_upper(&m, &b, false),
            Err(LinalgError::Singular { pivot_index: 1 })
        );
        assert_eq!(
            solve_lower_transposed(&m, &b, false),
            Err(LinalgError::Singular { pivot_index: 1 })
        );
        assert_eq!(
            solve_upper_transposed(&m, &b, false),
            Err(LinalgError::Singular { pivot_index: 0 })
        );
    }

    #[test]
    fn pivot_at_tolerance_boundary() {
        // |diag| == PIVOT_TOLERANCE is NOT below tolerance, so this must
        // still succeed (the check is strict `<`).
        let m = Matrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => PIVOT_TOLERANCE,
            (1, 0) => 0.0,
            (1, 1) => 1.0,
            _ => f64::NAN,
        });
        let b = Vector::from_slice(&[PIVOT_TOLERANCE, 2.0]);
        let x = solve_lower(&m, &b, false).expect("boundary pivot should solve");
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn three_by_three_round_trip_matches_forward_and_back() {
        // A slightly larger system exercises the inner accumulation loops
        // (j in 0..i / i+1..n) beyond the 2x2 minimal cases above.
        let l = Matrix::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (1, 0) => 1.0,
            (1, 1) => 3.0,
            (2, 0) => -1.0,
            (2, 1) => 0.5,
            (2, 2) => 4.0,
            _ => f64::NAN, // poison strict upper triangle
        });
        let b = Vector::from_slice(&[4.0, 11.0, 9.0]);
        let x = solve_lower(&l, &b, false).expect("solve");
        // Reconstruct L x and compare against b directly (no hand-solved
        // closed form needed).
        for i in 0..3 {
            let mut acc = 0.0;
            for j in 0..=i {
                acc += l[(i, j)] * x[j];
            }
            assert!((acc - b[i]).abs() < 1e-12, "row {i}: {acc} vs {}", b[i]);
        }
    }
}
