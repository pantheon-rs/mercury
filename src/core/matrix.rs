//! Heap-backed dynamic dense matrix (row-major) — hosts problem-scale data
//! OUTSIDE kernels.

use std::ops::{Index, IndexMut, Mul};

use super::Vector;

/// Dynamically sized dense row-major matrix.
///
/// NOT kernel-safe: heap allocation is not part of the AD-safe subset.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Zero matrix of shape `rows x cols`.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Builds each element from `(row, col)`.
    #[must_use]
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f64) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i * cols + j] = f(i, j);
            }
        }
        m
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Transposed copy.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |i, j| self[(j, i)])
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        let cols = self.cols;
        &mut self.data[i * cols + j]
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Vector;
    /// # Panics
    /// When `self.cols() != rhs.len()`.
    fn mul(self, rhs: &Vector) -> Vector {
        assert_eq!(self.cols, rhs.len(), "dimension mismatch in Matrix * Vector");
        Vector::from_vec(
            (0..self.rows)
                .map(|i| (0..self.cols).map(|k| self[(i, k)] * rhs[k]).sum())
                .collect(),
        )
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    /// # Panics
    /// When `self.cols() != rhs.rows()`.
    fn mul(self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.rows, "dimension mismatch in Matrix * Matrix");
        Matrix::from_fn(self.rows, rhs.cols, |i, j| {
            (0..self.cols).map(|k| self[(i, k)] * rhs[(k, j)]).sum()
        })
    }
}
