//! Heap-backed dynamic vector — hosts problem-scale data OUTSIDE kernels.

use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// Dynamically sized dense vector.
///
/// NOT kernel-safe: heap allocation is not part of the AD-safe subset.
/// Differentiated kernels receive this data via [`Vector::as_slice`].
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    /// Zero vector of length `n`.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    /// Takes ownership of an existing buffer.
    #[must_use]
    pub fn from_vec(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Copies from a slice.
    #[must_use]
    pub fn from_slice(data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector has zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrows elements as a slice (the kernel bridge).
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutably borrows elements as a slice.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Dot product.
    ///
    /// # Panics
    /// On length mismatch.
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> f64 {
        assert_eq!(self.len(), rhs.len(), "dimension mismatch in Vector::dot");
        self.data.iter().zip(&rhs.data).map(|(a, b)| a * b).sum()
    }

    /// Squared Euclidean norm.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.dot(self)
    }
}

impl Index<usize> for Vector {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}

impl Add for &Vector {
    type Output = Vector;
    /// # Panics
    /// On length mismatch.
    fn add(self, rhs: Self) -> Vector {
        assert_eq!(self.len(), rhs.len(), "dimension mismatch in Vector add");
        Vector::from_vec(
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

impl Sub for &Vector {
    type Output = Vector;
    /// # Panics
    /// On length mismatch.
    fn sub(self, rhs: Self) -> Vector {
        assert_eq!(self.len(), rhs.len(), "dimension mismatch in Vector sub");
        Vector::from_vec(
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;
    fn mul(self, s: f64) -> Vector {
        Vector::from_vec(self.data.iter().map(|a| a * s).collect())
    }
}
