//! Row-permutation type — replaces raw `Vec<usize>` bookkeeping (faer's
//! `Perm` idea, f64-only and without generics).

use super::Vector;

/// A permutation of `0..n`, tracked together with its transposition parity.
///
/// Convention (matching `LuFactors`): `perm[i]` is the original index that
/// occupies position `i` after permuting, so `(Pv)[i] = v[perm[i]]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Perm {
    perm: Vec<usize>,
    odd: bool,
}

impl Perm {
    /// The identity permutation on `0..n`.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            perm: (0..n).collect(),
            odd: false,
        }
    }

    /// Number of elements permuted.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.perm.len()
    }

    /// Whether the permutation is over zero elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// Swaps positions `i` and `j`, flipping parity when `i != j`.
    ///
    /// # Panics
    /// When `i` or `j` is out of range.
    pub fn swap(&mut self, i: usize, j: usize) {
        if i != j {
            self.perm.swap(i, j);
            self.odd = !self.odd;
        }
    }

    /// Applies the permutation: `out[i] = v[perm[i]]`.
    ///
    /// # Panics
    /// On length mismatch.
    #[must_use]
    pub fn apply(&self, v: &Vector) -> Vector {
        let n = self.len();
        assert_eq!(v.len(), n, "dimension mismatch in Perm::apply");
        let mut out = Vector::zeros(n);
        for i in 0..n {
            out[i] = v[self.perm[i]];
        }
        out
    }

    /// Applies the inverse permutation: `out[perm[i]] = v[i]`.
    ///
    /// # Panics
    /// On length mismatch.
    #[must_use]
    pub fn apply_inverse(&self, v: &Vector) -> Vector {
        let n = self.len();
        assert_eq!(v.len(), n, "dimension mismatch in Perm::apply_inverse");
        let mut out = Vector::zeros(n);
        for i in 0..n {
            out[self.perm[i]] = v[i];
        }
        out
    }

    /// Sign of the permutation: `+1.0` for even, `-1.0` for odd.
    #[must_use]
    pub const fn sign(&self) -> f64 {
        if self.odd { -1.0 } else { 1.0 }
    }
}
