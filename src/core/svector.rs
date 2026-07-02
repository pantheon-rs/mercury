//! Fixed-size stack vector — Mercury's kernel-safe vector type.

use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

/// Fixed-size column vector backed by a stack array (POD-transparent).
///
/// This type is safe to construct and use inside Enzyme-differentiated
/// kernels **via [`SVector::new`] or [`SVector::from_fn`]** (element-wise
/// stores). [`SVector::zeros`] uses array zero-init, which lowers to a
/// memset Enzyme cannot type — treat it as host-side only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SVector<const N: usize> {
    data: [f64; N],
}

impl<const N: usize> SVector<N> {
    /// Wraps an existing array.
    #[must_use]
    pub const fn new(data: [f64; N]) -> Self {
        Self { data }
    }

    /// Builds each element from its index (element-wise stores; kernel-safe).
    #[must_use]
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    pub fn from_fn(mut f: impl FnMut(usize) -> f64) -> Self {
        // Enzyme constraints, empirically pinned (2026-07-02 bisection):
        // - no `std::array::from_fn` (MaybeUninit machinery -> untyped memcpy)
        // - no `[0.0; N]` zero-init on kernel paths (memset); the N == 0
        //   early return below is the one exception (zero bytes, no memset)
        // - no iterator adapters in kernel-reachable loops (untypeable copy);
        //   plain `for i in 0..N` indexed loops only
        // - construct the aggregate in ONE expression, then mutate through
        //   the struct binding; `let arr = ...; Self { data: arr }` after a
        //   mutation loop is an aggregate rvalue that cannot merge into the
        //   return slot and leaves a memcpy Enzyme rejects.
        if N == 0 {
            return Self { data: [0.0; N] };
        }
        let mut v = Self { data: [f(0); N] };
        for i in 1..N {
            v.data[i] = f(i);
        }
        v
    }

    /// All-zeros vector. Host-side only: zero-init lowers to memset.
    #[must_use]
    pub const fn zeros() -> Self {
        Self { data: [0.0; N] }
    }

    /// Borrows the elements as a slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Dot product.
    #[must_use]
    #[inline(always)]
    pub fn dot(&self, rhs: &Self) -> f64 {
        let mut acc = 0.0;
        for i in 0..N {
            acc += self.data[i] * rhs.data[i];
        }
        acc
    }

    /// Squared Euclidean norm.
    #[must_use]
    #[inline(always)]
    pub fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Euclidean norm. Not differentiable at the origin (sqrt kink).
    #[must_use]
    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

impl SVector<3> {
    /// Cross product (3-vectors only).
    #[must_use]
    #[inline(always)]
    pub fn cross(&self, rhs: &Self) -> Self {
        Self::new([
            self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1],
            self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2],
            self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0],
        ])
    }
}

impl<const N: usize> Index<usize> for SVector<N> {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl<const N: usize> IndexMut<usize> for SVector<N> {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}

impl<const N: usize> Add for SVector<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.data[i] + rhs.data[i])
    }
}

impl<const N: usize> Sub for SVector<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.data[i] - rhs.data[i])
    }
}

impl<const N: usize> Neg for SVector<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_fn(|i| -self.data[i])
    }
}

impl<const N: usize> Mul<f64> for SVector<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f64) -> Self {
        Self::from_fn(|i| self.data[i] * s)
    }
}

impl<const N: usize> Div<f64> for SVector<N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f64) -> Self {
        Self::from_fn(|i| self.data[i] / s)
    }
}
