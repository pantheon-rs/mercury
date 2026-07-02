//! Fixed-size stack matrix — Mercury's kernel-safe matrix type.

use std::ops::{Add, Index, IndexMut, Mul, Sub};

use super::SVector;

/// Fixed-size row-major matrix backed by nested stack arrays
/// (POD-transparent).
///
/// Kernel-safe when constructed via [`SMatrix::new`] or
/// [`SMatrix::from_fn`]. [`SMatrix::zeros`] is host-side only (memset).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SMatrix<const R: usize, const C: usize> {
    data: [[f64; C]; R],
}

impl<const R: usize, const C: usize> SMatrix<R, C> {
    /// Wraps nested row arrays: `new([[row0...], [row1...]])`.
    #[must_use]
    pub const fn new(data: [[f64; C]; R]) -> Self {
        Self { data }
    }

    /// Builds each element from `(row, col)` (kernel-safe; see the Enzyme
    /// IR discipline in the plan's Global Constraints).
    #[must_use]
    #[allow(clippy::needless_range_loop)]
    #[inline(never)]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> f64) -> Self {
        // Construct-then-mutate pattern (Global Constraints rule 4).
        // f is called exactly once per element, in row-major order.
        //
        // Bisected empirically (2026-07-02, scratch kernels in a throwaway
        // test file, deleted after use -- see task-2 report). The brief's
        // suggested shape, seeding via `[[f(0, 0); C]; R]` (nesting
        // SVector::from_fn's proven 1D splat-then-overwrite pattern one
        // level deeper), FAILS: rustc lowers the *outer* repeat as
        // `llvm.memcpy` of the already-built 24-byte row into the other
        // R-1 row slots, and Enzyme's type analysis cannot deduce a type
        // for that copy ("Cannot deduce type of copy ... llvm.memcpy")
        // because its source is a runtime-computed aggregate (the result of
        // calling the closure). This is a genuinely different failure mode
        // from the 1D case: SVector's `[f(0); N]` splats a *scalar*, which
        // LLVM lowers as per-element stores, not a memcpy.
        //
        // Fix has two parts, BOTH required (each alone still fails):
        // 1. Seed with a compile-time CONSTANT instead of a value derived
        //    from `f`, then unconditionally overwrite every element. Since
        //    every element is overwritten via `f` below, `f` still runs
        //    exactly once per element with no observable change in
        //    behavior. Deliberately not `0.0` -- rule 2 (zero-init lowers
        //    to `memset`, which Enzyme also can't type) -- `1.0` is an
        //    arbitrary non-zero placeholder that is never read.
        // 2. `#[inline(never)]` on this fn. With a *single* from_fn call
        //    inlined into a differentiated kernel, LLVM's optimizer fully
        //    scalarizes the constant-seeded array before Enzyme's pass runs
        //    and the fix above is sufficient alone. But this kernel test
        //    calls `from_fn` twice (once for `a`, once for `b`) plus once
        //    more inside the `Mul` impl (for `c`); with multiple identical
        //    `[[1.0; C]; R]` constant seeds inlined into the same function,
        //    LLVM's CSE apparently reintroduces a shared-constant memcpy
        //    that Enzyme still can't type. Keeping `from_fn` a real,
        //    non-inlined call sidesteps that cross-call interaction:
        //    Enzyme differentiates the call interprocedurally either way,
        //    but each invocation's IR is analyzed in the smaller, simpler
        //    context of `from_fn`'s own body, where scalarization is
        //    reliable regardless of how many times it's called from the
        //    differentiated caller.
        //
        // Note the deliberate asymmetry with `SVector::from_fn`, which is
        // `#[inline(always)]`: a scalar splat scalarizes reliably inline,
        // a row-array splat does not. Different lowering, different fix.
        if R == 0 || C == 0 {
            // Zero-size exception: zero bytes, no memset is emitted.
            return Self {
                data: [[0.0; C]; R],
            };
        }
        let mut m = Self {
            data: [[1.0; C]; R],
        };
        for i in 0..R {
            for j in 0..C {
                m.data[i][j] = f(i, j);
            }
        }
        m
    }

    /// All-zeros matrix. Host-side only: zero-init lowers to memset.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            data: [[0.0; C]; R],
        }
    }

    /// Transposed copy.
    #[must_use]
    pub fn transpose(&self) -> SMatrix<C, R> {
        SMatrix::from_fn(|i, j| self.data[j][i])
    }
}

impl<const N: usize> SMatrix<N, N> {
    /// Identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_fn(|i, j| if i == j { 1.0 } else { 0.0 })
    }
}

impl<const R: usize, const C: usize> Index<(usize, usize)> for SMatrix<R, C> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i][j]
    }
}

impl<const R: usize, const C: usize> IndexMut<(usize, usize)> for SMatrix<R, C> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i][j]
    }
}

impl<const R: usize, const C: usize> Add for SMatrix<R, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|i, j| self.data[i][j] + rhs.data[i][j])
    }
}

impl<const R: usize, const C: usize> Sub for SMatrix<R, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_fn(|i, j| self.data[i][j] - rhs.data[i][j])
    }
}

impl<const R: usize, const C: usize> Mul<f64> for SMatrix<R, C> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::from_fn(|i, j| self.data[i][j] * s)
    }
}

impl<const R: usize, const C: usize, const K: usize> Mul<SMatrix<C, K>> for SMatrix<R, C> {
    type Output = SMatrix<R, K>;
    fn mul(self, rhs: SMatrix<C, K>) -> SMatrix<R, K> {
        SMatrix::from_fn(|i, j| {
            let mut acc = 0.0;
            for k in 0..C {
                acc += self.data[i][k] * rhs.data[k][j];
            }
            acc
        })
    }
}

impl<const R: usize, const C: usize> Mul<SVector<C>> for SMatrix<R, C> {
    type Output = SVector<R>;
    fn mul(self, rhs: SVector<C>) -> SVector<R> {
        SVector::from_fn(|i| {
            let mut acc = 0.0;
            for k in 0..C {
                acc += self.data[i][k] * rhs[k];
            }
            acc
        })
    }
}
