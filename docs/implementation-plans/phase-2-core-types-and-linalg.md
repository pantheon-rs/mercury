# Phase 2: Core Types + Linalg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mercury's POD-transparent math types (`SVector`, `SMatrix`, `Vector`, `Matrix`, `Quaternion`) plus the first owned-derivative-rule primitive: linear solve with the adjoint rule.

**Architecture:** Fixed-size stack types are the kernel vocabulary and must pass Enzyme (proven per-type by compile+gradient tests). Dynamic types host problem-scale data *outside* kernels and bridge in via slices. `linalg` delivers the thesis demo: a kernel-safe fixed-size solve Enzyme can differentiate through, a dynamic LU primitive, and the adjoint rule (`solve_vjp`/`solve_jvp`) — validated by a three-way gradient agreement test (finite differences vs Enzyme vs adjoint rule).

**Tech Stack:** Rust nightly (pinned Enzyme toolchain via `nix develop`), `std::autodiff`, zero dependencies.

## Global Constraints

- Edition 2024, `#![forbid(unsafe_code)]`, zero `[dependencies]`.
- `missing_docs = "warn"` — every public item gets a doc comment.
- Clippy `pedantic` + `nursery` are warn — keep code clean against them.
- All types f64-only. POD-transparency law (decision 0003): plain contiguous `f64` storage, no hidden allocation on differentiated paths, no `dyn`, no generic scalar.
- Tests ALWAYS run `--release` (Enzyme requires fat LTO; debug is unsupported). `./scripts/test.sh` handles this and auto-enters the Nix shell.
- Three-legged test law: every primitive ships (1) an Enzyme compile+gradient test, (2) a finite-difference cross-check via `mercury::validation`, (3) an analytic check where closed form exists.
- Enzyme IR discipline in anything kernel-reachable: construct arrays with `core::array::from_fn` / element-wise stores. No `[0.0; N]` zero-init, no bulk copies/`mem::swap` of arrays (memset/memcpy kill Enzyme type analysis — see `metis-ad-spike/linalg_compat/RESULTS.md`).
- Commit after every task. No pre-commit hooks are installed; still verify `git status` scope before committing.
- Gradient tolerances: Enzyme vs analytic `max_abs_error < 1e-9`; Enzyme vs finite differences `max_abs_error < 1e-4` (matches Phase 1 conventions in `tests/objective.rs`).

**Manual checkpoint convention:** after each task, run `./scripts/test.sh` yourself. Expected: all suites pass, with the new test file listed. Targeted single-suite run: `nix develop "path:$PWD" --command cargo test --release --test <file-stem>`.

---

### Task 0: Commit the Phase 1 baseline

The working tree holds the complete, green Phase 1 slice (objective macro + validation), uncommitted. Phase 2 tasks need a clean baseline to commit against.

**Files:**
- No new files. Commits existing working tree.

- [ ] **Step 1: Verify Phase 1 is green**

Run: `./scripts/test.sh`
Expected: `test result: ok` for `objective` and `validation` suites (and doc-tests). If anything fails, STOP — fix Phase 1 before proceeding.

- [ ] **Step 2: Review scope, then commit everything as the Phase 1 slice**

Run: `git status --short` — expect only Phase-1-related files (src/objective.rs, src/validation.rs, tests/, scripts/, flake/nix, Cargo.toml, README.md, docs/, deleted tests/public_api.rs).

```bash
git add -A
git commit -m "feat: phase 1 gradient validation slice

scalar_objective! macro (Enzyme reverse behind generated module),
ValueGradient, central-difference validation helpers, pinned Enzyme
dev tooling."
```

**Manual checkpoint 0:** `git log --oneline -3` shows the baseline commit on `phase1`; `git status` is clean.

---

### Task 1: `SVector<N>` — fixed-size vector

**Files:**
- Create: `src/core/mod.rs`
- Create: `src/core/svector.rs`
- Modify: `src/lib.rs` (add `pub mod core;`, re-export)
- Modify: `src/validation.rs:3` (`use core::fmt;` → `use std::fmt;` — a crate-root module named `core` makes `use core::...` ambiguous, E0659)
- Test: `tests/core_svector.rs`

**Interfaces:**
- Consumes: `mercury::validation::{central_difference_gradient, compare_gradients}` (Phase 1).
- Produces: `SVector<const N: usize>` with `new([f64; N])`, `from_fn(impl Fn(usize) -> f64)`, `zeros()`, `as_slice() -> &[f64]`, `dot(&self, &Self) -> f64`, `norm_squared() -> f64`, `norm() -> f64`, `cross(&self, &Self) -> Self` (N=3 only), `Index<usize>`/`IndexMut`, `Add`, `Sub`, `Neg`, `Mul<f64>`, `Div<f64>`; `Copy`, `Clone`, `Debug`, `PartialEq`. Re-exported as `mercury::SVector`.

- [ ] **Step 1: Write the failing test**

Create `tests/core_svector.rs`:

```rust
#![feature(autodiff)]

//! SVector unit tests + Enzyme kernel-safety test (three-legged law).

use mercury::SVector;
use mercury::validation::{central_difference_gradient, compare_gradients};
use std::autodiff::autodiff_reverse;

#[test]
fn constructors_and_indexing() {
    let v = SVector::new([1.0, 2.0, 3.0]);
    let w = SVector::<3>::from_fn(|i| (i as f64) + 1.0);
    assert_eq!(v, w);
    assert_eq!(v[2], 3.0);
    assert_eq!(SVector::<4>::zeros().as_slice(), &[0.0; 4]);

    let mut m = v;
    m[0] = 9.0;
    assert_eq!(m[0], 9.0);
}

#[test]
fn arithmetic_ops() {
    let v = SVector::new([1.0, 2.0, 3.0]);
    let w = SVector::new([4.0, 5.0, 6.0]);
    assert_eq!((v + w).as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!((w - v).as_slice(), &[3.0, 3.0, 3.0]);
    assert_eq!((-v).as_slice(), &[-1.0, -2.0, -3.0]);
    assert_eq!((v * 2.0).as_slice(), &[2.0, 4.0, 6.0]);
    assert_eq!((w / 2.0).as_slice(), &[2.0, 2.5, 3.0]);
}

#[test]
fn dot_norm_cross() {
    let v = SVector::new([1.0, 2.0, 3.0]);
    let w = SVector::new([4.0, 5.0, 6.0]);
    assert!((v.dot(&w) - 32.0).abs() < 1e-15);
    assert!((v.norm_squared() - 14.0).abs() < 1e-15);
    assert!((v.norm() - 14.0_f64.sqrt()).abs() < 1e-15);
    // e1 x e2 = e3
    let e1 = SVector::new([1.0, 0.0, 0.0]);
    let e2 = SVector::new([0.0, 1.0, 0.0]);
    assert_eq!(e1.cross(&e2).as_slice(), &[0.0, 0.0, 1.0]);
}

// --- Enzyme leg: differentiate a kernel built from SVector ops ---

#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let v = SVector::<3>::from_fn(|i| x[i]);
    let w = SVector::<3>::from_fn(|i| x[3 + i]);
    let c = v.cross(&w);
    *out = v.dot(&w) + c.norm_squared();
}

fn kernel_value(x: &[f64]) -> f64 {
    let mut out = 0.0;
    kernel(x, &mut out);
    out
}

#[test]
fn enzyme_gradient_matches_finite_differences() {
    let x = [0.7, -1.3, 2.1, 0.4, 1.9, -0.6];
    let mut grad = vec![0.0; 6];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6)
        .expect("finite-difference gradient should compute");
    let check = compare_gradients(&grad, &fd).expect("same shape");
    assert!(check.max_abs_error < 1.0e-4, "{check:?}\n ad={grad:?}\n fd={fd:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop "path:$PWD" --command cargo test --release --test core_svector`
Expected: COMPILE ERROR — `unresolved import mercury::SVector`.

- [ ] **Step 3: Implement**

Create `src/core/svector.rs`:

```rust
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
    pub fn from_fn(f: impl FnMut(usize) -> f64) -> Self {
        Self { data: std::array::from_fn(f) }
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
    pub fn dot(&self, rhs: &Self) -> f64 {
        let mut acc = 0.0;
        for i in 0..N {
            acc += self.data[i] * rhs.data[i];
        }
        acc
    }

    /// Squared Euclidean norm.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Euclidean norm. Not differentiable at the origin (sqrt kink).
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

impl SVector<3> {
    /// Cross product (3-vectors only).
    #[must_use]
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
    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.data[i] + rhs.data[i])
    }
}

impl<const N: usize> Sub for SVector<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.data[i] - rhs.data[i])
    }
}

impl<const N: usize> Neg for SVector<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_fn(|i| -self.data[i])
    }
}

impl<const N: usize> Mul<f64> for SVector<N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::from_fn(|i| self.data[i] * s)
    }
}

impl<const N: usize> Div<f64> for SVector<N> {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self::from_fn(|i| self.data[i] / s)
    }
}
```

Create `src/core/mod.rs`:

```rust
//! Mercury-owned core math types (POD-transparency law, decision 0003).

mod svector;

pub use svector::SVector;
```

In `src/lib.rs` add after `mod objective;`:

```rust
pub mod core;

pub use crate::core::SVector;
```

In `src/validation.rs` change line 3 from `use core::fmt;` to `use std::fmt;`.

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop "path:$PWD" --command cargo test --release --test core_svector`
Expected: 4 tests PASS (including `enzyme_gradient_matches_finite_differences`).

- [ ] **Step 5: Commit**

```bash
git add src/core/ src/lib.rs src/validation.rs tests/core_svector.rs
git commit -m "feat(core): SVector fixed-size vector with Enzyme kernel test"
```

**Manual checkpoint 1:** run `./scripts/test.sh` — all suites green. The Enzyme test in `core_svector` IS the proof that mercury-owned types pass where nalgebra's failed: same toolchain, same kernel shape, no memcpy error.

---

### Task 2: `SMatrix<R, C>` — fixed-size matrix

**Files:**
- Create: `src/core/smatrix.rs`
- Modify: `src/core/mod.rs`, `src/lib.rs` (exports)
- Test: `tests/core_smatrix.rs`

**Interfaces:**
- Consumes: `SVector<N>` from Task 1.
- Produces: `SMatrix<const R: usize, const C: usize>` with `new([[f64; C]; R])` (row-major rows), `from_fn(impl FnMut(usize, usize) -> f64)`, `zeros()`, `transpose() -> SMatrix<C, R>`, `Index<(usize, usize)>`/`IndexMut`, `Add`, `Sub`, `Mul<f64>`, `Mul<SMatrix<C, K>> -> SMatrix<R, K>`, `Mul<SVector<C>> -> SVector<R>`; square-only `identity()` on `SMatrix<N, N>`. Re-exported as `mercury::SMatrix`.

- [ ] **Step 1: Write the failing test**

Create `tests/core_smatrix.rs`:

```rust
#![feature(autodiff)]

//! SMatrix unit tests + Enzyme kernel test (the linalg_compat control kernel
//! rewritten with Mercury types, plus an analytic oracle).

use mercury::{SMatrix, SVector};
use mercury::validation::{central_difference_gradient, compare_gradients};
use std::autodiff::autodiff_reverse;

#[test]
fn constructors_indexing_identity() {
    let a = SMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(a[(0, 1)], 2.0);
    assert_eq!(a, SMatrix::<2, 2>::from_fn(|i, j| (2 * i + j + 1) as f64));

    let eye = SMatrix::<3, 3>::identity();
    assert_eq!(eye[(1, 1)], 1.0);
    assert_eq!(eye[(1, 2)], 0.0);
    assert_eq!(SMatrix::<2, 3>::zeros()[(1, 2)], 0.0);
}

#[test]
fn matmul_matvec_transpose() {
    let a = SMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = SMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let c = a * b;
    assert_eq!(c, SMatrix::new([[19.0, 22.0], [43.0, 50.0]]));

    let v = SVector::new([1.0, 1.0]);
    assert_eq!((a * v).as_slice(), &[3.0, 7.0]);

    let t = a.transpose();
    assert_eq!(t, SMatrix::new([[1.0, 3.0], [2.0, 4.0]]));

    assert_eq!((a + b), SMatrix::new([[6.0, 8.0], [10.0, 12.0]]));
    assert_eq!((b - a), SMatrix::new([[4.0, 4.0], [4.0, 4.0]]));
    assert_eq!((a * 2.0), SMatrix::new([[2.0, 4.0], [6.0, 8.0]]));
}

// --- Enzyme leg: f(x) = sum((A*B) elementwise squared), A,B from x ---
// Same kernel as linalg_compat's control/na_fixed_new; analytic gradient:
// dF/dA = 2 (A B) B^T,  dF/dB = 2 A^T (A B).

#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| x[3 * i + j]);
    let b = SMatrix::<3, 3>::from_fn(|i, j| x[9 + 3 * i + j]);
    let c = a * b;
    let mut acc = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            acc += c[(i, j)] * c[(i, j)];
        }
    }
    *out = acc;
}

fn kernel_value(x: &[f64]) -> f64 {
    let mut out = 0.0;
    kernel(x, &mut out);
    out
}

#[test]
fn enzyme_gradient_matches_fd_and_analytic() {
    let x: Vec<f64> = (0..18).map(|i| 0.3 + 0.1 * (i as f64)).collect();
    let mut grad = vec![0.0; 18];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    // Finite differences.
    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6).expect("fd");
    let fd_check = compare_gradients(&grad, &fd).expect("shape");
    assert!(fd_check.max_abs_error < 1.0e-4, "{fd_check:?}");

    // Analytic: dF/dA = 2 C B^T, dF/dB = 2 A^T C with C = A B.
    let a = SMatrix::<3, 3>::from_fn(|i, j| x[3 * i + j]);
    let b = SMatrix::<3, 3>::from_fn(|i, j| x[9 + 3 * i + j]);
    let c = a * b;
    let da = (c * b.transpose()) * 2.0;
    let db = (a.transpose() * c) * 2.0;
    let mut analytic = vec![0.0; 18];
    for i in 0..3 {
        for j in 0..3 {
            analytic[3 * i + j] = da[(i, j)];
            analytic[9 + 3 * i + j] = db[(i, j)];
        }
    }
    let an_check = compare_gradients(&grad, &analytic).expect("shape");
    assert!(an_check.max_abs_error < 1.0e-9, "{an_check:?}\n ad={grad:?}\n an={analytic:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop "path:$PWD" --command cargo test --release --test core_smatrix`
Expected: COMPILE ERROR — `unresolved import mercury::SMatrix`.

- [ ] **Step 3: Implement**

Create `src/core/smatrix.rs`:

```rust
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

    /// Builds each element from `(row, col)` (element-wise stores;
    /// kernel-safe).
    #[must_use]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> f64) -> Self {
        Self { data: std::array::from_fn(|i| std::array::from_fn(|j| f(i, j))) }
    }

    /// All-zeros matrix. Host-side only: zero-init lowers to memset.
    #[must_use]
    pub const fn zeros() -> Self {
        Self { data: [[0.0; C]; R] }
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
```

In `src/core/mod.rs` add `mod smatrix;` and `pub use smatrix::SMatrix;`.
In `src/lib.rs` change the re-export to `pub use crate::core::{SMatrix, SVector};`.

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop "path:$PWD" --command cargo test --release --test core_smatrix`
Expected: 3 tests PASS. The analytic leg passing at `< 1e-9` proves Enzyme is producing exact derivatives through mercury matmul, not approximations.

- [ ] **Step 5: Commit**

```bash
git add src/core/ src/lib.rs tests/core_smatrix.rs
git commit -m "feat(core): SMatrix fixed-size matrix with Enzyme + analytic tests"
```

**Manual checkpoint 2:** `./scripts/test.sh` green. This is nalgebra's `na_fixed`/`na_fixed_new` scenario with zero construction restrictions — any constructor works because we control the IR.

---

### Task 3: `Quaternion`

**Files:**
- Create: `src/geometry/mod.rs`
- Create: `src/geometry/quaternion.rs`
- Modify: `src/lib.rs` (add `pub mod geometry;`, re-export `Quaternion`)
- Test: `tests/geometry_quaternion.rs`

**Interfaces:**
- Consumes: `SVector<3>`, `SMatrix<3, 3>`.
- Produces: `Quaternion` (`pub w, x, y, z: f64`, scalar-first Hamilton convention) with `new(w, x, y, z)`, `identity()`, `from_axis_angle(axis: &SVector<3>, angle: f64)` (axis must be unit; documented), `conjugate()`, `norm()`, `normalized()`, `rotate(&self, v: &SVector<3>) -> SVector<3>` (assumes unit quaternion), `to_dcm(&self) -> SMatrix<3, 3>`, `Mul` (Hamilton product). Re-exported as `mercury::Quaternion`.

- [ ] **Step 1: Write the failing test**

Create `tests/geometry_quaternion.rs`:

```rust
#![feature(autodiff)]

//! Quaternion unit tests + Enzyme rotation-kernel test.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{Quaternion, SVector};
use std::autodiff::autodiff_reverse;
use std::f64::consts::FRAC_PI_2;

#[test]
fn identity_and_hamilton_product() {
    let q = Quaternion::new(0.5, -0.3, 0.8, 0.1).normalized();
    let i = Quaternion::identity();
    let qi = q * i;
    assert!((qi.w - q.w).abs() < 1e-15 && (qi.x - q.x).abs() < 1e-15);

    // i * j = k in Hamilton convention.
    let qi_ = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let qj = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    let qk = qi_ * qj;
    assert!((qk.z - 1.0).abs() < 1e-15 && qk.w.abs() < 1e-15);
}

#[test]
fn rotation_and_dcm_agree() {
    // 90 degrees about z: e1 -> e2.
    let axis = SVector::new([0.0, 0.0, 1.0]);
    let q = Quaternion::from_axis_angle(&axis, FRAC_PI_2);
    let e1 = SVector::new([1.0, 0.0, 0.0]);

    let r = q.rotate(&e1);
    assert!((r[0]).abs() < 1e-12 && (r[1] - 1.0).abs() < 1e-12 && r[2].abs() < 1e-12);

    let dcm = q.to_dcm();
    let rd = dcm * e1;
    assert!((rd[0] - r[0]).abs() < 1e-12 && (rd[1] - r[1]).abs() < 1e-12);

    // Rotation preserves length.
    let v = SVector::new([0.3, -1.2, 2.2]);
    assert!((q.rotate(&v).norm() - v.norm()).abs() < 1e-12);

    // conjugate rotates back.
    let back = q.conjugate().rotate(&r);
    assert!((back[0] - 1.0).abs() < 1e-12 && back[1].abs() < 1e-12);
}

// --- Enzyme leg: rotate a vector by a normalized quaternion, sum squares.
// x[0..4] = raw quaternion (normalized in-kernel), x[4..7] = vector.

#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let q = Quaternion::new(x[0], x[1], x[2], x[3]).normalized();
    let v = SVector::new([x[4], x[5], x[6]]);
    let r = q.rotate(&v);
    *out = r.norm_squared() + r[0] * r[1];
}

fn kernel_value(x: &[f64]) -> f64 {
    let mut out = 0.0;
    kernel(x, &mut out);
    out
}

#[test]
fn enzyme_gradient_matches_finite_differences() {
    let x = [0.9, 0.2, -0.3, 0.1, 0.5, -1.1, 0.8];
    let mut grad = vec![0.0; 7];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6).expect("fd");
    let check = compare_gradients(&grad, &fd).expect("shape");
    assert!(check.max_abs_error < 1.0e-4, "{check:?}\n ad={grad:?}\n fd={fd:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop "path:$PWD" --command cargo test --release --test geometry_quaternion`
Expected: COMPILE ERROR — `unresolved import mercury::Quaternion`.

- [ ] **Step 3: Implement**

Create `src/geometry/quaternion.rs`:

```rust
//! Unit quaternion for attitude representation (scalar-first, Hamilton).

use std::ops::Mul;

use crate::core::{SMatrix, SVector};

/// Quaternion `w + xi + yj + zk` (scalar-first, Hamilton convention).
///
/// Rotation methods ([`Quaternion::rotate`], [`Quaternion::to_dcm`]) assume
/// a unit quaternion; call [`Quaternion::normalized`] first when in doubt.
/// All operations are analytic and kernel-safe; `normalized` inherits the
/// sqrt kink at zero norm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    /// Scalar part.
    pub w: f64,
    /// First vector component (i).
    pub x: f64,
    /// Second vector component (j).
    pub y: f64,
    /// Third vector component (k).
    pub z: f64,
}

impl Quaternion {
    /// Builds a quaternion from components (scalar first).
    #[must_use]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Identity rotation.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Rotation of `angle` radians about a **unit** `axis`.
    #[must_use]
    pub fn from_axis_angle(axis: &SVector<3>, angle: f64) -> Self {
        let (s, c) = (angle / 2.0).sin_cos();
        Self::new(c, s * axis[0], s * axis[1], s * axis[2])
    }

    /// Conjugate (inverse for unit quaternions).
    #[must_use]
    pub const fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Euclidean norm of the 4-tuple.
    #[must_use]
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Unit-norm copy. Kink at zero norm (division by `norm()`).
    #[must_use]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        Self::new(self.w / n, self.x / n, self.y / n, self.z / n)
    }

    /// Rotates a vector: `q v q*` for unit `q`, via the two-cross expansion
    /// `v + w t + qv x t` with `t = 2 qv x v`.
    #[must_use]
    pub fn rotate(&self, v: &SVector<3>) -> SVector<3> {
        let qv = SVector::new([self.x, self.y, self.z]);
        let t = qv.cross(v) * 2.0;
        *v + t * self.w + qv.cross(&t)
    }

    /// Direction-cosine (rotation) matrix equivalent, for unit `q`.
    #[must_use]
    pub fn to_dcm(&self) -> SMatrix<3, 3> {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        SMatrix::new([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ])
    }
}

impl Mul for Quaternion {
    type Output = Self;
    /// Hamilton product `self * rhs` (applies `rhs` first, then `self`).
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        )
    }
}
```

Create `src/geometry/mod.rs`:

```rust
//! Rotation and attitude types with analytic derivatives.

mod quaternion;

pub use quaternion::Quaternion;
```

In `src/lib.rs` add `pub mod geometry;` and extend re-exports:
`pub use crate::geometry::Quaternion;`

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop "path:$PWD" --command cargo test --release --test geometry_quaternion`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/geometry/ src/lib.rs tests/geometry_quaternion.rs
git commit -m "feat(geometry): Quaternion with rotation kernel Enzyme test"
```

**Manual checkpoint 3:** `./scripts/test.sh` green. The Enzyme kernel here differentiates through `normalized()` (sqrt + divides) and two cross products — a real attitude-dynamics-shaped kernel.

---

### Task 4: Dynamic `Vector` and `Matrix` (host-side)

These host problem-scale data *outside* kernels. They are NOT promised kernel-safe (heap allocation risks the same overflow-check intrinsic that killed faer); kernels receive their data via `as_slice()`. Document that contract loudly.

**Files:**
- Create: `src/core/vector.rs`
- Create: `src/core/matrix.rs`
- Modify: `src/core/mod.rs`, `src/lib.rs` (exports)
- Test: `tests/core_dynamic.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Vector`: `zeros(n)`, `from_vec(Vec<f64>)`, `from_slice(&[f64])`, `len()`, `is_empty()`, `as_slice()`, `as_mut_slice()`, `dot(&self, &Self) -> f64` (panics on length mismatch; documented), `norm_squared()`, `Index<usize>`/`IndexMut`, `Clone`, `Debug`, `PartialEq`, `Add<&Vector> for &Vector`, `Sub`, `Mul<f64> for &Vector`.
  - `Matrix` (row-major): `zeros(rows, cols)`, `from_fn(rows, cols, impl FnMut(usize, usize) -> f64)`, `rows()`, `cols()`, `Index<(usize, usize)>`/`IndexMut`, `transpose()`, `Mul<&Vector> for &Matrix -> Vector`, `Mul<&Matrix> for &Matrix -> Matrix` (panic on dimension mismatch; documented), `Clone`, `Debug`, `PartialEq`.
  - Re-exported as `mercury::{Vector, Matrix}`.

- [ ] **Step 1: Write the failing test**

Create `tests/core_dynamic.rs`:

```rust
//! Dynamic Vector/Matrix unit tests (host-side types; no Enzyme leg —
//! kernels receive dynamic data as slices, which Phase 1 already proves).

use mercury::{Matrix, Vector};

#[test]
fn vector_basics() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let w = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());
    assert_eq!(v[1], 2.0);
    assert!((v.dot(&w) - 32.0).abs() < 1e-15);
    assert!((v.norm_squared() - 14.0).abs() < 1e-15);
    assert_eq!((&v + &w).as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!((&w - &v).as_slice(), &[3.0, 3.0, 3.0]);
    assert_eq!((&v * 2.0).as_slice(), &[2.0, 4.0, 6.0]);

    let mut z = Vector::zeros(2);
    z[0] = 7.0;
    assert_eq!(z.as_slice(), &[7.0, 0.0]);
}

#[test]
fn matrix_basics() {
    let a = Matrix::from_fn(2, 3, |i, j| (3 * i + j) as f64); // [[0,1,2],[3,4,5]]
    assert_eq!((a.rows(), a.cols()), (2, 3));
    assert_eq!(a[(1, 2)], 5.0);

    let t = a.transpose();
    assert_eq!((t.rows(), t.cols()), (3, 2));
    assert_eq!(t[(2, 1)], 5.0);

    let v = Vector::from_slice(&[1.0, 1.0, 1.0]);
    assert_eq!((&a * &v).as_slice(), &[3.0, 12.0]);

    let b = Matrix::from_fn(3, 2, |i, j| (2 * i + j) as f64); // [[0,1],[2,3],[4,5]]
    let c = &a * &b;
    assert_eq!((c.rows(), c.cols()), (2, 2));
    // row0 = [0,1,2]·cols -> [0*0+1*2+2*4, 0*1+1*3+2*5] = [10, 13]
    assert_eq!(c[(0, 0)], 10.0);
    assert_eq!(c[(0, 1)], 13.0);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn matvec_dimension_mismatch_panics() {
    let a = Matrix::zeros(2, 3);
    let v = Vector::zeros(2);
    let _ = &a * &v;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop "path:$PWD" --command cargo test --release --test core_dynamic`
Expected: COMPILE ERROR — `unresolved import mercury::{Matrix, Vector}`.

- [ ] **Step 3: Implement**

Create `src/core/vector.rs`:

```rust
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
        Self { data: data.to_vec() }
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
        Vector::from_vec(self.data.iter().zip(&rhs.data).map(|(a, b)| a + b).collect())
    }
}

impl Sub for &Vector {
    type Output = Vector;
    /// # Panics
    /// On length mismatch.
    fn sub(self, rhs: Self) -> Vector {
        assert_eq!(self.len(), rhs.len(), "dimension mismatch in Vector sub");
        Vector::from_vec(self.data.iter().zip(&rhs.data).map(|(a, b)| a - b).collect())
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;
    fn mul(self, s: f64) -> Vector {
        Vector::from_vec(self.data.iter().map(|a| a * s).collect())
    }
}
```

Create `src/core/matrix.rs`:

```rust
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
```

In `src/core/mod.rs` add `mod matrix; mod vector;` and `pub use matrix::Matrix; pub use vector::Vector;`.
In `src/lib.rs` update: `pub use crate::core::{Matrix, SMatrix, SVector, Vector};`

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop "path:$PWD" --command cargo test --release --test core_dynamic`
Expected: 3 tests PASS (one via `should_panic`).

- [ ] **Step 5: Commit**

```bash
git add src/core/ src/lib.rs tests/core_dynamic.rs
git commit -m "feat(core): dynamic Vector/Matrix host-side types with slice bridges"
```

**Manual checkpoint 4:** `./scripts/test.sh` green.

---

### Task 5: `solve_fixed<N>` — kernel-safe stack solve (differentiate-through)

The flagship: a partial-pivot Gaussian elimination written entirely in POD style, so Enzyme differentiates *through* it — the thing nalgebra's `LU::solve` could not do. This is the correct approach for small in-kernel systems; the adjoint rule (Task 7) is the correct approach at problem scale.

**Files:**
- Create: `src/linalg/mod.rs`
- Create: `src/linalg/error.rs`
- Create: `src/linalg/fixed.rs`
- Modify: `src/lib.rs` (add `pub mod linalg;`, re-export `solve_fixed`, `LinalgError`)
- Test: `tests/linalg_fixed.rs`

**Interfaces:**
- Consumes: `SMatrix<N, N>`, `SVector<N>`.
- Produces:
  - `LinalgError` enum: `Singular { pivot_index: usize }`, `DimensionMismatch { rows: usize, cols: usize }` — `Display` + `std::error::Error` (mirrors `ValidationError` style).
  - `pub fn solve_fixed<const N: usize>(a: &SMatrix<N, N>, b: &SVector<N>) -> Result<SVector<N>, LinalgError>`
  - Re-exported: `mercury::solve_fixed`, `mercury::LinalgError`.

- [ ] **Step 1: Write the failing test**

Create `tests/linalg_fixed.rs`:

```rust
#![feature(autodiff)]

//! solve_fixed unit tests + the differentiate-through-the-solver Enzyme test.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{LinalgError, SMatrix, SVector, solve_fixed};
use std::autodiff::autodiff_reverse;

#[test]
fn solves_known_system() {
    // A = [[4,1],[1,3]], b = [1,2] -> x = [1/11, 7/11]
    let a = SMatrix::new([[4.0, 1.0], [1.0, 3.0]]);
    let b = SVector::new([1.0, 2.0]);
    let x = solve_fixed(&a, &b).expect("well-conditioned");
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);

    // Residual check on a 4x4 that forces pivoting (zero on the diagonal).
    let a4 = SMatrix::new([
        [0.0, 2.0, 1.0, -1.0],
        [3.0, 0.5, -2.0, 1.0],
        [1.0, -1.0, 4.0, 2.0],
        [2.0, 1.0, 1.0, 3.0],
    ]);
    let b4 = SVector::new([1.0, -2.0, 3.0, 0.5]);
    let x4 = solve_fixed(&a4, &b4).expect("invertible");
    let r = a4 * x4 - b4;
    assert!(r.norm() < 1e-12, "residual {r:?}");
}

#[test]
fn singular_matrix_is_an_error() {
    let a = SMatrix::new([[1.0, 2.0], [2.0, 4.0]]); // rank 1
    let b = SVector::new([1.0, 1.0]);
    assert!(matches!(
        solve_fixed(&a, &b),
        Err(LinalgError::Singular { .. })
    ));
}

// --- Enzyme leg: differentiate THROUGH the pivoting solver.
// x[0..9] -> A = M + 5I (well-conditioned), x[9..12] -> b, out = |A^{-1} b|^2.
// This is exactly the kernel nalgebra's LU::solve failed to compile
// (metis-ad-spike/linalg_compat, na_solve_new).

#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(x: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| x[3 * i + j] + if i == j { 5.0 } else { 0.0 });
    let b = SVector::<3>::from_fn(|i| x[9 + i]);
    let s = solve_fixed(&a, &b).expect("shifted matrix is well-conditioned");
    *out = s.norm_squared();
}

fn kernel_value(x: &[f64]) -> f64 {
    let mut out = 0.0;
    kernel(x, &mut out);
    out
}

#[test]
fn enzyme_differentiates_through_solve() {
    let x = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];
    let mut grad = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&x, &mut grad, &mut out, &mut dout);

    let fd = central_difference_gradient(kernel_value, &x, 1.0e-6).expect("fd");
    let check = compare_gradients(&grad, &fd).expect("shape");
    assert!(check.max_abs_error < 1.0e-4, "{check:?}\n ad={grad:?}\n fd={fd:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop "path:$PWD" --command cargo test --release --test linalg_fixed`
Expected: COMPILE ERROR — `unresolved import mercury::solve_fixed`.

- [ ] **Step 3: Implement**

Create `src/linalg/error.rs`:

```rust
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
```

Create `src/linalg/fixed.rs`:

```rust
//! Kernel-safe fixed-size dense solve (partial-pivot Gaussian elimination).

use crate::core::{SMatrix, SVector};

use super::LinalgError;

/// Pivot magnitudes below this are treated as singular.
const PIVOT_TOLERANCE: f64 = 1.0e-12;

/// Solves `A x = b` for small fixed-size systems on the stack.
///
/// Written in the Enzyme-safe POD style (element-wise construction, no bulk
/// array copies/swaps), so it is valid to differentiate *through* this
/// function inside a kernel: pivot choices are piecewise-constant in the
/// inputs, and Enzyme differentiates the taken branch. For problem-scale
/// systems use the dynamic [`solve`](crate::linalg::solve) primitive and its
/// adjoint rule instead.
///
/// # Errors
///
/// [`LinalgError::Singular`] when the best available pivot is below
/// tolerance.
pub fn solve_fixed<const N: usize>(
    a: &SMatrix<N, N>,
    b: &SVector<N>,
) -> Result<SVector<N>, LinalgError> {
    // Working copies, built element-wise (no memcpy on the AD path).
    let mut m: [[f64; N]; N] = std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)]));
    let mut y: [f64; N] = std::array::from_fn(|i| b[i]);

    for k in 0..N {
        // Partial pivot: largest magnitude in column k at or below row k.
        let mut pivot_row = k;
        let mut pivot_mag = m[k][k].abs();
        for i in (k + 1)..N {
            let mag = m[i][k].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_mag < PIVOT_TOLERANCE {
            return Err(LinalgError::Singular { pivot_index: k });
        }
        if pivot_row != k {
            // Element-wise row swap (mem::swap of arrays risks memcpy).
            for j in 0..N {
                let tmp = m[k][j];
                m[k][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
            let tmp = y[k];
            y[k] = y[pivot_row];
            y[pivot_row] = tmp;
        }

        // Eliminate below the pivot.
        for i in (k + 1)..N {
            let factor = m[i][k] / m[k][k];
            m[i][k] = 0.0;
            for j in (k + 1)..N {
                m[i][j] -= factor * m[k][j];
            }
            y[i] -= factor * y[k];
        }
    }

    // Back substitution, in place over y.
    for i in (0..N).rev() {
        let mut acc = y[i];
        for j in (i + 1)..N {
            acc -= m[i][j] * y[j];
        }
        y[i] = acc / m[i][i];
    }

    Ok(SVector::from_fn(|i| y[i]))
}
```

Create `src/linalg/mod.rs`:

```rust
//! Linear-algebra primitives with Mercury-owned derivative rules.

mod error;
mod fixed;

pub use error::LinalgError;
pub use fixed::solve_fixed;
```

In `src/lib.rs` add `pub mod linalg;` and extend re-exports:
`pub use crate::linalg::{LinalgError, solve_fixed};`

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop "path:$PWD" --command cargo test --release --test linalg_fixed`
Expected: 3 tests PASS. `enzyme_differentiates_through_solve` compiling at all is the headline — this exact kernel shape fails with nalgebra.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/ src/lib.rs tests/linalg_fixed.rs
git commit -m "feat(linalg): kernel-safe solve_fixed, Enzyme differentiates through pivoting"
```

**Manual checkpoint 5:** `./scripts/test.sh` green. For the side-by-side: `nix develop "path:$PWD" --command cargo test --release --test linalg_fixed enzyme -- --nocapture` here, versus `na_solve_new`'s compile failure in the spike.

---

### Task 6: Dynamic LU factorization + solve (primal)

**Files:**
- Create: `src/linalg/lu.rs`
- Modify: `src/linalg/mod.rs`, `src/lib.rs` (exports)
- Test: `tests/linalg_lu.rs`

**Interfaces:**
- Consumes: `Matrix`, `Vector`, `LinalgError`.
- Produces:
  - `pub struct LuFactors` (owns factored data; reusable for many solves and for the adjoint rule).
  - `pub fn lu_factor(a: &Matrix) -> Result<LuFactors, LinalgError>`
  - `impl LuFactors { pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>; pub fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError>; pub fn dimension(&self) -> usize }`
  - `pub fn solve(a: &Matrix, b: &Vector) -> Result<Vector, LinalgError>` (factor + solve convenience)
  - Re-exported: `mercury::{lu_factor, solve, LuFactors}`.
- Permutation convention: `perm[i]` = original row of `a` occupying position `i` after pivoting, i.e. `P a = L U` with `(P b)[i] = b[perm[i]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/linalg_lu.rs`:

```rust
//! Dynamic LU factor/solve tests (primal only; adjoint rule is next task).

use mercury::{Matrix, Vector, lu_factor, solve};

fn test_matrix() -> Matrix {
    // Needs pivoting (zero at (0,0)); well-conditioned.
    Matrix::from_fn(4, 4, |i, j| match (i, j) {
        (0, 0) => 0.0,
        (i, j) if i == j => 4.0 + i as f64,
        (i, j) => 0.3 * ((2 * i + 3 * j) as f64).sin(),
    })
}

#[test]
fn solve_has_small_residual() {
    let a = test_matrix();
    let b = Vector::from_slice(&[1.0, -2.0, 3.0, 0.5]);
    let x = solve(&a, &b).expect("invertible");
    let r = &(&a * &x) - &b;
    assert!(r.norm_squared().sqrt() < 1e-12, "residual {r:?}");
}

#[test]
fn factors_are_reusable_and_transposed_solve_works() {
    let a = test_matrix();
    let f = lu_factor(&a).expect("invertible");
    assert_eq!(f.dimension(), 4);

    let b1 = Vector::from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let b2 = Vector::from_slice(&[0.0, 1.0, -1.0, 2.0]);
    let x1 = f.solve(&b1).expect("solve");
    let x2 = f.solve(&b2).expect("solve");
    assert!((&(&a * &x1) - &b1).norm_squared().sqrt() < 1e-12);
    assert!((&(&a * &x2) - &b2).norm_squared().sqrt() < 1e-12);

    // A^T z = c via the same factors.
    let c = Vector::from_slice(&[0.5, 1.5, -0.5, 1.0]);
    let z = f.solve_transposed(&c).expect("transposed solve");
    let at = a.transpose();
    assert!((&(&at * &z) - &c).norm_squared().sqrt() < 1e-12);
}

#[test]
fn singular_matrix_errors() {
    let a = Matrix::from_fn(3, 3, |i, _| i as f64); // identical columns
    assert!(lu_factor(&a).is_err());
}

#[test]
fn non_square_errors() {
    let a = Matrix::zeros(3, 4);
    assert!(lu_factor(&a).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop "path:$PWD" --command cargo test --release --test linalg_lu`
Expected: COMPILE ERROR — `unresolved import mercury::lu_factor`.

- [ ] **Step 3: Implement**

Create `src/linalg/lu.rs`:

```rust
//! Dynamic partial-pivot LU factorization — the backend seam.
//!
//! This hand-rolled factorization is Mercury's initial backend; the public
//! contract (`lu_factor`, `LuFactors::solve`) is designed so a faer backend
//! can replace the internals without any caller change. Callers never
//! differentiate through this code — derivatives come from the adjoint rule
//! (`solve_vjp`/`solve_jvp`).

use crate::core::{Matrix, Vector};

use super::LinalgError;

/// Pivot magnitudes below this are treated as singular.
const PIVOT_TOLERANCE: f64 = 1.0e-12;

/// Reusable LU factors of a square matrix (`P A = L U`).
#[derive(Debug, Clone)]
pub struct LuFactors {
    /// Combined L (unit lower, below diagonal) and U (upper) storage.
    lu: Matrix,
    /// `perm[i]` = original row of `A` occupying position `i`.
    perm: Vec<usize>,
}

/// Factors a square matrix with partial pivoting.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] for non-square input;
/// [`LinalgError::Singular`] when a pivot falls below tolerance.
pub fn lu_factor(a: &Matrix) -> Result<LuFactors, LinalgError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LinalgError::DimensionMismatch { rows: a.rows(), cols: a.cols() });
    }

    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_mag = lu[(k, k)].abs();
        for i in (k + 1)..n {
            let mag = lu[(i, k)].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_mag < PIVOT_TOLERANCE {
            return Err(LinalgError::Singular { pivot_index: k });
        }
        if pivot_row != k {
            for j in 0..n {
                let tmp = lu[(k, j)];
                lu[(k, j)] = lu[(pivot_row, j)];
                lu[(pivot_row, j)] = tmp;
            }
            perm.swap(k, pivot_row);
        }

        for i in (k + 1)..n {
            let factor = lu[(i, k)] / lu[(k, k)];
            lu[(i, k)] = factor; // store the L multiplier in place
            for j in (k + 1)..n {
                let delta = factor * lu[(k, j)];
                lu[(i, j)] -= delta;
            }
        }
    }

    Ok(LuFactors { lu, perm })
}

impl LuFactors {
    /// Side length of the factored matrix.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.perm.len()
    }

    /// Solves `A x = b` using the stored factors.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch { rows: b.len(), cols: 1 });
        }

        // Forward: L y = P b (L unit lower).
        let mut y = Vector::zeros(n);
        for i in 0..n {
            let mut acc = b[self.perm[i]];
            for j in 0..i {
                acc -= self.lu[(i, j)] * y[j];
            }
            y[i] = acc;
        }
        // Backward: U x = y.
        for i in (0..n).rev() {
            let mut acc = y[i];
            for j in (i + 1)..n {
                acc -= self.lu[(i, j)] * y[j];
            }
            y[i] = acc / self.lu[(i, i)];
        }
        Ok(y)
    }

    /// Solves `A^T z = c` using the same factors
    /// (`A^T = U^T L^T P`, so solve `U^T w = c`, then `L^T v = w`, then
    /// undo the permutation).
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `c` has the wrong length.
    pub fn solve_transposed(&self, c: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        if c.len() != n {
            return Err(LinalgError::DimensionMismatch { rows: c.len(), cols: 1 });
        }

        // Forward: U^T w = c (U^T lower, non-unit diagonal).
        let mut w = Vector::zeros(n);
        for i in 0..n {
            let mut acc = c[i];
            for j in 0..i {
                acc -= self.lu[(j, i)] * w[j];
            }
            w[i] = acc / self.lu[(i, i)];
        }
        // Backward: L^T v = w (L^T upper, unit diagonal).
        for i in (0..n).rev() {
            let mut acc = w[i];
            for j in (i + 1)..n {
                acc -= self.lu[(j, i)] * w[j];
            }
            w[i] = acc;
        }
        // P x = v  =>  x[perm[i]] = v[i].
        let mut x = Vector::zeros(n);
        for i in 0..n {
            x[self.perm[i]] = w[i];
        }
        Ok(x)
    }
}

/// Convenience: factor + solve in one call.
///
/// # Errors
///
/// Propagates [`lu_factor`] and [`LuFactors::solve`] errors.
pub fn solve(a: &Matrix, b: &Vector) -> Result<Vector, LinalgError> {
    lu_factor(a)?.solve(b)
}
```

In `src/linalg/mod.rs` add `mod lu;` and `pub use lu::{LuFactors, lu_factor, solve};`.
In `src/lib.rs` extend: `pub use crate::linalg::{LinalgError, LuFactors, lu_factor, solve, solve_fixed};`

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop "path:$PWD" --command cargo test --release --test linalg_lu`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/ src/lib.rs tests/linalg_lu.rs
git commit -m "feat(linalg): dynamic partial-pivot LU with reusable factors + transposed solve"
```

**Manual checkpoint 6:** `./scripts/test.sh` green.

---

### Task 7: The adjoint rule — `solve_vjp` / `solve_jvp` + three-way agreement

The identity thesis made executable: derivative of a solve computed by the mathematical rule (two solves), never by differentiating the factorization — then shown to agree with Enzyme-through-`solve_fixed` and finite differences on the same problem.

Math (for `x = A^{-1} b`):
- VJP with output cotangent `x̄`: `b̄ = A^{-T} x̄`, `Ā = -b̄ x^T`.
- JVP with input tangents `(Ȧ, ḃ)`: `ẋ = A^{-1}(ḃ - Ȧ x)`.

**Files:**
- Create: `src/linalg/adjoint.rs`
- Modify: `src/linalg/mod.rs`, `src/lib.rs` (exports)
- Test: `tests/linalg_adjoint.rs`

**Interfaces:**
- Consumes: `LuFactors` (Task 6), `Matrix`, `Vector`; `solve_fixed` + Enzyme kernel from Task 5's test (re-stated locally — tests are separate crates).
- Produces:
  - `pub struct SolveGradients { pub a_bar: Matrix, pub b_bar: Vector }`
  - `pub fn solve_vjp(factors: &LuFactors, x: &Vector, x_bar: &Vector) -> Result<SolveGradients, LinalgError>`
  - `pub fn solve_jvp(factors: &LuFactors, x: &Vector, a_dot: &Matrix, b_dot: &Vector) -> Result<Vector, LinalgError>`
  - Re-exported: `mercury::{solve_vjp, solve_jvp, SolveGradients}`.

- [ ] **Step 1: Write the failing test**

Create `tests/linalg_adjoint.rs`:

```rust
#![feature(autodiff)]

//! Adjoint-rule tests: solve_vjp/solve_jvp vs finite differences vs Enzyme.
//!
//! The three-way agreement test is the Phase 2 thesis demo: the same
//! gradient from (1) finite differences, (2) Enzyme differentiating through
//! solve_fixed, (3) the adjoint rule composing two LU solves.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{
    Matrix, SMatrix, SVector, Vector, lu_factor, solve_fixed, solve_jvp, solve_vjp,
};
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

/// Objective: theta[0..9] -> A = M + 5I, theta[9..12] -> b, f = |x|^2.
fn objective(theta: &[f64]) -> f64 {
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let x = lu_factor(&a).expect("wc").solve(&b).expect("wc");
    x.norm_squared()
}

/// The same objective as an Enzyme kernel through solve_fixed.
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    let x = solve_fixed(&a, &b).expect("wc");
    *out = x.norm_squared();
}

fn adjoint_gradient(theta: &[f64]) -> Vec<f64> {
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let f = lu_factor(&a).expect("wc");
    let x = f.solve(&b).expect("wc");

    // f = |x|^2  =>  x_bar = 2x.
    let x_bar = &x * 2.0;
    let grads = solve_vjp(&f, &x, &x_bar).expect("vjp");

    // d theta_(3i+j) = a_bar[i][j] (diag shift is constant), d theta_(9+i) = b_bar[i].
    let mut g = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            g[3 * i + j] = grads.a_bar[(i, j)];
        }
        g[9 + i] = grads.b_bar[i];
    }
    g
}

#[test]
fn three_way_gradient_agreement() {
    let theta = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];

    // (1) finite differences
    let fd = central_difference_gradient(objective, &theta, 1.0e-6).expect("fd");

    // (2) Enzyme through solve_fixed
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&theta, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&theta)).abs() < 1e-12, "primal mismatch");

    // (3) adjoint rule
    let adjoint = adjoint_gradient(&theta);

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs adjoint: {enzyme_vs_adjoint:?}\n enzyme={enzyme:?}\n adjoint={adjoint:?}"
    );

    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "adjoint vs fd: {adjoint_vs_fd:?}"
    );
}

#[test]
fn jvp_matches_directional_finite_difference() {
    let theta = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let f = lu_factor(&a).expect("wc");
    let x = f.solve(&b).expect("wc");

    // Direction: perturb every A entry and b entry.
    let a_dot = Matrix::from_fn(3, 3, |i, j| 0.1 * ((i + 2 * j) as f64) - 0.2);
    let b_dot = Vector::from_slice(&[0.3, -0.1, 0.2]);
    let x_dot = solve_jvp(&f, &x, &a_dot, &b_dot).expect("jvp");

    // FD in that direction: x(A + h*Adot, b + h*bdot).
    let h = 1.0e-7;
    let a_p = Matrix::from_fn(3, 3, |i, j| a[(i, j)] + h * a_dot[(i, j)]);
    let a_m = Matrix::from_fn(3, 3, |i, j| a[(i, j)] - h * a_dot[(i, j)]);
    let b_p = Vector::from_slice(&[b[0] + h * b_dot[0], b[1] + h * b_dot[1], b[2] + h * b_dot[2]]);
    let b_m = Vector::from_slice(&[b[0] - h * b_dot[0], b[1] - h * b_dot[1], b[2] - h * b_dot[2]]);
    let x_p = lu_factor(&a_p).expect("wc").solve(&b_p).expect("wc");
    let x_m = lu_factor(&a_m).expect("wc").solve(&b_m).expect("wc");

    for i in 0..3 {
        let fd_i = (x_p[i] - x_m[i]) / (2.0 * h);
        assert!(
            (x_dot[i] - fd_i).abs() < 1.0e-5,
            "component {i}: jvp={} fd={fd_i}",
            x_dot[i]
        );
    }
}

#[test]
fn vjp_dimension_mismatch_errors() {
    let a = Matrix::from_fn(3, 3, |i, j| if i == j { 5.0 } else { 0.1 });
    let f = lu_factor(&a).expect("wc");
    let x = Vector::zeros(3);
    let bad = Vector::zeros(2);
    assert!(solve_vjp(&f, &x, &bad).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop "path:$PWD" --command cargo test --release --test linalg_adjoint`
Expected: COMPILE ERROR — `unresolved import mercury::solve_vjp`.

- [ ] **Step 3: Implement**

Create `src/linalg/adjoint.rs`:

```rust
//! The adjoint rule for linear solves — Mercury's first owned derivative
//! rule (decision 0003).
//!
//! For `x = A^{-1} b`:
//! - reverse (VJP), given output cotangent `x_bar`:
//!   `b_bar = A^{-T} x_bar`, `A_bar = -b_bar x^T`
//! - forward (JVP), given input tangents `(A_dot, b_dot)`:
//!   `x_dot = A^{-1} (b_dot - A_dot x)`
//!
//! Both reuse the primal [`LuFactors`] — no factorization is ever
//! differentiated, and the factorization backend can change freely.

use crate::core::{Matrix, Vector};

use super::{LinalgError, LuFactors};

/// Input cotangents produced by [`solve_vjp`].
#[derive(Debug, Clone, PartialEq)]
pub struct SolveGradients {
    /// Cotangent of the matrix: `A_bar = -b_bar x^T`.
    pub a_bar: Matrix,
    /// Cotangent of the right-hand side: `b_bar = A^{-T} x_bar`.
    pub b_bar: Vector,
}

/// Reverse-mode rule for `x = A^{-1} b`.
///
/// `factors` and `x` must come from the primal solve.
///
/// # Errors
///
/// Propagates dimension errors from the transposed solve.
pub fn solve_vjp(
    factors: &LuFactors,
    x: &Vector,
    x_bar: &Vector,
) -> Result<SolveGradients, LinalgError> {
    let b_bar = factors.solve_transposed(x_bar)?;
    let n = factors.dimension();
    let a_bar = Matrix::from_fn(n, n, |i, j| -b_bar[i] * x[j]);
    Ok(SolveGradients { a_bar, b_bar })
}

/// Forward-mode rule for `x = A^{-1} b`.
///
/// `factors` and `x` must come from the primal solve.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when tangent shapes disagree with the
/// factors; propagates solve errors.
pub fn solve_jvp(
    factors: &LuFactors,
    x: &Vector,
    a_dot: &Matrix,
    b_dot: &Vector,
) -> Result<Vector, LinalgError> {
    let n = factors.dimension();
    if a_dot.rows() != n || a_dot.cols() != n {
        return Err(LinalgError::DimensionMismatch { rows: a_dot.rows(), cols: a_dot.cols() });
    }
    let rhs = &*b_dot - &(a_dot * x);
    factors.solve(&rhs)
}
```

Note: `&*b_dot - &(a_dot * x)` uses the `Sub` impl on `&Vector` — if the borrow form fights you, bind `let ax = a_dot * x;` first and write `b_dot - &ax` with explicit references matching the `impl Sub for &Vector` signature.

In `src/linalg/mod.rs` add `mod adjoint;` and `pub use adjoint::{SolveGradients, solve_jvp, solve_vjp};`.
In `src/lib.rs` extend: `pub use crate::linalg::{LinalgError, LuFactors, SolveGradients, lu_factor, solve, solve_fixed, solve_jvp, solve_vjp};`

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop "path:$PWD" --command cargo test --release --test linalg_adjoint`
Expected: 3 tests PASS. `three_way_gradient_agreement` asserting Enzyme-vs-adjoint at `< 1e-9` is the money line: two completely different derivative mechanisms, same exact numbers.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/ src/lib.rs tests/linalg_adjoint.rs
git commit -m "feat(linalg): solve adjoint rule (vjp/jvp) with three-way gradient agreement"
```

**Manual checkpoint 7:** `./scripts/test.sh` green, then see the demo directly:
`nix develop "path:$PWD" --command cargo test --release --test linalg_adjoint three_way -- --nocapture`

---

### Task 8: Runnable example + docs wiring + full verify

**Files:**
- Create: `examples/solve_gradient.rs`
- Modify: `src/lib.rs` (crate docs describing the module map)
- Modify: `README.md` (source layout + Phase 2 blurb)

**Interfaces:**
- Consumes: everything above. Produces: no new API.

- [ ] **Step 1: Write the example**

Create `examples/solve_gradient.rs`:

```rust
//! Phase 2 demo: one gradient, three ways.
//!
//! d/d(theta) |A(theta)^{-1} b(theta)|^2 computed by
//!   (1) central finite differences,
//!   (2) Enzyme reverse-mode differentiating THROUGH solve_fixed,
//!   (3) the adjoint rule (two LU solves) — no AD inside the solve at all.
//!
//! Run: `nix develop "path:$PWD" --command cargo run --release --example solve_gradient`
#![feature(autodiff)]

use mercury::validation::central_difference_gradient;
use mercury::{Matrix, SMatrix, SVector, Vector, lu_factor, solve_fixed, solve_vjp};
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

fn objective(theta: &[f64]) -> f64 {
    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    lu_factor(&a).unwrap().solve(&b).unwrap().norm_squared()
}

#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    *out = solve_fixed(&a, &b).unwrap().norm_squared();
}

fn main() {
    let theta = [
        0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
    ];

    let fd = central_difference_gradient(objective, &theta, 1.0e-6).unwrap();

    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&theta, &mut enzyme, &mut out, &mut dout);

    let a = Matrix::from_fn(3, 3, |i, j| {
        theta[3 * i + j] + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = Vector::from_slice(&theta[9..12]);
    let f = lu_factor(&a).unwrap();
    let x = f.solve(&b).unwrap();
    let grads = solve_vjp(&f, &x, &(&x * 2.0)).unwrap();
    let mut adjoint = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            adjoint[3 * i + j] = grads.a_bar[(i, j)];
        }
        adjoint[9 + i] = grads.b_bar[i];
    }

    println!("objective value = {out:.12}\n");
    println!("{:>4}  {:>18}  {:>18}  {:>18}", "i", "finite-diff", "enzyme", "adjoint-rule");
    for i in 0..12 {
        println!(
            "{i:>4}  {:>18.12}  {:>18.12}  {:>18.12}",
            fd[i], enzyme[i], adjoint[i]
        );
    }
}
```

- [ ] **Step 2: Run the example**

Run: `nix develop "path:$PWD" --command cargo run --release --example solve_gradient`
Expected: a 12-row table where the `enzyme` and `adjoint-rule` columns agree to ~12 digits and `finite-diff` agrees to ~7.

- [ ] **Step 3: Update crate docs and README**

In `src/lib.rs`, replace the crate doc comment with:

```rust
//! Differentiable math substrate for `pantheon-rs`.
//!
//! Every Mercury primitive is plain-`f64` Rust with a validated,
//! Mercury-owned derivative rule (decision 0003). Enzyme differentiates
//! user kernels; Mercury owns the rules at the joints.
//!
//! - [`core`]: POD-transparent types. Fixed-size (`SVector`, `SMatrix`) are
//!   kernel-safe; dynamic (`Vector`, `Matrix`) host data outside kernels.
//! - [`geometry`]: `Quaternion` and rotations (analytic derivatives).
//! - [`linalg`]: `solve_fixed` (kernel-safe, differentiate-through) and the
//!   dynamic LU `solve` whose derivative is the adjoint rule
//!   (`solve_vjp`/`solve_jvp`) — never the factorization.
//! - [`validation`]: finite-difference oracles for the three-legged test law.
```

(keep the existing `#![feature(autodiff)]`, `#![forbid(unsafe_code)]`, module decls, and re-exports below it).

In `README.md`, update the Source Layout block to:

```text
src/
  lib.rs
  objective.rs     # scalar_objective! macro (Enzyme reverse entry points)
  validation.rs    # finite-difference oracles
  core/            # SVector, SMatrix (kernel-safe) + Vector, Matrix (host-side)
  geometry/        # Quaternion
  linalg/          # solve_fixed, LU solve + adjoint rule (solve_vjp/solve_jvp)
tests/             # one suite per module, three-legged test law
examples/
  solve_gradient.rs  # one gradient, three ways (fd / enzyme / adjoint)
```

and add under the Phase 1 bullet list:

```markdown
Phase 2 adds the owned core types and the first owned derivative rule:
kernel-safe `SVector`/`SMatrix`/`Quaternion` (proven against Enzyme per
type), host-side `Vector`/`Matrix`, and linear solve where small systems
differentiate through `solve_fixed` while problem-scale systems use the
LU primitive with the adjoint rule. See
`docs/decisions/0003-differentiable-primitives-identity.md`.
```

- [ ] **Step 4: Full verification**

Run: `./scripts/ci.sh`
Expected: format check, clippy, build, tests all green. Fix any clippy pedantic complaints surfaced here before committing.

- [ ] **Step 5: Commit**

```bash
git add examples/ src/lib.rs README.md
git commit -m "docs+example: one-gradient-three-ways demo, Phase 2 module map"
```

**Manual checkpoint 8 (final):** run the demo yourself:
```bash
cd ~/sources/pantheon-rs/mercury
nix develop "path:$PWD" --command cargo run --release --example solve_gradient
./scripts/ci.sh
```

---

## Deferred (explicitly NOT this plan)

- Negative-pattern compile-fail harness (Enzyme failures happen under fat LTO; `trybuild`-style testing needs its own design) — known-bad patterns are documented in rustdoc (`zeros()`) and `metis-ad-spike/linalg_compat/RESULTS.md` for now.
- faer as the LU backend behind `lu_factor` (contract already shaped for it).
- `jvp`/`vjp` for the `scalar_objective!` macro (forward mode) — Phase 1 exit criterion, separate slice.
- Interpolation, root finding, integration — Phases 3 and 4 per decision 0003.
