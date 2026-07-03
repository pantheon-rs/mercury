# Phase 3: Decomposition Suite + Structural Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faer-inspired decomposition suite (Cholesky LLT/LDLT, Householder QR + least squares) with a unified `Factorization`-trait adjoint rule, a kernel-safe SPD solve, and structural layer (`Perm`, triangular substrate, reductions), per `docs/superpowers/specs/2026-07-03-phase-3-decompositions-design.md`.

**Architecture:** Host-side decompositions operate on the dynamic `Matrix`/`Vector` types and share a triangular-substitution substrate; every square-solve factorization implements one `Factorization` trait so `solve_vjp`/`solve_jvp` generalize across LU/LLT/LDLT. Least squares gets its own adjoint rule. The only kernel-side addition is `solve_spd_fixed_unchecked`, written under Enzyme IR rules. Views (`MatRef`/`MatMut`) are **YAGNI-gated out of this plan** per spec phasing step 7: every algorithm below works with owned `Matrix` indexing; views land only when blocked algorithms need them (Phase 4 candidate).

**Tech Stack:** Rust nightly 2026-03-03 + Enzyme (`-Zautodiff=Enable`), no external deps, branch `phase3_faer_refactor`.

## Global Constraints

Copied from the spec and decision 0003; every task's requirements implicitly include these.

- **Toolchain:** all builds/tests run `--release` (Enzyme requires fat LTO; debug profile fails). Run inside the nix dev shell (`nix develop` or direnv). Test command shape: `cargo test --release --test <name>`.
- **f64 only.** No generic scalars. `#![forbid(unsafe_code)]` stands — no `unsafe` anywhere, including views-style tricks.
- **Enzyme IR rules bind all kernel-reachable code** (anything callable from an `#[autodiff_reverse]` kernel):
  1. No `std::array::from_fn`.
  2. No `[0.0; N]` zero-init on kernel paths; runtime splat `[f(0); N]` is OK.
  3. No iterator adapters in kernel loops — plain `for i in 0..N` only.
  4. Construct-then-mutate-through-binding (never bind-array-then-wrap).
  5. `-Z autodiff=LooseTypes` is banned (compiles, wrong gradients).
  6. Nested aggregate splat needs a constant seed + `#[inline(never)]` (see `SMatrix::from_fn`).
  7. No `Result<T, E>` returns from kernel-reachable functions — `_unchecked` NaN-propagation variants instead.
  8. Enzyme test legs go behind `#[cfg(not(coverage))]` (coverage atomics break Enzyme).
- **Host-side code** (dynamic `Matrix`/`Vector`, factorizations) is never differentiated through — normal Rust rules apply there.
- **Errors:** one shared `LinalgError`. All new host entry points validate dimensions and return `Result` — the only panicking APIs are `Index`/`IndexMut` and core-type operators (established convention). Pivot threshold is the shared `pub(crate) PIVOT_TOLERANCE = 1.0e-12` in `src/linalg/mod.rs`.
- **Tolerances:** reconstruction/cross-checks `1e-12`; rule-vs-Enzyme gradient agreement `1e-9`; rule-vs-finite-difference `1e-4` (FD step `1e-6`).
- **Test files** start with `#![allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::many_single_char_names)]`; Enzyme test files additionally start with `#![feature(autodiff)]`.
- **Lints:** clippy `all`/`pedantic`/`nursery` are warn-as-CI-fail; `missing_docs = "warn"` — every public item gets a doc comment with `# Errors`/`# Panics` sections where applicable.
- **Phase 2 tests are pins:** refactoring tasks (3, 4) must leave every existing test passing unchanged — do not edit existing test files except where a task explicitly says to.
- **Commits:** conventional-commit style, `--no-verify` NOT needed unless the treefmt hook stages unrelated files — verify `git status` is clean of unrelated changes before committing.

---

### Task 1: `Perm` type

**Files:**
- Create: `src/core/perm.rs`
- Modify: `src/core/mod.rs` (add `mod perm; pub use perm::Perm;`)
- Modify: `src/lib.rs` (add `Perm` to the `crate::core` re-export list)
- Test: `tests/core_perm.rs`

**Interfaces:**
- Consumes: `Vector` (`crate::core::Vector`) — `zeros(n)`, `len()`, `Index`/`IndexMut`.
- Produces: `Perm` with `identity(n) -> Perm`, `len() -> usize` (const), `is_empty() -> bool` (const), `swap(&mut self, i, j)`, `apply(&self, v: &Vector) -> Vector`, `apply_inverse(&self, v: &Vector) -> Vector`, `sign(&self) -> f64`. Task 3 (LU refactor) and Task 10 (determinant) rely on exactly these names.

- [ ] **Step 1: Write the failing test**

Create `tests/core_perm.rs`:

```rust
// Exact float asserts are intentional in tests.
#![allow(clippy::float_cmp)]

//! `Perm` permutation type tests.

use mercury::{Perm, Vector};

#[test]
fn identity_applies_as_noop() {
    let p = Perm::identity(3);
    assert_eq!(p.len(), 3);
    assert!(!p.is_empty());
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(p.apply(&v), v);
    assert_eq!(p.apply_inverse(&v), v);
    assert_eq!(p.sign(), 1.0);
}

#[test]
fn swap_permutes_and_inverse_round_trips() {
    let mut p = Perm::identity(3);
    p.swap(0, 2); // perm = [2, 1, 0]
    let v = Vector::from_slice(&[10.0, 20.0, 30.0]);
    // (Pv)[i] = v[perm[i]]
    let pv = p.apply(&v);
    assert_eq!(pv, Vector::from_slice(&[30.0, 20.0, 10.0]));
    // apply_inverse undoes apply
    assert_eq!(p.apply_inverse(&pv), v);
}

#[test]
fn sign_tracks_transposition_parity() {
    let mut p = Perm::identity(4);
    assert_eq!(p.sign(), 1.0);
    p.swap(0, 1);
    assert_eq!(p.sign(), -1.0);
    p.swap(2, 3);
    assert_eq!(p.sign(), 1.0);
    p.swap(1, 1); // self-swap is a no-op, parity unchanged
    assert_eq!(p.sign(), 1.0);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn apply_panics_on_length_mismatch() {
    let p = Perm::identity(3);
    let v = Vector::zeros(2);
    let _ = p.apply(&v);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test core_perm`
Expected: FAIL to compile — `no 'Perm' in the root`.

- [ ] **Step 3: Write the implementation**

Create `src/core/perm.rs`:

```rust
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
```

In `src/core/mod.rs`, add alongside the existing module declarations/re-exports:

```rust
mod perm;
pub use perm::Perm;
```

In `src/lib.rs`, extend the core re-export:

```rust
pub use crate::core::{Matrix, Perm, SMatrix, SVector, Vector};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test core_perm`
Expected: PASS (4 tests). Then run the full suite to confirm nothing broke: `cargo test --release`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/perm.rs src/core/mod.rs src/lib.rs tests/core_perm.rs
git commit -m "feat: add Perm permutation type with parity tracking"
```

---

### Task 2: Triangular-substitution substrate

**Files:**
- Create: `src/linalg/triangular.rs`
- Modify: `src/linalg/mod.rs` (add `mod triangular;`)
- Test: in-module `#[cfg(test)] mod tests` inside `src/linalg/triangular.rs` (the functions are `pub(crate)` — integration tests in `tests/` cannot see them)

**Interfaces:**
- Consumes: `Matrix`, `Vector`, `LinalgError`, `PIVOT_TOLERANCE`.
- Produces (all `pub(crate)`, all `-> Result<Vector, LinalgError>`):
  - `solve_lower(m: &Matrix, b: &Vector, unit_diag: bool)` — solves `L x = b` reading only the lower triangle of `m` (diagonal implied 1 when `unit_diag`).
  - `solve_upper(m: &Matrix, b: &Vector, unit_diag: bool)` — solves `U x = b` reading only the upper triangle.
  - `solve_lower_transposed(m: &Matrix, b: &Vector, unit_diag: bool)` — solves `Lᵀ x = b`.
  - `solve_upper_transposed(m: &Matrix, b: &Vector, unit_diag: bool)` — solves `Uᵀ x = b`.
  - All four: `DimensionMismatch` for non-square `m` or wrong `b` length; `Singular { pivot_index: i }` when a non-unit diagonal entry has `|m[(i,i)]| < PIVOT_TOLERANCE`.
  - Reading only one triangle is load-bearing: Tasks 3/5/6 pass combined-storage matrices (LU's `lu`, LDLT's unit-lower `l`) whose other triangle holds unrelated data.

- [ ] **Step 1: Write the failing tests**

Create `src/linalg/triangular.rs` with the tests first (implementation stubs come in Step 3; to make the test compile-fail cleanly, write the whole file in Step 3 and just run the module tests — for this in-module case, combine: write tests referring to the four functions, verify the crate fails to compile):

```rust
//! Shared triangular-substitution substrate (faer's `triangular_solve` idea).
//!
//! Every function reads ONLY its named triangle of `m`, so callers may pass
//! combined-storage matrices (e.g. LU's packed `L\U`, LDLT's unit-lower `l`)
//! whose other triangle holds unrelated data.

use crate::core::{Matrix, Vector};

use super::{LinalgError, PIVOT_TOLERANCE};

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
}
```

Note the `Err(...)` comparison requires `LinalgError: PartialEq` — it already derives it.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release --lib triangular`
Expected: FAIL to compile — `cannot find function 'solve_lower'`. (First add `mod triangular;` to `src/linalg/mod.rs`, below the existing `mod` lines.)

- [ ] **Step 3: Write the implementation**

Add above the `#[cfg(test)]` module in `src/linalg/triangular.rs`:

```rust
/// Validates a square system: `m` is `n x n` and `b` has length `n`.
fn check_square_system(m: &Matrix, b: &Vector) -> Result<usize, LinalgError> {
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
pub(crate) fn solve_lower(
    m: &Matrix,
    b: &Vector,
    unit_diag: bool,
) -> Result<Vector, LinalgError> {
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
pub(crate) fn solve_upper(
    m: &Matrix,
    b: &Vector,
    unit_diag: bool,
) -> Result<Vector, LinalgError> {
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
pub(crate) fn solve_lower_transposed(
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
pub(crate) fn solve_upper_transposed(
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --lib triangular`
Expected: PASS (7 tests). Then `cargo test --release` — everything green.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/triangular.rs src/linalg/mod.rs
git commit -m "feat: add shared triangular-substitution substrate"
```

---

### Task 3: Refactor LU onto `Perm` + triangular substrate

**Files:**
- Modify: `src/linalg/lu.rs`
- Test: existing `tests/linalg_lu.rs`, `tests/linalg_adjoint.rs` are the pins — do NOT edit them; they must pass unchanged.

**Interfaces:**
- Consumes: `Perm` (Task 1), `solve_lower`/`solve_upper`/`solve_lower_transposed`/`solve_upper_transposed` (Task 2).
- Produces: `LuFactors` public API unchanged (`dimension`, `solve`, `solve_transposed`, `lu_factor`, `solve`). Internal `perm` field becomes `Perm` — Task 10's `determinant` relies on `self.perm.sign()`.

- [ ] **Step 1: Refactor `LuFactors` internals**

In `src/linalg/lu.rs`:

1. Change imports:

```rust
use crate::core::{Matrix, Perm, Vector};

use super::triangular::{
    solve_lower, solve_lower_transposed, solve_upper, solve_upper_transposed,
};
use super::{LinalgError, PIVOT_TOLERANCE};
```

2. Change the struct field:

```rust
pub struct LuFactors {
    /// Combined L (unit lower, below diagonal) and U (upper) storage.
    lu: Matrix,
    /// `perm[i]` = original row of `A` occupying position `i`.
    perm: Perm,
}
```

3. In `lu_factor`, replace `let mut perm: Vec<usize> = (0..n).collect();` with `let mut perm = Perm::identity(n);` — the existing `perm.swap(k, pivot_row);` call compiles unchanged against `Perm::swap`.

4. Replace the body of `solve` (keep the doc comment and the up-front length check):

```rust
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch {
                rows: b.len(),
                cols: 1,
            });
        }
        // L y = P b (L unit lower), then U x = y.
        let y = solve_lower(&self.lu, &self.perm.apply(b), true)?;
        solve_upper(&self.lu, &y, false)
    }
```

5. Replace the body of `solve_transposed` (keep doc comment and length check):

```rust
    pub fn solve_transposed(&self, c: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        if c.len() != n {
            return Err(LinalgError::DimensionMismatch {
                rows: c.len(),
                cols: 1,
            });
        }
        // A^T = U^T L^T P: solve U^T w = c, then L^T v = w, then undo P.
        let w = solve_upper_transposed(&self.lu, c, false)?;
        let v = solve_lower_transposed(&self.lu, &w, true)?;
        Ok(self.perm.apply_inverse(&v))
    }
```

6. `dimension()` stays `pub const fn` — `Perm::len` is const.

- [ ] **Step 2: Run the pins**

Run: `cargo test --release`
Expected: ALL tests pass, none modified. Pay attention to `linalg_lu` and `linalg_adjoint` (the three-way agreement test) — they pin the refactor's behavior exactly.

- [ ] **Step 3: Commit**

```bash
git add src/linalg/lu.rs
git commit -m "refactor: LU onto Perm + shared triangular substrate"
```

---

### Task 4: `Factorization` trait + adjoint generalization

**Files:**
- Create: `src/linalg/factorization.rs`
- Modify: `src/linalg/mod.rs` (add `mod factorization; pub use factorization::Factorization;`)
- Modify: `src/linalg/adjoint.rs` (generalize `solve_vjp`/`solve_jvp`)
- Modify: `src/linalg/lu.rs` (impl `Factorization for LuFactors`)
- Modify: `src/lib.rs` (export `Factorization`)
- Test: existing `tests/linalg_adjoint.rs` is the pin (unchanged); trait impl exercised via it.

**Interfaces:**
- Consumes: `LuFactors` (with its inherent `dimension`/`solve`/`solve_transposed`).
- Produces:

```rust
pub trait Factorization {
    fn dimension(&self) -> usize;
    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>;
    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError>;
}
```

  and `solve_vjp<F: Factorization>(factors: &F, x: &Vector, x_bar: &Vector) -> Result<SolveGradients, LinalgError>`, `solve_jvp<F: Factorization>(factors: &F, x: &Vector, a_dot: &Matrix, b_dot: &Vector) -> Result<Vector, LinalgError>`. Tasks 5/6 implement this trait for `LltFactors`/`LdltFactors` and get the adjoint rule for free.

- [ ] **Step 1: Write the trait**

Create `src/linalg/factorization.rs`:

```rust
//! The `Factorization` abstraction: any factorization that can solve
//! `A x = b` and `Aᵀ x = b` powers the same adjoint rule
//! (`solve_vjp`/`solve_jvp`) — decision 0003's rule-owning joint,
//! factorization-agnostic by construction.

use crate::core::Vector;

use super::LinalgError;

/// A reusable factorization of a square matrix `A`.
pub trait Factorization {
    /// Side length of the factored matrix.
    fn dimension(&self) -> usize;

    /// Solves `A x = b`.
    ///
    /// # Errors
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length;
    /// [`LinalgError::Singular`] on numerical breakdown.
    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>;

    /// Solves `Aᵀ x = b`. For symmetric factorizations this is `solve`.
    ///
    /// # Errors
    /// Same as [`Factorization::solve`].
    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError>;
}
```

In `src/linalg/mod.rs` add:

```rust
mod factorization;
pub use factorization::Factorization;
```

- [ ] **Step 2: Implement the trait for `LuFactors`**

At the bottom of `src/linalg/lu.rs` (delegating to the inherent methods, which take precedence at call sites — no behavior change):

```rust
impl super::Factorization for LuFactors {
    fn dimension(&self) -> usize {
        Self::dimension(self)
    }

    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve(self, b)
    }

    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve_transposed(self, b)
    }
}
```

- [ ] **Step 3: Generalize the adjoint rules**

In `src/linalg/adjoint.rs`, change the imports and both signatures (bodies are already written against `dimension`/`solve`/`solve_transposed`, so only the parameter type changes):

```rust
use super::{Factorization, LinalgError};
```

```rust
pub fn solve_vjp<F: Factorization>(
    factors: &F,
    x: &Vector,
    x_bar: &Vector,
) -> Result<SolveGradients, LinalgError> {
```

```rust
pub fn solve_jvp<F: Factorization>(
    factors: &F,
    x: &Vector,
    a_dot: &Matrix,
    b_dot: &Vector,
) -> Result<Vector, LinalgError> {
```

Remove the now-unused `use super::LuFactors;` import (it was `use super::{LinalgError, LuFactors};`). Update the module doc comment's reference from "reuse the primal [`LuFactors`]" to "reuse the primal [`Factorization`]".

In `src/lib.rs`, add `Factorization` to the `crate::linalg` re-export list.

- [ ] **Step 4: Run the pins**

Run: `cargo test --release`
Expected: ALL tests pass unchanged — `linalg_adjoint` calls `solve_vjp(&f, ...)` with `f: LuFactors`, which now resolves through the generic. Also run `cargo clippy --release --all-targets` — no new lints.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/factorization.rs src/linalg/mod.rs src/linalg/adjoint.rs src/linalg/lu.rs src/lib.rs
git commit -m "feat: Factorization trait; generalize solve_vjp/solve_jvp over it"
```

---

### Task 5: Cholesky LLT

**Files:**
- Create: `src/linalg/cholesky.rs`
- Modify: `src/linalg/error.rs` (add `NotPositiveDefinite` variant)
- Modify: `src/linalg/mod.rs` (add `mod cholesky; pub use cholesky::{LltFactors, llt_factor};`)
- Modify: `src/lib.rs` (export `LltFactors`, `llt_factor`)
- Test: `tests/linalg_cholesky.rs`

**Interfaces:**
- Consumes: `solve_lower`, `solve_lower_transposed` (Task 2); `Factorization` trait + generic `solve_vjp` (Task 4); `solve_fixed_unchecked` + validation helpers (Phase 2) for the three-way test.
- Produces: `llt_factor(a: &Matrix) -> Result<LltFactors, LinalgError>`; `LltFactors::{dimension() -> usize, solve(&Vector) -> Result<Vector, LinalgError>, l() -> &Matrix}`; `impl Factorization for LltFactors`; `LinalgError::NotPositiveDefinite { pivot_index: usize }`. Task 10 adds `determinant()` to this struct.

- [ ] **Step 1: Add the error variant**

In `src/linalg/error.rs`, add to the enum:

```rust
    /// Cholesky factorization hit a non-positive (or breakdown) pivot.
    NotPositiveDefinite {
        /// Column where factorization broke down.
        pivot_index: usize,
    },
```

and to the `Display` impl:

```rust
            Self::NotPositiveDefinite { pivot_index } => {
                write!(
                    f,
                    "matrix is not positive definite (breakdown at column {pivot_index})"
                )
            }
```

- [ ] **Step 2: Write the failing tests**

Create `tests/linalg_cholesky.rs`:

```rust
#![feature(autodiff)]
// Exact float asserts, tiny index->f64 casts, and short math names are
// intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Cholesky LLT tests: correctness, reconstruction, errors, cross-check vs
//! LU, and the three-way gradient agreement through the generic adjoint rule.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{
    LinalgError, Matrix, SMatrix, SVector, Vector, llt_factor, lu_factor, solve_fixed_unchecked,
    solve_vjp,
};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

/// Symmetric SPD test matrix from 9 raw params:
/// `a_ij = 0.5*(theta[3i+j] + theta[3j+i]) + 5*delta_ij`.
fn spd_from(theta: &[f64]) -> Matrix {
    Matrix::from_fn(3, 3, |i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    })
}

const THETA: [f64; 12] = [
    0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
];

#[test]
fn known_solution_2x2() {
    // A = [[4, 2], [2, 3]], b = [8, 7] => x = [1.25, 1.5]
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let b = Vector::from_slice(&[8.0, 7.0]);
    let x = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    assert!((x[0] - 1.25).abs() < 1e-14);
    assert!((x[1] - 1.5).abs() < 1e-14);
}

#[test]
fn reconstruction_l_lt_matches_a() {
    let a = spd_from(&THETA);
    let f = llt_factor(&a).expect("spd");
    let l = f.l();
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                // L is lower: entry (i, k) is zero for k > i.
                let lik = if k <= i { l[(i, k)] } else { 0.0 };
                let ljk = if k <= j { l[(j, k)] } else { 0.0 };
                acc += lik * ljk;
            }
            assert!(
                (acc - a[(i, j)]).abs() < 1e-12,
                "reconstruction ({i},{j}): {acc} vs {}",
                a[(i, j)]
            );
        }
    }
}

#[test]
fn llt_solve_matches_lu_solve() {
    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let x_llt = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    let x_lu = lu_factor(&a).expect("wc").solve(&b).expect("solve");
    for i in 0..3 {
        assert!((x_llt[i] - x_lu[i]).abs() < 1e-12, "component {i}");
    }
}

#[test]
fn only_lower_triangle_is_read() {
    // Poison the strict upper triangle: result must be identical.
    let a = spd_from(&THETA);
    let poisoned = Matrix::from_fn(3, 3, |i, j| if j > i { f64::NAN } else { a[(i, j)] });
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let x_clean = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    let x_poisoned = llt_factor(&poisoned).expect("spd").solve(&b).expect("solve");
    assert_eq!(x_clean, x_poisoned);
}

#[test]
fn indefinite_matrix_errors() {
    // Eigenvalues 3 and -1: not positive definite.
    let a = Matrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 2.0 });
    assert_eq!(
        llt_factor(&a).map(|_| ()),
        Err(LinalgError::NotPositiveDefinite { pivot_index: 1 })
    );
}

#[test]
fn dimension_mismatches_error() {
    let rect = Matrix::zeros(2, 3);
    assert!(llt_factor(&rect).is_err());
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let f = llt_factor(&a).expect("spd");
    assert!(f.solve(&Vector::zeros(3)).is_err());
}

/// Objective: theta[0..9] -> symmetrized SPD A, theta[9..12] -> b, f = |x|^2.
#[cfg(not(coverage))]
fn objective(theta: &[f64]) -> f64 {
    let a = spd_from(theta);
    let b = Vector::from_slice(&theta[9..12]);
    let x = llt_factor(&a).expect("spd").solve(&b).expect("solve");
    x.norm_squared()
}

/// Same objective as an Enzyme kernel (LU kernel path — mathematically the
/// same function of theta, so gradients must agree with the LLT rule leg).
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    let x = solve_fixed_unchecked(&a, &b);
    *out = x.norm_squared();
}

#[test]
#[cfg(not(coverage))]
fn three_way_gradient_agreement_llt() {
    // (1) finite differences
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");

    // (2) Enzyme
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&THETA)).abs() < 1e-12, "primal mismatch");

    // (3) generic adjoint rule through LltFactors
    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let f = llt_factor(&a).expect("spd");
    let x = f.solve(&b).expect("solve");
    let x_bar = &x * 2.0;
    let grads = solve_vjp(&f, &x, &x_bar).expect("vjp");
    // Chain through the symmetrization: d a_kl / d theta_(3i+j) contributes
    // 0.5*(a_bar[i][j] + a_bar[j][i]) to g[3i+j].
    let mut adjoint = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            adjoint[3 * i + j] = 0.5 * (grads.a_bar[(i, j)] + grads.a_bar[(j, i)]);
        }
        adjoint[9 + i] = grads.b_bar[i];
    }

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs LLT adjoint: {enzyme_vs_adjoint:?}\n enzyme={enzyme:?}\n adjoint={adjoint:?}"
    );
    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "LLT adjoint vs fd: {adjoint_vs_fd:?}"
    );
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test --release --test linalg_cholesky`
Expected: FAIL to compile — `llt_factor` not found.

- [ ] **Step 4: Write the implementation**

Create `src/linalg/cholesky.rs`:

```rust
//! Cholesky factorizations for symmetric positive-definite systems.
//!
//! Only the lower triangle of the input is read (faer's `Side::Lower`
//! convention with the side fixed); symmetry is the caller's contract.
//! Callers never differentiate through this code — derivatives come from
//! the generic adjoint rule over [`Factorization`](super::Factorization).

use crate::core::{Matrix, Vector};

use super::triangular::{solve_lower, solve_lower_transposed};
use super::{LinalgError, PIVOT_TOLERANCE};

/// Reusable LLT factors of an SPD matrix (`A = L Lᵀ`, `L` lower).
#[derive(Debug, Clone)]
pub struct LltFactors {
    /// Lower-triangular factor; the strict upper triangle is zero.
    l: Matrix,
}

/// Factors a symmetric positive-definite matrix as `A = L Lᵀ`.
///
/// Reads only the lower triangle of `a`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] for non-square input;
/// [`LinalgError::NotPositiveDefinite`] when a pivot `l_jj²` falls at or
/// below tolerance (the matrix is indefinite or numerically singular).
pub fn llt_factor(a: &Matrix) -> Result<LltFactors, LinalgError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let mut l = Matrix::zeros(n, n);
    for j in 0..n {
        // Diagonal: l_jj = sqrt(a_jj - sum_k l_jk^2).
        let mut sum = a[(j, j)];
        for k in 0..j {
            sum -= l[(j, k)] * l[(j, k)];
        }
        if sum <= PIVOT_TOLERANCE {
            return Err(LinalgError::NotPositiveDefinite { pivot_index: j });
        }
        let ljj = sum.sqrt();
        l[(j, j)] = ljj;
        // Below-diagonal column j.
        for i in (j + 1)..n {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / ljj;
        }
    }
    Ok(LltFactors { l })
}

impl LltFactors {
    /// Side length of the factored matrix.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.l.rows()
    }

    /// The lower-triangular factor `L`.
    #[must_use]
    pub const fn l(&self) -> &Matrix {
        &self.l
    }

    /// Solves `A x = b` via `L y = b`, then `Lᵀ x = y`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let y = solve_lower(&self.l, b, false)?;
        solve_lower_transposed(&self.l, &y, false)
    }
}

impl super::Factorization for LltFactors {
    fn dimension(&self) -> usize {
        Self::dimension(self)
    }

    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve(self, b)
    }

    /// `A` is symmetric, so `Aᵀ x = b` is the same solve.
    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve(self, b)
    }
}
```

Note: `Matrix::rows()` is `const fn`, so `dimension`/`l` can be `const fn`. If clippy's `missing_const_for_fn` fires differently, follow clippy.

In `src/linalg/mod.rs`:

```rust
mod cholesky;
pub use cholesky::{LltFactors, llt_factor};
```

In `src/lib.rs`, add `LltFactors, llt_factor` to the `crate::linalg` re-export.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --release --test linalg_cholesky`
Expected: PASS (7 tests, including the three-way agreement). Then `cargo test --release` and `cargo clippy --release --all-targets`.

- [ ] **Step 6: Commit**

```bash
git add src/linalg/cholesky.rs src/linalg/error.rs src/linalg/mod.rs src/lib.rs tests/linalg_cholesky.rs
git commit -m "feat: Cholesky LLT with Factorization impl and three-way gradient test"
```

---

### Task 6: Cholesky LDLT (unpivoted)

**Files:**
- Modify: `src/linalg/cholesky.rs` (add LDLT below LLT)
- Modify: `src/linalg/mod.rs`, `src/lib.rs` (export `LdltFactors`, `ldlt_factor`)
- Test: append to `tests/linalg_cholesky.rs`

**Interfaces:**
- Consumes: `solve_lower`, `solve_lower_transposed` (unit-diag variants), `Factorization`.
- Produces: `ldlt_factor(a: &Matrix) -> Result<LdltFactors, LinalgError>`; `LdltFactors::{dimension(), solve(), l() -> &Matrix (unit-lower), d() -> &Vector}`; `impl Factorization for LdltFactors`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/linalg_cholesky.rs` (add `ldlt_factor` to the existing `use mercury::{...}` list):

```rust
#[test]
fn ldlt_known_solution_2x2() {
    // A = [[4, 2], [2, 3]]: d = [4, 2], l10 = 0.5.
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let f = ldlt_factor(&a).expect("factor");
    assert!((f.d()[0] - 4.0).abs() < 1e-14);
    assert!((f.d()[1] - 2.0).abs() < 1e-14);
    assert!((f.l()[(1, 0)] - 0.5).abs() < 1e-14);
    let b = Vector::from_slice(&[8.0, 7.0]);
    let x = f.solve(&b).expect("solve");
    assert!((x[0] - 1.25).abs() < 1e-14);
    assert!((x[1] - 1.5).abs() < 1e-14);
}

#[test]
fn ldlt_handles_indefinite_and_matches_lu() {
    // Eigenvalues 3 and -1: indefinite, LLT fails but unpivoted LDLT works.
    let a = Matrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 2.0 });
    let b = Vector::from_slice(&[1.0, -1.0]);
    let x_ldlt = ldlt_factor(&a).expect("factor").solve(&b).expect("solve");
    let x_lu = lu_factor(&a).expect("wc").solve(&b).expect("solve");
    for i in 0..2 {
        assert!((x_ldlt[i] - x_lu[i]).abs() < 1e-12, "component {i}");
    }
}

#[test]
fn ldlt_breakdown_errors() {
    // Second pivot d_1 = 1 - 1*1*1 = 0: breakdown.
    let a = Matrix::from_fn(2, 2, |_, _| 1.0);
    assert_eq!(
        ldlt_factor(&a).map(|_| ()),
        Err(LinalgError::NotPositiveDefinite { pivot_index: 1 })
    );
}

#[test]
#[cfg(not(coverage))]
fn ldlt_adjoint_matches_enzyme_and_fd() {
    // Same objective/kernel as the LLT three-way test; only the rule leg
    // changes factorization backend. Gradients must be identical.
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);

    let a = spd_from(&THETA);
    let b = Vector::from_slice(&THETA[9..12]);
    let f = ldlt_factor(&a).expect("factor");
    let x = f.solve(&b).expect("solve");
    let x_bar = &x * 2.0;
    let grads = solve_vjp(&f, &x, &x_bar).expect("vjp");
    let mut adjoint = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            adjoint[3 * i + j] = 0.5 * (grads.a_bar[(i, j)] + grads.a_bar[(j, i)]);
        }
        adjoint[9 + i] = grads.b_bar[i];
    }

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs LDLT adjoint: {enzyme_vs_adjoint:?}"
    );
    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "LDLT adjoint vs fd: {adjoint_vs_fd:?}"
    );
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release --test linalg_cholesky`
Expected: FAIL to compile — `ldlt_factor` not found.

- [ ] **Step 3: Write the implementation**

Append to `src/linalg/cholesky.rs`:

```rust
/// Reusable LDLT factors of a symmetric matrix (`A = L D Lᵀ`, `L`
/// unit-lower, `D` diagonal).
///
/// Unpivoted: exact for any symmetric matrix whose leading principal minors
/// are nonsingular, and tolerant of indefinite `D`, but can be inaccurate on
/// strongly indefinite matrices — Bunch-Kaufman pivoting is the deferred
/// answer for those (spec non-goal).
#[derive(Debug, Clone)]
pub struct LdltFactors {
    /// Unit-lower factor; multipliers strictly below the diagonal (the
    /// diagonal itself is implied 1 and the stored values are unused).
    l: Matrix,
    /// Diagonal of `D`.
    d: Vector,
}

/// Factors a symmetric matrix as `A = L D Lᵀ` without pivoting.
///
/// Reads only the lower triangle of `a`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] for non-square input;
/// [`LinalgError::NotPositiveDefinite`] when `|d_j|` falls at or below
/// tolerance (breakdown of the unpivoted recurrence).
pub fn ldlt_factor(a: &Matrix) -> Result<LdltFactors, LinalgError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let mut l = Matrix::zeros(n, n);
    let mut d = Vector::zeros(n);
    for j in 0..n {
        // d_j = a_jj - sum_k l_jk^2 d_k.
        let mut dj = a[(j, j)];
        for k in 0..j {
            dj -= l[(j, k)] * l[(j, k)] * d[k];
        }
        if dj.abs() <= PIVOT_TOLERANCE {
            return Err(LinalgError::NotPositiveDefinite { pivot_index: j });
        }
        d[j] = dj;
        // l_ij = (a_ij - sum_k l_ik l_jk d_k) / d_j.
        for i in (j + 1)..n {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)] * d[k];
            }
            l[(i, j)] = s / dj;
        }
    }
    Ok(LdltFactors { l, d })
}

impl LdltFactors {
    /// Side length of the factored matrix.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.l.rows()
    }

    /// The unit-lower factor `L` (diagonal implied 1).
    #[must_use]
    pub const fn l(&self) -> &Matrix {
        &self.l
    }

    /// The diagonal of `D`.
    #[must_use]
    pub const fn d(&self) -> &Vector {
        &self.d
    }

    /// Solves `A x = b` via `L z = b`, `D w = z`, `Lᵀ x = w`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let n = self.dimension();
        let z = solve_lower(&self.l, b, true)?;
        let mut w = Vector::zeros(n);
        for i in 0..n {
            w[i] = z[i] / self.d[i];
        }
        solve_lower_transposed(&self.l, &w, true)
    }
}

impl super::Factorization for LdltFactors {
    fn dimension(&self) -> usize {
        Self::dimension(self)
    }

    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve(self, b)
    }

    /// `A` is symmetric, so `Aᵀ x = b` is the same solve.
    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError> {
        Self::solve(self, b)
    }
}
```

Update exports: `pub use cholesky::{LdltFactors, LltFactors, ldlt_factor, llt_factor};` in `src/linalg/mod.rs`; mirror in `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test linalg_cholesky`
Expected: PASS (11 tests). Then the full suite + clippy.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/cholesky.rs src/linalg/mod.rs src/lib.rs tests/linalg_cholesky.rs
git commit -m "feat: unpivoted LDLT with Factorization impl and adjoint agreement test"
```

---

### Task 7: Kernel-side `solve_spd_fixed_unchecked`

**Files:**
- Modify: `src/linalg/fixed.rs` (append the new function)
- Modify: `src/linalg/mod.rs`, `src/lib.rs` (export)
- Test: `tests/linalg_spd_fixed.rs`

**Interfaces:**
- Consumes: `SMatrix`, `SVector` and their kernel-safe `from_fn`/`Index`/`IndexMut` (Phase 2).
- Produces: `solve_spd_fixed_unchecked<const N: usize>(a: &SMatrix<N, N>, b: &SVector<N>) -> SVector<N>` — kernel-safe, no `Result`, NaN-propagation on non-SPD input. Task 9's Enzyme test leg differentiates through this.

**Enzyme constraints (Global Constraints rules 1–8 in full force):** plain `for` loops only, `from_fn` construction, mutation through `IndexMut`, no `Result`, no bulk copies/swaps. Model the code on the existing `solve_fixed_unchecked` in the same file.

- [ ] **Step 1: Write the failing tests**

Create `tests/linalg_spd_fixed.rs`:

```rust
#![feature(autodiff)]
// Exact float asserts, tiny index->f64 casts, and short math names are
// intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Three-legged tests for the kernel-safe SPD solve: primal correctness vs
//! LU, NaN propagation on non-SPD input, Enzyme compile + gradient leg, and
//! FD cross-check.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{Matrix, SMatrix, SVector, Vector, lu_factor, solve_spd_fixed_unchecked};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

const DIAG_SHIFT: f64 = 5.0;

const THETA: [f64; 12] = [
    0.7, -0.3, 0.2, 0.1, 0.9, -0.4, -0.2, 0.5, 0.6, 1.0, -2.0, 0.5,
];

fn spd_smatrix(theta: &[f64]) -> SMatrix<3, 3> {
    SMatrix::<3, 3>::from_fn(|i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    })
}

#[test]
fn primal_matches_lu_solve() {
    let a = spd_smatrix(&THETA);
    let b = SVector::<3>::from_fn(|i| THETA[9 + i]);
    let x = solve_spd_fixed_unchecked(&a, &b);

    let a_dyn = Matrix::from_fn(3, 3, |i, j| a[(i, j)]);
    let b_dyn = Vector::from_slice(&THETA[9..12]);
    let x_lu = lu_factor(&a_dyn).expect("wc").solve(&b_dyn).expect("wc");
    for i in 0..3 {
        assert!((x[i] - x_lu[i]).abs() < 1e-12, "component {i}");
    }
}

#[test]
fn non_spd_propagates_nan() {
    // Indefinite: sqrt of a negative pivot must produce NaN, not panic.
    let a = SMatrix::<2, 2>::from_fn(|i, j| if i == j { 1.0 } else { 2.0 });
    let b = SVector::<2>::from_fn(|_| 1.0);
    let x = solve_spd_fixed_unchecked(&a, &b);
    assert!(x[0].is_nan() || x[1].is_nan(), "expected NaN propagation");
}

/// Objective f = |x|^2 through the SPD kernel solve.
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    let a = SMatrix::<3, 3>::from_fn(|i, j| {
        0.5 * (theta[3 * i + j] + theta[3 * j + i]) + if i == j { DIAG_SHIFT } else { 0.0 }
    });
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    let x = solve_spd_fixed_unchecked(&a, &b);
    *out = x.norm_squared();
}

#[cfg(not(coverage))]
fn objective(theta: &[f64]) -> f64 {
    let a = spd_smatrix(theta);
    let b = SVector::<3>::from_fn(|i| theta[9 + i]);
    let x = solve_spd_fixed_unchecked(&a, &b);
    x.norm_squared()
}

#[test]
#[cfg(not(coverage))]
fn enzyme_gradient_matches_finite_differences() {
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");
    let mut enzyme = vec![0.0; 12];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&THETA)).abs() < 1e-12, "primal mismatch");
    let cmp = compare_gradients(&enzyme, &fd).expect("shape");
    assert!(cmp.max_abs_error < 1.0e-4, "Enzyme vs fd: {cmp:?}");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release --test linalg_spd_fixed`
Expected: FAIL to compile — `solve_spd_fixed_unchecked` not found.

- [ ] **Step 3: Write the implementation**

Append to `src/linalg/fixed.rs`:

```rust
/// Solves SPD `A x = b` for small fixed-size systems on the stack via
/// unpivoted Cholesky (LLT), without a definiteness check.
///
/// Kernel-facing (same contract family as [`solve_fixed_unchecked`]): no
/// `Result` return, written in the Enzyme-safe POD style. Cheaper than the
/// LU kernel path for SPD systems — no pivot search. Reads only the lower
/// triangle of `a`; symmetry is the caller's contract.
///
/// If `A` is not positive definite, the factorization takes the square
/// root of a non-positive number and the affected outputs become `NaN`,
/// propagating through the solves. Callers needing a hard error should
/// factor host-side with [`llt_factor`](crate::linalg::llt_factor).
#[must_use]
pub fn solve_spd_fixed_unchecked<const N: usize>(
    a: &SMatrix<N, N>,
    b: &SVector<N>,
) -> SVector<N> {
    // Working copy in the Enzyme-safe shape (Global Constraints rules 1-4).
    let mut l = SMatrix::<N, N>::from_fn(|i, j| a[(i, j)]);

    // In-place LLT on the lower triangle.
    for j in 0..N {
        let mut sum = l[(j, j)];
        for k in 0..j {
            sum -= l[(j, k)] * l[(j, k)];
        }
        let ljj = sum.sqrt(); // non-SPD => NaN, propagated by design
        l[(j, j)] = ljj;
        for i in (j + 1)..N {
            let mut s = l[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / ljj;
        }
    }

    // Forward: L y = b.
    let mut y = SVector::<N>::from_fn(|i| b[i]);
    for i in 0..N {
        let mut acc = y[i];
        for j in 0..i {
            acc -= l[(i, j)] * y[j];
        }
        y[i] = acc / l[(i, i)];
    }
    // Backward: Lᵀ x = y, in place over y.
    for i in (0..N).rev() {
        let mut acc = y[i];
        for j in (i + 1)..N {
            acc -= l[(j, i)] * y[j];
        }
        y[i] = acc / l[(i, i)];
    }
    y
}
```

Update `src/linalg/mod.rs`: `pub use fixed::{solve_fixed, solve_fixed_unchecked, solve_spd_fixed_unchecked};` and mirror in `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test linalg_spd_fixed`
Expected: PASS (3 tests). If Enzyme fails to compile the kernel ("Cannot deduce type" or similar), the implementation drifted from rules 1–7 — compare against `solve_fixed_unchecked` in the same file and fix the shape, do NOT reach for LooseTypes (banned). Then full suite + clippy.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/fixed.rs src/linalg/mod.rs src/lib.rs tests/linalg_spd_fixed.rs
git commit -m "feat: kernel-safe SPD solve (unpivoted LLT) with three-legged tests"
```

---

### Task 8: Householder QR + least squares

**Files:**
- Create: `src/linalg/qr.rs`
- Modify: `src/linalg/error.rs` (add `RankDeficient` variant)
- Modify: `src/linalg/mod.rs`, `src/lib.rs` (export `QrFactors`, `qr_factor`)
- Test: `tests/linalg_qr.rs`

**Interfaces:**
- Consumes: `Matrix`, `Vector`, `solve_upper` (Task 2).
- Produces: `qr_factor(a: &Matrix) -> Result<QrFactors, LinalgError>` (requires `m >= n`); `QrFactors::{rows(), cols(), solve(&Vector) -> Result<Vector, _>` (square only), `solve_lstsq(&Vector) -> Result<Vector, _>` (m ≥ n), `q_transpose_apply(&Vector) -> Result<Vector, _>`, `r() -> Matrix` (n×n upper copy)}; `LinalgError::RankDeficient { column: usize }`. Task 9 uses `r()` and `solve_lstsq`.

**Algorithm (LAPACK-style compact Householder):** for each column `k`, with `x = column k, rows k..m`: `norm = sqrt(x₀² + Σᵢ₍ᵢ₎ xᵢ²)`; `beta = -sign(x₀)·norm` (r_kk, stable against cancellation); reflector `v` normalized to `v₀ = 1`: `vᵢ = xᵢ/(x₀ − beta)`; `tau = (beta − x₀)/beta`; `H = I − tau·v·vᵀ` gives `Hx = beta·e₁`. Store `beta` at `(k,k)`, `vᵢ` below it, `tau` in a side vector; apply `H` to trailing columns.

- [ ] **Step 1: Add the error variant**

In `src/linalg/error.rs`:

```rust
    /// QR factorization found a (numerically) rank-deficient column.
    RankDeficient {
        /// Column whose diagonal of `R` fell below tolerance.
        column: usize,
    },
```

Display arm:

```rust
            Self::RankDeficient { column } => {
                write!(f, "matrix is rank deficient (column {column})")
            }
```

- [ ] **Step 2: Write the failing tests**

Create `tests/linalg_qr.rs`:

```rust
// Exact float asserts, tiny index->f64 casts, and short math names are
// intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Householder QR tests: square solve vs LU, least squares via normal
//! equations, orthogonality, R correctness, and error paths.

use mercury::{LinalgError, Matrix, Vector, lu_factor, qr_factor};

fn tall_a() -> Matrix {
    // 4x2 full-rank.
    Matrix::from_fn(4, 2, |i, j| {
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]][i][j]
    })
}

#[test]
fn square_solve_matches_lu() {
    let a = Matrix::from_fn(3, 3, |i, j| {
        [[2.0, 1.0, -0.5], [0.3, 3.0, 0.2], [-0.1, 0.4, 1.5]][i][j]
    });
    let b = Vector::from_slice(&[1.0, -2.0, 0.5]);
    let x_qr = qr_factor(&a).expect("qr").solve(&b).expect("solve");
    let x_lu = lu_factor(&a).expect("wc").solve(&b).expect("wc");
    for i in 0..3 {
        assert!((x_qr[i] - x_lu[i]).abs() < 1e-12, "component {i}");
    }
}

#[test]
fn lstsq_satisfies_normal_equations() {
    // Fit y = c0 + c1*t to noisy points: residual must be A-orthogonal.
    let a = tall_a();
    let b = Vector::from_slice(&[2.1, 3.9, 6.2, 7.8]);
    let x = qr_factor(&a).expect("qr").solve_lstsq(&b).expect("lstsq");
    // r = b - A x; check Aᵀ r ≈ 0.
    let ax = &a * &x;
    let r = &b - &ax;
    for j in 0..2 {
        let mut dot = 0.0;
        for i in 0..4 {
            dot += a[(i, j)] * r[i];
        }
        assert!(dot.abs() < 1e-12, "normal equation {j}: {dot}");
    }
}

#[test]
fn lstsq_exact_fit_recovers_coefficients() {
    // b generated exactly from c = [1.0, 0.5]: lstsq must recover it.
    let a = tall_a();
    let b = Vector::from_slice(&[1.5, 2.0, 2.5, 3.0]);
    let x = qr_factor(&a).expect("qr").solve_lstsq(&b).expect("lstsq");
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 0.5).abs() < 1e-12);
}

#[test]
fn q_transpose_apply_preserves_norm() {
    // Qᵀ is orthogonal: ‖Qᵀb‖ = ‖b‖.
    let a = tall_a();
    let f = qr_factor(&a).expect("qr");
    let b = Vector::from_slice(&[1.0, -2.0, 0.5, 3.0]);
    let qtb = f.q_transpose_apply(&b).expect("apply");
    assert_eq!(qtb.len(), 4);
    assert!((qtb.norm_squared() - b.norm_squared()).abs() < 1e-12);
}

#[test]
fn r_transpose_r_equals_a_transpose_a() {
    // AᵀA = RᵀR pins R without exposing Q.
    let a = tall_a();
    let r = qr_factor(&a).expect("qr").r();
    for p in 0..2 {
        for q in 0..2 {
            let mut ata = 0.0;
            for i in 0..4 {
                ata += a[(i, p)] * a[(i, q)];
            }
            let mut rtr = 0.0;
            for i in 0..2 {
                rtr += r[(i, p)] * r[(i, q)];
            }
            assert!((ata - rtr).abs() < 1e-12, "({p},{q}): {ata} vs {rtr}");
        }
    }
}

#[test]
fn rank_deficient_errors() {
    // Second column is 2x the first: rank 1.
    let a = Matrix::from_fn(3, 2, |i, j| {
        let base = (i + 1) as f64;
        if j == 0 { base } else { 2.0 * base }
    });
    assert_eq!(
        qr_factor(&a).map(|_| ()),
        Err(LinalgError::RankDeficient { column: 1 })
    );
}

#[test]
fn dimension_errors() {
    // Underdetermined (m < n) rejected at factor time.
    let wide = Matrix::zeros(2, 3);
    assert!(qr_factor(&wide).is_err());
    // solve() requires square.
    let f = qr_factor(&tall_a()).expect("qr");
    let b4 = Vector::zeros(4);
    assert!(f.solve(&b4).is_err());
    // wrong b length for lstsq / q_transpose_apply.
    let b3 = Vector::zeros(3);
    assert!(f.solve_lstsq(&b3).is_err());
    assert!(f.q_transpose_apply(&b3).is_err());
}
```

(Note: delete the dead first `let a = ...transpose().transpose();` binding in `rank_deficient_errors` — it is shadowed; write only the second binding. Clippy would flag it.)

- [ ] **Step 3: Run to verify failure**

Run: `cargo test --release --test linalg_qr`
Expected: FAIL to compile — `qr_factor` not found.

- [ ] **Step 4: Write the implementation**

Create `src/linalg/qr.rs`:

```rust
//! Householder QR factorization with least-squares solve.
//!
//! LAPACK-style compact storage: reflectors below the diagonal, `R` on and
//! above it, `tau` coefficients in a side vector. Callers never
//! differentiate through this code — least-squares derivatives come from
//! the dedicated adjoint rule (`lstsq_vjp`/`lstsq_jvp`).

use crate::core::{Matrix, Vector};

use super::triangular::solve_upper;
use super::{LinalgError, PIVOT_TOLERANCE};

/// Reusable Householder QR factors of an `m x n` matrix, `m >= n`.
#[derive(Debug, Clone)]
pub struct QrFactors {
    /// Compact storage: `R` on/above the diagonal, reflector tails below.
    qr: Matrix,
    /// Householder coefficients, one per column.
    tau: Vector,
}

/// Factors `A = Q R` by Householder reflections. Requires `m >= n`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when `m < n`;
/// [`LinalgError::RankDeficient`] when a diagonal of `R` falls below
/// tolerance (numerically dependent columns).
pub fn qr_factor(a: &Matrix) -> Result<QrFactors, LinalgError> {
    let m = a.rows();
    let n = a.cols();
    if m < n {
        return Err(LinalgError::DimensionMismatch {
            rows: m,
            cols: n,
        });
    }

    let mut qr = a.clone();
    let mut tau = Vector::zeros(n);

    for k in 0..n {
        // Column norm over rows k..m.
        let x0 = qr[(k, k)];
        let mut sigma = 0.0;
        for i in (k + 1)..m {
            sigma += qr[(i, k)] * qr[(i, k)];
        }
        let norm = (x0 * x0 + sigma).sqrt();
        if norm < PIVOT_TOLERANCE {
            return Err(LinalgError::RankDeficient { column: k });
        }
        // beta = -sign(x0) * norm avoids cancellation in x0 - beta.
        let beta = if x0 >= 0.0 { -norm } else { norm };
        let tau_k = (beta - x0) / beta;
        let denom = x0 - beta;
        // Store the reflector tail (v0 = 1 implied) and R's diagonal.
        for i in (k + 1)..m {
            qr[(i, k)] /= denom;
        }
        qr[(k, k)] = beta;
        tau[k] = tau_k;

        // Apply H = I - tau v v^T to the trailing columns.
        for j in (k + 1)..n {
            let mut w = qr[(k, j)];
            for i in (k + 1)..m {
                w += qr[(i, k)] * qr[(i, j)];
            }
            let tw = tau_k * w;
            qr[(k, j)] -= tw;
            for i in (k + 1)..m {
                let delta = tw * qr[(i, k)];
                qr[(i, j)] -= delta;
            }
        }
    }

    Ok(QrFactors { qr, tau })
}

impl QrFactors {
    /// Row count of the factored matrix.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.qr.rows()
    }

    /// Column count of the factored matrix.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.qr.cols()
    }

    /// The `n x n` upper-triangular factor `R` (thin), as a copy.
    #[must_use]
    pub fn r(&self) -> Matrix {
        let n = self.cols();
        Matrix::from_fn(n, n, |i, j| if j >= i { self.qr[(i, j)] } else { 0.0 })
    }

    /// Applies `Qᵀ` to a length-`m` vector via the stored reflectors.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn q_transpose_apply(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let m = self.rows();
        let n = self.cols();
        if b.len() != m {
            return Err(LinalgError::DimensionMismatch {
                rows: b.len(),
                cols: 1,
            });
        }
        let mut y = b.clone();
        for k in 0..n {
            let mut w = y[k];
            for i in (k + 1)..m {
                w += self.qr[(i, k)] * y[i];
            }
            let tw = self.tau[k] * w;
            y[k] -= tw;
            for i in (k + 1)..m {
                let delta = tw * self.qr[(i, k)];
                y[i] -= delta;
            }
        }
        Ok(y)
    }

    /// Solves square `A x = b` via `R x = Qᵀ b`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when the factored matrix is not
    /// square or `b` has the wrong length.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let m = self.rows();
        let n = self.cols();
        if m != n {
            return Err(LinalgError::DimensionMismatch { rows: m, cols: n });
        }
        let y = self.q_transpose_apply(b)?;
        solve_upper(&self.qr, &y, false)
    }

    /// Solves the full-rank least-squares problem `min ‖A x − b‖₂`
    /// via `R x = (Qᵀ b)[0..n]`.
    ///
    /// # Errors
    ///
    /// [`LinalgError::DimensionMismatch`] when `b` has the wrong length.
    pub fn solve_lstsq(&self, b: &Vector) -> Result<Vector, LinalgError> {
        let n = self.cols();
        let y = self.q_transpose_apply(b)?;
        // Back-substitute R x = y[0..n] directly on the compact storage
        // (R's rows live in the top n rows of qr).
        let mut x = Vector::zeros(n);
        for i in (0..n).rev() {
            let mut acc = y[i];
            for j in (i + 1)..n {
                acc -= self.qr[(i, j)] * x[j];
            }
            x[i] = acc / self.qr[(i, i)];
        }
        Ok(x)
    }
}
```

(No `Factorization` impl: QR's `solve_transposed` would need `Q`, and the square-solve adjoint already has LU/LLT/LDLT backends — YAGNI.)

Update `src/linalg/mod.rs`:

```rust
mod qr;
pub use qr::{QrFactors, qr_factor};
```

Mirror in `src/lib.rs`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --release --test linalg_qr`
Expected: PASS (7 tests). Full suite + clippy.

- [ ] **Step 6: Commit**

```bash
git add src/linalg/qr.rs src/linalg/error.rs src/linalg/mod.rs src/lib.rs tests/linalg_qr.rs
git commit -m "feat: Householder QR with square solve and full-rank least squares"
```

---

### Task 9: Least-squares adjoint rule

**Files:**
- Modify: `src/linalg/adjoint.rs` (append `lstsq_vjp`/`lstsq_jvp`)
- Modify: `src/linalg/mod.rs`, `src/lib.rs` (export)
- Test: `tests/linalg_lstsq_adjoint.rs`

**Interfaces:**
- Consumes: `QrFactors::r()` (Task 8), `solve_upper`/`solve_upper_transposed` (Task 2), `SolveGradients` (Phase 2), `solve_spd_fixed_unchecked` (Task 7) for the Enzyme leg, `Matrix::transpose`/`Mul` operators.
- Produces:
  - `lstsq_vjp(factors: &QrFactors, a: &Matrix, b: &Vector, x: &Vector, x_bar: &Vector) -> Result<SolveGradients, LinalgError>`
  - `lstsq_jvp(factors: &QrFactors, a: &Matrix, b: &Vector, x: &Vector, a_dot: &Matrix, b_dot: &Vector) -> Result<Vector, LinalgError>`

**Mathematics (from the spec):** for full-column-rank `x = argmin ‖Ax − b‖₂`, residual `r = b − Ax`, differentiate the normal equations `AᵀA x = Aᵀb`:
- JVP: `ẋ = (AᵀA)⁻¹ (Ȧᵀ r − Aᵀ Ȧ x + Aᵀ ḃ)`
- VJP: with `z = (AᵀA)⁻¹ x̄`: `b̄ = A z`, `Ā = −(A z) xᵀ + r zᵀ`
- `(AᵀA)⁻¹ w` = `R⁻¹ R⁻ᵀ w` — two triangular solves against `R`; never form `AᵀA`.

- [ ] **Step 1: Write the failing test**

Create `tests/linalg_lstsq_adjoint.rs`:

```rust
#![feature(autodiff)]
// Exact float asserts, tiny index->f64 casts, and short math names are
// intentional in tests.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

//! Least-squares adjoint rule: three-way agreement on a 5x3 full-rank
//! problem. The Enzyme leg solves the normal equations `(AᵀA)x = Aᵀb` with
//! the kernel-safe SPD solve — mathematically the same function of theta as
//! the host-side QR least-squares solve.

use mercury::validation::{central_difference_gradient, compare_gradients};
use mercury::{
    Matrix, SMatrix, SVector, Vector, lstsq_jvp, lstsq_vjp, qr_factor, solve_spd_fixed_unchecked,
};
#[cfg(not(coverage))]
use std::autodiff::autodiff_reverse;

// theta[0..15]: A (5x3, row-major, offset to be well-conditioned);
// theta[15..20]: b.
const THETA: [f64; 20] = [
    1.0, 0.2, -0.1, 0.3, 1.5, 0.4, -0.2, 0.1, 2.0, 0.5, -0.3, 0.2, 0.7, 0.6, 1.1, 2.1, -0.4, 0.9,
    1.3, -1.7,
];

fn a_from(theta: &[f64]) -> Matrix {
    Matrix::from_fn(5, 3, |i, j| theta[3 * i + j])
}

fn b_from(theta: &[f64]) -> Vector {
    Vector::from_slice(&theta[15..20])
}

/// Host objective f = |x|^2 through QR least squares.
fn objective(theta: &[f64]) -> f64 {
    let a = a_from(theta);
    let b = b_from(theta);
    let x = qr_factor(&a).expect("qr").solve_lstsq(&b).expect("lstsq");
    x.norm_squared()
}

/// Same objective as an Enzyme kernel via the normal equations.
#[cfg(not(coverage))]
#[autodiff_reverse(d_kernel, Duplicated, Duplicated)]
fn kernel(theta: &[f64], out: &mut f64) {
    // G = AᵀA (3x3), rhs = Aᵀb — both built element-wise from theta.
    let g = SMatrix::<3, 3>::from_fn(|p, q| {
        let mut acc = 0.0;
        for i in 0..5 {
            acc += theta[3 * i + p] * theta[3 * i + q];
        }
        acc
    });
    let rhs = SVector::<3>::from_fn(|p| {
        let mut acc = 0.0;
        for i in 0..5 {
            acc += theta[3 * i + p] * theta[15 + i];
        }
        acc
    });
    let x = solve_spd_fixed_unchecked(&g, &rhs);
    *out = x.norm_squared();
}

#[test]
#[cfg(not(coverage))]
fn three_way_gradient_agreement_lstsq() {
    // (1) finite differences
    let fd = central_difference_gradient(objective, &THETA, 1.0e-6).expect("fd");

    // (2) Enzyme through the normal-equations kernel
    let mut enzyme = vec![0.0; 20];
    let (mut out, mut dout) = (0.0, 1.0);
    d_kernel(&THETA, &mut enzyme, &mut out, &mut dout);
    assert!((out - objective(&THETA)).abs() < 1e-10, "primal mismatch");

    // (3) the least-squares adjoint rule
    let a = a_from(&THETA);
    let b = b_from(&THETA);
    let f = qr_factor(&a).expect("qr");
    let x = f.solve_lstsq(&b).expect("lstsq");
    let x_bar = &x * 2.0;
    let grads = lstsq_vjp(&f, &a, &b, &x, &x_bar).expect("vjp");
    let mut adjoint = vec![0.0; 20];
    for i in 0..5 {
        for j in 0..3 {
            adjoint[3 * i + j] = grads.a_bar[(i, j)];
        }
        adjoint[15 + i] = grads.b_bar[i];
    }

    let enzyme_vs_adjoint = compare_gradients(&enzyme, &adjoint).expect("shape");
    assert!(
        enzyme_vs_adjoint.max_abs_error < 1.0e-9,
        "Enzyme vs lstsq adjoint: {enzyme_vs_adjoint:?}\n enzyme={enzyme:?}\n adjoint={adjoint:?}"
    );
    let adjoint_vs_fd = compare_gradients(&adjoint, &fd).expect("shape");
    assert!(
        adjoint_vs_fd.max_abs_error < 1.0e-4,
        "lstsq adjoint vs fd: {adjoint_vs_fd:?}"
    );
}

#[test]
fn jvp_matches_directional_finite_difference() {
    let a = a_from(&THETA);
    let b = b_from(&THETA);
    let f = qr_factor(&a).expect("qr");
    let x = f.solve_lstsq(&b).expect("lstsq");

    let a_dot = Matrix::from_fn(5, 3, |i, j| 0.05 * ((i + 2 * j) as f64) - 0.1);
    let b_dot = Vector::from_slice(&[0.3, -0.1, 0.2, 0.05, -0.25]);
    let x_dot = lstsq_jvp(&f, &a, &b, &x, &a_dot, &b_dot).expect("jvp");

    let h = 1.0e-7;
    let a_p = Matrix::from_fn(5, 3, |i, j| a[(i, j)] + h * a_dot[(i, j)]);
    let a_m = Matrix::from_fn(5, 3, |i, j| a[(i, j)] - h * a_dot[(i, j)]);
    let mut bp = Vec::new();
    let mut bm = Vec::new();
    for i in 0..5 {
        bp.push(b[i] + h * b_dot[i]);
        bm.push(b[i] - h * b_dot[i]);
    }
    let x_p = qr_factor(&a_p)
        .expect("qr")
        .solve_lstsq(&Vector::from_vec(bp))
        .expect("lstsq");
    let x_m = qr_factor(&a_m)
        .expect("qr")
        .solve_lstsq(&Vector::from_vec(bm))
        .expect("lstsq");
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
fn lstsq_vjp_dimension_mismatches_error() {
    let a = a_from(&THETA);
    let b = b_from(&THETA);
    let f = qr_factor(&a).expect("qr");
    let x = f.solve_lstsq(&b).expect("lstsq");
    let x_bar = &x * 2.0;
    // wrong x_bar length
    assert!(lstsq_vjp(&f, &a, &b, &x, &Vector::zeros(4)).is_err());
    // wrong x length
    assert!(lstsq_vjp(&f, &a, &b, &Vector::zeros(4), &x_bar).is_err());
    // wrong b length
    assert!(lstsq_vjp(&f, &a, &Vector::zeros(4), &x, &x_bar).is_err());
    // wrong a shape
    assert!(lstsq_vjp(&f, &Matrix::zeros(5, 2), &b, &x, &x_bar).is_err());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release --test linalg_lstsq_adjoint`
Expected: FAIL to compile — `lstsq_vjp` not found.

- [ ] **Step 3: Write the implementation**

Append to `src/linalg/adjoint.rs` (add imports: `use super::QrFactors;` and `use super::triangular::{solve_upper, solve_upper_transposed};`):

```rust
/// Validates shapes shared by the least-squares rules; returns `(m, n)`.
fn check_lstsq_shapes(
    factors: &QrFactors,
    a: &Matrix,
    b: &Vector,
    x: &Vector,
) -> Result<(usize, usize), LinalgError> {
    let m = factors.rows();
    let n = factors.cols();
    if a.rows() != m || a.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a.rows(),
            cols: a.cols(),
        });
    }
    if b.len() != m {
        return Err(LinalgError::DimensionMismatch {
            rows: b.len(),
            cols: 1,
        });
    }
    if x.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: x.len(),
            cols: 1,
        });
    }
    Ok((m, n))
}

/// Solves `(AᵀA) out = w` as `R⁻¹ R⁻ᵀ w` — two triangular solves against
/// the QR factor, never forming `AᵀA`.
fn solve_normal(r: &Matrix, w: &Vector) -> Result<Vector, LinalgError> {
    let t = solve_upper_transposed(r, w, false)?;
    solve_upper(r, &t, false)
}

/// Reverse-mode rule for full-rank least squares `x = argmin ‖A x − b‖₂`.
///
/// With residual `r = b − A x` and `z = (AᵀA)⁻¹ x̄`:
/// `b̄ = A z`, `Ā = −(A z) xᵀ + r zᵀ`.
///
/// `factors`, `a`, `b`, `x` must come from the primal solve.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when any operand disagrees with the
/// factors; propagates triangular-solve errors.
pub fn lstsq_vjp(
    factors: &QrFactors,
    a: &Matrix,
    b: &Vector,
    x: &Vector,
    x_bar: &Vector,
) -> Result<SolveGradients, LinalgError> {
    let (m, n) = check_lstsq_shapes(factors, a, b, x)?;
    if x_bar.len() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: x_bar.len(),
            cols: 1,
        });
    }
    let r_mat = factors.r();
    let z = solve_normal(&r_mat, x_bar)?;
    let b_bar = a * &z;
    let ax = a * x;
    let resid = b - &ax;
    let a_bar = Matrix::from_fn(m, n, |i, j| -(b_bar[i] * x[j]) + resid[i] * z[j]);
    Ok(SolveGradients { a_bar, b_bar })
}

/// Forward-mode rule for full-rank least squares.
///
/// `ẋ = (AᵀA)⁻¹ (Ȧᵀ r − Aᵀ Ȧ x + Aᵀ ḃ)` with residual `r = b − A x`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] when any operand disagrees with the
/// factors; propagates triangular-solve errors.
pub fn lstsq_jvp(
    factors: &QrFactors,
    a: &Matrix,
    b: &Vector,
    x: &Vector,
    a_dot: &Matrix,
    b_dot: &Vector,
) -> Result<Vector, LinalgError> {
    let (m, n) = check_lstsq_shapes(factors, a, b, x)?;
    if a_dot.rows() != m || a_dot.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            rows: a_dot.rows(),
            cols: a_dot.cols(),
        });
    }
    if b_dot.len() != m {
        return Err(LinalgError::DimensionMismatch {
            rows: b_dot.len(),
            cols: 1,
        });
    }
    let ax = a * x;
    let resid = b - &ax;
    let at = a.transpose();
    let adot_t = a_dot.transpose();
    // t = Ȧᵀ r − Aᵀ Ȧ x + Aᵀ ḃ
    let term1 = &adot_t * &resid;
    let adot_x = a_dot * x;
    let term2 = &at * &adot_x;
    let term3 = &at * b_dot;
    let t = &(&term1 - &term2) + &term3;
    let r_mat = factors.r();
    solve_normal(&r_mat, &t)
}
```

Update `src/linalg/mod.rs`: `pub use adjoint::{SolveGradients, lstsq_jvp, lstsq_vjp, solve_jvp, solve_vjp};` and mirror in `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test linalg_lstsq_adjoint`
Expected: PASS (3 tests, three-way agreement < 1e-9). Full suite + clippy.

- [ ] **Step 5: Commit**

```bash
git add src/linalg/adjoint.rs src/linalg/mod.rs src/lib.rs tests/linalg_lstsq_adjoint.rs
git commit -m "feat: least-squares adjoint rule (lstsq_vjp/lstsq_jvp) with three-way test"
```

---

### Task 10: Reductions, determinants, docs

**Files:**
- Create: `src/linalg/reductions.rs`
- Modify: `src/linalg/mod.rs` (add `pub mod reductions;`)
- Modify: `src/core/matrix.rs` (add `as_slice`)
- Modify: `src/linalg/lu.rs` (add `LuFactors::determinant`)
- Modify: `src/linalg/cholesky.rs` (add `LltFactors::determinant`)
- Modify: `src/lib.rs` (module doc: mention Phase 3 capabilities)
- Modify: `docs/architecture.md` (one short subsection: Phase 3 decomposition suite)
- Test: `tests/linalg_reductions.rs`

**Interfaces:**
- Consumes: `Perm::sign` (Task 1), `LuFactors`/`LltFactors` internals, `Vector::as_slice`.
- Produces: `mercury::linalg::reductions::{sum, norm_l1, norm_l2, norm_max}` (free functions on `&[f64]`); `Matrix::as_slice(&self) -> &[f64]`; `LuFactors::determinant(&self) -> f64`; `LltFactors::determinant(&self) -> f64`.

- [ ] **Step 1: Write the failing tests**

Create `tests/linalg_reductions.rs`:

```rust
// Exact float asserts are intentional in tests.
#![allow(clippy::float_cmp)]

//! Reductions and factorization determinants.

use mercury::linalg::reductions::{norm_l1, norm_l2, norm_max, sum};
use mercury::{Matrix, Vector, llt_factor, lu_factor};

#[test]
fn vector_reductions() {
    let v = Vector::from_slice(&[3.0, -4.0]);
    assert_eq!(sum(v.as_slice()), -1.0);
    assert_eq!(norm_l1(v.as_slice()), 7.0);
    assert_eq!(norm_l2(v.as_slice()), 5.0);
    assert_eq!(norm_max(v.as_slice()), 4.0);
}

#[test]
fn matrix_reductions_use_all_elements() {
    // norm_l2 over a matrix slice is the Frobenius norm.
    let m = Matrix::from_fn(2, 2, |i, j| [[1.0, -2.0], [2.0, 4.0]][i][j]);
    assert_eq!(sum(m.as_slice()), 5.0);
    assert_eq!(norm_l1(m.as_slice()), 9.0);
    assert_eq!(norm_l2(m.as_slice()), 5.0);
    assert_eq!(norm_max(m.as_slice()), 4.0);
}

#[test]
fn empty_reductions_are_zero() {
    assert_eq!(sum(&[]), 0.0);
    assert_eq!(norm_l1(&[]), 0.0);
    assert_eq!(norm_l2(&[]), 0.0);
    assert_eq!(norm_max(&[]), 0.0);
}

#[test]
fn lu_determinant_with_permutation_sign() {
    // [[0, 1], [1, 0]] forces a pivot swap; det = -1.
    let a = Matrix::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
    let det = lu_factor(&a).expect("wc").determinant();
    assert!((det - (-1.0)).abs() < 1e-14);
    // [[2, 1], [1, 3]]: det = 5.
    let b = Matrix::from_fn(2, 2, |i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
    let det_b = lu_factor(&b).expect("wc").determinant();
    assert!((det_b - 5.0).abs() < 1e-12);
}

#[test]
fn llt_determinant_matches_lu() {
    let a = Matrix::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let det_llt = llt_factor(&a).expect("spd").determinant();
    let det_lu = lu_factor(&a).expect("wc").determinant();
    assert!((det_llt - 8.0).abs() < 1e-12);
    assert!((det_llt - det_lu).abs() < 1e-12);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release --test linalg_reductions`
Expected: FAIL to compile — `reductions` module not found.

- [ ] **Step 3: Write the implementation**

Create `src/linalg/reductions.rs`:

```rust
//! Element-wise reductions over raw `f64` slices (faer's `reductions/`
//! idea, flattened: `Vector::as_slice()` / `Matrix::as_slice()` feed these,
//! and for matrices `norm_l2` is the Frobenius norm).
//!
//! Host-side conveniences — no kernel-safety claims. `norm_l2` uses the
//! naive sum-of-squares (no overflow rescaling); fine for the magnitudes
//! Mercury works with.

/// Sum of all elements (`0.0` for empty input).
#[must_use]
pub fn sum(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v;
    }
    acc
}

/// Sum of absolute values.
#[must_use]
pub fn norm_l1(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v.abs();
    }
    acc
}

/// Euclidean norm (Frobenius norm for matrix slices).
#[must_use]
pub fn norm_l2(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v * v;
    }
    acc.sqrt()
}

/// Largest absolute value (`0.0` for empty input).
#[must_use]
pub fn norm_max(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        let a = v.abs();
        if a > acc {
            acc = a;
        }
    }
    acc
}
```

In `src/linalg/mod.rs` add:

```rust
pub mod reductions;
```

In `src/core/matrix.rs`, add to `impl Matrix`:

```rust
    /// Borrows all elements as a row-major slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
```

In `src/linalg/lu.rs`, add to `impl LuFactors`:

```rust
    /// Determinant of the factored matrix: `sign(P) · Π u_ii`.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let mut det = self.perm.sign();
        for i in 0..self.dimension() {
            det *= self.lu[(i, i)];
        }
        det
    }
```

In `src/linalg/cholesky.rs`, add to `impl LltFactors`:

```rust
    /// Determinant of the factored matrix: `Π l_ii²`.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let mut det = 1.0;
        for i in 0..self.dimension() {
            det *= self.l[(i, i)] * self.l[(i, i)];
        }
        det
    }
```

- [ ] **Step 4: Update docs**

In `src/lib.rs`, update the `linalg` bullet of the crate doc to mention the Phase 3 surface (keep it one bullet, e.g.):

```rust
//! - [`linalg`]: kernel-safe fixed solves (`solve_fixed_unchecked`,
//!   `solve_spd_fixed_unchecked`) and host-side factorizations (LU, LLT,
//!   LDLT, QR) behind one [`linalg::Factorization`] adjoint rule
//!   (`solve_vjp`/`solve_jvp`), plus a dedicated least-squares rule
//!   (`lstsq_vjp`/`lstsq_jvp`) — never differentiate the factorization.
```

In `docs/architecture.md`, append this subsection at the end of the linalg-related section (adjust the heading level to match neighbors):

```markdown
### Phase 3: decomposition suite

Phase 3 mines faer's layout (vendored at `ref/faer`) for an Enzyme-compatible
decomposition suite: Cholesky (`llt_factor`, unpivoted `ldlt_factor`) and
Householder QR (`qr_factor` with square `solve` and full-rank `solve_lstsq`),
built on a shared triangular-substitution substrate and a `Perm` type. All
square-solve factorizations (LU, LLT, LDLT) implement one `Factorization`
trait, so a single adjoint rule (`solve_vjp`/`solve_jvp`) serves them all;
least squares carries its own rule (`lstsq_vjp`/`lstsq_jvp`), validated by
the same three-way agreement law. The one kernel-side addition is
`solve_spd_fixed_unchecked` (unpivoted LLT, NaN-propagating). Matrix views
and SVD/EVD are Phase 4 candidates per the Phase 3 spec.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --release --test linalg_reductions`
Expected: PASS (5 tests). Then the full suite, clippy, and `cargo doc --release --no-deps` (missing_docs is a warn — keep it clean).

- [ ] **Step 6: Commit**

```bash
git add src/linalg/reductions.rs src/linalg/mod.rs src/core/matrix.rs src/linalg/lu.rs src/linalg/cholesky.rs src/lib.rs docs/architecture.md tests/linalg_reductions.rs
git commit -m "feat: reductions module and factorization determinants; Phase 3 docs"
```

---

## Final verification (after all tasks)

- `cargo test --release` — full suite green (expect ~55+ tests).
- `cargo clippy --release --all-targets` — clean.
- `nix flake check` — all checks green (requires `git add`-ing any new files first; flakes only see tracked files).
- `scripts/ci.sh` if present — green.
- Version bump: run `scripts/bump-version.sh` to `0.3.0` before the merge PR (the version-bump CI gate on PRs to `main` requires it).
