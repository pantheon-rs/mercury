# Phase 3: Decomposition Suite + Structural Layer — Design

**Status:** approved design, pre-implementation
**Date:** 2026-07-03
**Depends on:** decision 0003 (differentiable-primitives identity), Phase 2 (core types, LU, adjoint rule)
**Reference:** `ref/faer/` (vendored faer 0.24.4 checkout) — the source of the ideas mined here

## Context

Phase 2 proved the thesis: mercury owns POD-transparent types, Enzyme differentiates
user kernels, and mercury owns the derivative rule at the linear-solve joint
(three-way agreement < 1e-9). Phase 3 deepens the linear algebra layer by mining
faer's best ideas — its layout, capability surface, and correctness practices —
re-expressed inside mercury's Enzyme constraints. This phase replaces interp tables
as Phase 3 in decision 0003's phasing table (user decision, 2026-07-03); interp
shifts to a later phase, and SVD/EVD is planned as Phase 4.

## Goals

1. Cholesky family: LLT and unpivoted LDLT factorizations with uniform solve API.
2. Householder QR: square solve + full-rank overdetermined least squares.
3. Structural layer taken from faer: host-side matrix views, a real `Perm` type,
   shared triangular-solve primitives, reductions (norms/sum/determinant).
4. Derivative-rule unification: one `Factorization` trait powering `solve_vjp` /
   `solve_jvp` across LU/LLT/LDLT; a dedicated least-squares adjoint rule.
5. Kernel-side SPD solve (`solve_spd_fixed_unchecked`) obeying Enzyme IR rules 1–8.

## Non-Goals (deliberate)

- **SVD / EVD / GEVD** — deferred to **Phase 4** (faer's heaviest machinery;
  derivative rules complicated by degenerate singular values). This is the next
  phase, not indefinite deferral.
- **Bunch-Kaufman / pivoted LDLT** — deferred until indefinite KKT systems demand it.
- **SIMD (pulp), rayon parallelism, gemm backends** — performance machinery with
  Enzyme-hostile IR; orthogonal to the thesis.
- **Generic scalars (`ComplexField`)** — mercury is f64-only (POD law, decision 0003).
- **Sparse, stats, operator modules** — out of scope.
- **Views inside Enzyme kernels** — views are host-side only; kernels keep owned
  `SMatrix`/`SVector`. No new kernel IR surface is introduced by the structural layer.

## What we take from faer

| faer idea | mercury form |
|---|---|
| Solver objects (`Llt`, `PartialPivLu`, `Qr` + uniform `.solve()`) | `LltFactors`, `LdltFactors`, `QrFactors` alongside existing `LuFactors` |
| Write-once-against-views (`MatRef`/`MatMut`) | Host-side `MatRef<'a>`/`MatMut<'a>` windows into `Matrix` |
| `triangular_solve.rs` as shared substrate | `src/linalg/triangular.rs`; LU refactors onto it |
| `Perm` type | `src/core/perm.rs`, replaces LU's raw `Vec<usize>` |
| `reductions/` (norms, sum, determinant) | `src/linalg/reductions.rs` |
| Per-decomposition error types + equator asserts | Variants on the one shared `LinalgError`; per-axis asserts (established Phase 2) |
| Per-decomposition test modules | `tests/linalg_cholesky.rs`, `tests/linalg_qr.rs`, … |

## Architecture

### Module layout

```
src/core/
  view.rs         MatRef<'a>, MatMut<'a>          (new; host-side only)
  perm.rs         Perm                             (new)
src/linalg/
  triangular.rs   forward/back substitution        (new; shared substrate)
  cholesky.rs     llt_factor, ldlt_factor          (new)
  qr.rs           qr_factor, solve, solve_lstsq    (new)
  reductions.rs   norms, sum, determinant          (new)
  adjoint.rs      generalizes over Factorization   (modified)
  lu.rs           refactors onto triangular + Perm (modified)
  fixed.rs        + solve_spd_fixed_unchecked      (modified)
  error.rs        + NotPositiveDefinite, RankDeficient (modified)
```

### Views (host-side only)

`MatRef<'a>` / `MatMut<'a>` are borrowed windows into a `Matrix`: pointer to the
parent's storage plus `(row_offset, col_offset, rows, cols)` against the parent's
row stride. They exist so Householder QR and blocked/right-looking factorizations
can operate on trailing sub-blocks without copies — faer's central layering idea.

Constraints:
- Views never cross into kernel-reachable code. The AD-safe subset remains
  owned `SVector`/`SMatrix` + `&[f64]` slices, exactly as in Phase 2.
- `unsafe_code = "forbid"` stands. Views are implemented with safe splitting
  (`split_at_row_mut`-style borrows on `&mut [f64]`), not raw pointers. If a safe
  formulation of some splitting op proves impossible, the algorithm indexes the
  owned `Matrix` directly instead — correctness first, faer-style zero-copy second.
- Indexing through views carries the same per-axis bounds asserts as `Matrix`.

### `Perm`

```rust
pub struct Perm(Vec<usize>);           // perm[i] = original row index
impl Perm {
    pub fn identity(n: usize) -> Self;
    pub fn len(&self) -> usize;
    pub fn apply(&self, v: &Vector) -> Vector;          // (Pv)[i] = v[perm[i]]
    pub fn apply_inverse(&self, v: &Vector) -> Vector;  // out[perm[i]] = v[i]
    pub fn swap(&mut self, i: usize, j: usize);
}
```

`LuFactors` stores a `Perm` instead of `Vec<usize>`; the convention
(`perm[i]` = original row) is unchanged from Phase 2.

### Triangular substrate

```rust
// src/linalg/triangular.rs — all host-side, operating on Matrix/Vector.
pub(crate) fn solve_lower(l: &Matrix, b: &Vector, unit_diag: bool) -> Result<Vector, LinalgError>;
pub(crate) fn solve_upper(u: &Matrix, b: &Vector, unit_diag: bool) -> Result<Vector, LinalgError>;
pub(crate) fn solve_lower_transposed(l: &Matrix, b: &Vector, unit_diag: bool) -> Result<Vector, LinalgError>;
pub(crate) fn solve_upper_transposed(u: &Matrix, b: &Vector, unit_diag: bool) -> Result<Vector, LinalgError>;
```

Zero-pivot detection uses the shared `PIVOT_TOLERANCE`. `LuFactors::solve` /
`solve_transposed` refactor onto these with behavior pinned by the existing Phase 2
tests (which must pass unchanged).

## Capability surface

### Cholesky LLT

```rust
pub fn llt_factor(a: &Matrix) -> Result<LltFactors, LinalgError>;
impl LltFactors {
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>;
    pub fn l(&self) -> &Matrix;                 // lower factor, A = L Lᵀ
    pub fn determinant(&self) -> f64;           // Π l_ii²
}
```

- Input must be square; symmetry is the caller's contract (only the lower triangle
  is read, faer's `Side::Lower` convention with the side fixed).
- A non-positive pivot (`l_ii² ≤ PIVOT_TOLERANCE`) fails with
  `LinalgError::NotPositiveDefinite { pivot_index }`.

### Cholesky LDLT (unpivoted)

```rust
pub fn ldlt_factor(a: &Matrix) -> Result<LdltFactors, LinalgError>;
impl LdltFactors {
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>;
    pub fn l(&self) -> &Matrix;                 // unit-lower
    pub fn d(&self) -> &Vector;                 // diagonal of D
}
```

- No square roots; tolerates negative diagonals (indefinite A) but fails with
  `LinalgError::NotPositiveDefinite` only when `|d_i| ≤ PIVOT_TOLERANCE`
  (breakdown). Unpivoted LDLT on strongly indefinite matrices can be inaccurate —
  documented; Bunch-Kaufman is the deferred answer.

### Householder QR

```rust
pub fn qr_factor(a: &Matrix) -> Result<QrFactors, LinalgError>;  // m >= n required
impl QrFactors {
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>;        // square only
    pub fn solve_lstsq(&self, b: &Vector) -> Result<Vector, LinalgError>;  // min ‖Ax−b‖₂
    pub fn r(&self) -> Matrix;              // upper-triangular n×n (thin R)
    pub fn q_transpose_apply(&self, b: &Vector) -> Result<Vector, LinalgError>;  // Qᵀb via reflectors
}
```

- Stored factored form: reflectors `v_k` packed below the diagonal + `tau` vector +
  `R` on/above the diagonal (LAPACK-style compact storage; faer's
  `householder.rs` idea without the block variant).
- `m < n` (underdetermined) rejected with `DimensionMismatch`; a zero diagonal of R
  (`|r_kk| ≤ PIVOT_TOLERANCE`) fails with `LinalgError::RankDeficient { column }`.
- `solve` requires `m == n`; `solve_lstsq` requires `m >= n`, returns the unique
  full-rank least-squares solution `x = R⁻¹ Qᵀb` (thin).

### Reductions

```rust
// on Vector and Matrix (free functions or inherent methods, host-side)
pub fn norm_l1 / norm_l2 / norm_max / sum
// determinant via factorization objects:
LuFactors::determinant()   // sign(P) · Π u_ii
LltFactors::determinant()  // Π l_ii²
```

`Vector::norm_squared` (exists, kernel-safe) is unchanged; the new reductions are
host-side conveniences and make no kernel-safety claims.

### Kernel-side SPD solve

```rust
/// Unpivoted LLT solve for small fixed SPD systems inside Enzyme kernels.
/// On breakdown (non-SPD), propagates NaN instead of erroring —
/// same contract as `solve_fixed_unchecked`.
pub fn solve_spd_fixed_unchecked<const N: usize>(a: &SMatrix<N, N>, b: &SVector<N>) -> SVector<N>;
```

- Written under Enzyme IR rules 1–8: plain `for i in 0..N` loops, construct-then-
  mutate via the proven `from_fn` patterns, no `Result` return, no iterator adapters.
- Cheaper than the LU kernel path for SPD systems (no pivot search), and the
  natural kernel primitive for the NLP end-game's KKT-related solves.
- Gets its own Enzyme compile + gradient test legs (three-legged law).

## Derivative rules (the identity work)

### Unified square-solve rule: the `Factorization` trait

The adjoint of `x = A⁻¹b` needs only solves against the factors — never the
factorization internals:

```rust
pub trait Factorization {
    fn dimension(&self) -> usize;
    fn solve(&self, b: &Vector) -> Result<Vector, LinalgError>;
    fn solve_transposed(&self, b: &Vector) -> Result<Vector, LinalgError>;
}
```

- Implemented by `LuFactors` (exists), `LltFactors`, `LdltFactors`
  (symmetric ⇒ `solve_transposed` delegates to `solve`).
- `solve_vjp` / `solve_jvp` generalize from `&LuFactors` to `&impl Factorization`
  with formulas unchanged:
  - VJP: `b̄ = A⁻ᵀ x̄`, `Ā = −b̄ xᵀ`
  - JVP: `ẋ = A⁻¹(ḃ − Ȧ x)`
- All Phase 2 adjoint tests must pass unchanged after the generalization; the
  three-way agreement test (FD vs Enzyme vs rule) is added per new factorization
  with an SPD test matrix (`M + 5I` with `M = 0.1·(random-ish fixed values)`,
  symmetrized).

### Least-squares rule (new, distinct from the square-solve rule)

For full-column-rank `x = argmin ‖Ax − b‖₂` (A is m×n, m ≥ n), with residual
`r = b − Ax`, differentiate the normal equations `AᵀA x = Aᵀb`:

- **JVP:** `ẋ = (AᵀA)⁻¹ (Ȧᵀ r − Aᵀ Ȧ x + Aᵀ ḃ)`
- **VJP:** with `z = (AᵀA)⁻¹ x̄`:
  - `b̄ = A z`
  - `Ā = −(A z) xᵀ + r zᵀ`

`(AᵀA)⁻¹ w` is computed as `R⁻¹ R⁻ᵀ w` — two triangular solves against the QR
factors, never forming `AᵀA`. API:

```rust
pub fn lstsq_vjp(factors: &QrFactors, a: &Matrix, b: &Vector, x: &Vector, x_bar: &Vector)
    -> Result<SolveGradients, LinalgError>;
pub fn lstsq_jvp(factors: &QrFactors, a: &Matrix, b: &Vector, x: &Vector,
    a_dot: &Matrix, b_dot: &Vector) -> Result<Vector, LinalgError>;
```

(The rule needs `A` and `r`, unlike the square rule — hence the wider signature.)
Validated by a three-way agreement test on a 5×3 full-rank problem: FD vs Enzyme
(through a small fixed-size lstsq kernel is NOT required — the Enzyme leg
differentiates through `solve_spd_fixed_unchecked` on the normal equations
`(AᵀA)x = Aᵀb` of the same problem, which is a mathematically identical function
of the inputs) vs `lstsq_vjp`. FD tolerance 1e-4, rule-vs-Enzyme 1e-9, matching
the Phase 2 thresholds.

## Error handling

`LinalgError` (one shared enum — user decision) grows:

```rust
NotPositiveDefinite { pivot_index: usize },
RankDeficient { column: usize },
```

Existing variants (`Singular`, `DimensionMismatch`, `NotSquare`, …) unchanged.
All new entry points validate dimensions first and return errors — no panicking
API except `Index`/`IndexMut` (established convention).

## Testing strategy

Per-decomposition test files, each with:

1. **Known-solution correctness** — hand-checkable small systems.
2. **Reconstruction** — `L·Lᵀ ≈ A`, `L·D·Lᵀ ≈ A`, `Q·R ≈ A` within 1e-12 on
   well-conditioned inputs.
3. **Residual bounds** — `‖Ax − b‖` small for solves; for lstsq, verify the normal
   equations `Aᵀ(Ax − b) ≈ 0`.
4. **Error paths** — non-SPD input to LLT, rank-deficient input to QR, every
   dimension-mismatch arm.
5. **Cross-checks** — LLT solve vs LU solve on the same SPD system agree to 1e-12.
6. **Derivative rules** — three-way agreement per rule (FD / Enzyme / mercury rule),
   Enzyme legs behind `cfg(not(coverage))` (rule 8).
7. **Refactor pins** — all Phase 2 tests pass unchanged after the LU-onto-triangular
   and adjoint-generalization refactors.

Kernel additions (`solve_spd_fixed_unchecked`) get the three-legged law: Enzyme
compile test, FD cross-check, analytic check.

## Phasing within Phase 3 (implementation order)

1. Structural layer: `Perm`, `triangular.rs`, LU refactor (pinned by existing tests).
2. `Factorization` trait + adjoint generalization (pinned by existing tests).
3. LLT (+ trait impl, three-way test) → LDLT (+ trait impl, three-way test).
4. `solve_spd_fixed_unchecked` (kernel path, three-legged tests).
5. QR + `solve` + `solve_lstsq`.
6. Least-squares adjoint rule + three-way test.
7. Views (`MatRef`/`MatMut`) — introduced when QR/blocked code needs them, not
   before (YAGNI: if steps 3–6 stay clean with owned `Matrix` indexing, views may
   land as a refactor at the end of the phase or slip to Phase 4).
8. Reductions + determinants.

## Deferred / Phase 4 preview

- **SVD/EVD** is the planned Phase 4: faer's `svd/` (bidiagonalization + implicit-QR)
  and `evd/` as the primal references; derivative rules will need care around
  repeated/degenerate singular values.
- Bunch-Kaufman (indefinite KKT), pivoted LLT, blocked/Householder-block variants,
  sparse — unscheduled.
