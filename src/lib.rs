//! Differentiable math substrate for `pantheon-rs`.
//!
//! Every Mercury primitive is plain-`f64` Rust with a validated,
//! Mercury-owned derivative rule (decision 0003). Enzyme differentiates
//! user kernels; Mercury owns the rules at the joints.
//!
//! - [`core`]: POD-transparent types. Fixed-size (`SVector`, `SMatrix`) are
//!   kernel-safe; dynamic (`Vector`, `Matrix`) host data outside kernels.
//! - [`geometry`]: `Quaternion` and rotations (analytic derivatives).
//! - [`linalg`]: `solve_fixed_unchecked` (kernel-safe, differentiate-through;
//!   `solve_fixed` is its `Result`-returning host-side wrapper) and the
//!   dynamic LU `solve` whose derivative is the adjoint rule
//!   (`solve_vjp`/`solve_jvp`) — never the factorization.
//! - [`validation`]: finite-difference oracles for the three-legged test law.
#![feature(autodiff)]
#![forbid(unsafe_code)]

mod objective;

pub use objective::ValueGradient;

pub mod core;

pub use crate::core::{Matrix, Perm, SMatrix, SVector, Vector};

pub mod geometry;

pub use crate::geometry::Quaternion;

pub mod linalg;

pub use crate::linalg::{
    Factorization, LinalgError, LdltFactors, LltFactors, LuFactors, QrFactors, SolveGradients,
    ldlt_factor, llt_factor, lu_factor, qr_factor, solve, solve_fixed, solve_fixed_unchecked,
    solve_jvp, solve_spd_fixed_unchecked, solve_vjp,
};

pub mod validation;
