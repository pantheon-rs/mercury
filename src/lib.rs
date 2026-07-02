//! Differentiable math substrate for `pantheon-rs`.
//!
//! Mercury Phase 1 is a plain-`f64` core differentiated with Rust nightly
//! `std::autodiff` / Enzyme.
#![feature(autodiff)]
#![forbid(unsafe_code)]

mod objective;

pub use objective::ValueGradient;

pub mod core;

pub use crate::core::{Matrix, SMatrix, SVector, Vector};

pub mod geometry;

pub use crate::geometry::Quaternion;

pub mod linalg;

pub use crate::linalg::{LinalgError, LuFactors, lu_factor, solve, solve_fixed, solve_fixed_unchecked};

pub mod validation;
