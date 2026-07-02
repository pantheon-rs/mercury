//! Differentiable math substrate for `pantheon-rs`.
//!
//! Mercury Phase 1 is a plain-`f64` core differentiated with Rust nightly
//! `std::autodiff` / Enzyme.
#![feature(autodiff)]
#![forbid(unsafe_code)]

mod objective;

pub use objective::ValueGradient;

pub mod core;

pub use crate::core::{SMatrix, SVector};

pub mod validation;
