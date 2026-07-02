//! Mercury-owned core math types (POD-transparency law, decision 0003).

mod matrix;
mod smatrix;
mod svector;
mod vector;

pub use matrix::Matrix;
pub use smatrix::SMatrix;
pub use svector::SVector;
pub use vector::Vector;
