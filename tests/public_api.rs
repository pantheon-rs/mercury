//! Public API smoke tests.

use mercury::{square, where_};

#[test]
fn public_square_is_generic() {
    assert!((square(4.0_f32) - 16.0).abs() < 1.0e-6);
    assert!((square(4.0_f64) - 16.0).abs() < 1.0e-12);
}

#[test]
fn explicit_selection_is_public() {
    let value = where_(2.0_f64 > 1.0_f64, 10.0_f64, -10.0_f64);
    assert!((value - 10.0).abs() < 1.0e-12);
}
