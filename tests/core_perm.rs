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

#[test]
#[should_panic(expected = "index out of range in Perm::swap")]
fn swap_panics_on_out_of_range_even_when_equal() {
    let mut p = Perm::identity(3);
    // Equal indices must not bypass the range check.
    p.swap(5, 5);
}
