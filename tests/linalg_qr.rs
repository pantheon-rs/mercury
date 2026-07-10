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

#[test]
fn rows_and_cols_accessors_report_factored_shape() {
    let f = qr_factor(&tall_a()).expect("qr");
    assert_eq!(f.rows(), 4);
    assert_eq!(f.cols(), 2);

    let square = Matrix::from_fn(3, 3, |i, j| if i == j { 1.0 + i as f64 } else { 0.1 });
    let fs = qr_factor(&square).expect("qr");
    assert_eq!(fs.rows(), 3);
    assert_eq!(fs.cols(), 3);
}

#[test]
fn square_rank_deficient_errors() {
    // First column entirely zero: breakdown at column 0. Distinct from the
    // tall (m > n) rank-deficient case above, where the failing column
    // isn't the first one and m != n.
    let a = Matrix::from_fn(2, 2, |_, j| if j == 0 { 0.0 } else { 1.0 });
    assert_eq!(
        qr_factor(&a).map(|_| ()),
        Err(LinalgError::RankDeficient { column: 0 })
    );
}

#[test]
fn single_column_tall_matrix_solves_lstsq() {
    // n = 1: least squares reduces to the projection x = (aᵀb)/(aᵀa).
    let a = Matrix::from_fn(3, 1, |i, _| (i + 1) as f64); // [1, 2, 3]^T
    let b = Vector::from_slice(&[2.0, 4.0, 5.0]);
    let x = qr_factor(&a).expect("qr").solve_lstsq(&b).expect("lstsq");

    let mut ata = 0.0;
    let mut atb = 0.0;
    for i in 0..3 {
        ata += a[(i, 0)] * a[(i, 0)];
        atb += a[(i, 0)] * b[i];
    }
    assert!((x[0] - atb / ata).abs() < 1e-9);
}

#[test]
fn lstsq_solution_minimizes_residual_norm() {
    // Optimality check: any perturbation away from the least-squares
    // solution must not decrease the residual norm (property-based, no
    // hand-computed closed form needed beyond the perturbation itself).
    let a = tall_a();
    let b = Vector::from_slice(&[2.1, 3.9, 6.2, 7.8]);
    let x = qr_factor(&a).expect("qr").solve_lstsq(&b).expect("lstsq");
    let residual_at = |candidate: &Vector| -> f64 {
        let ax = &a * candidate;
        let r = &b - &ax;
        r.norm_squared()
    };
    let baseline = residual_at(&x);
    for (di, dj) in [(0.1, 0.0), (0.0, 0.1), (-0.05, 0.2), (0.3, -0.3)] {
        let perturbed = Vector::from_slice(&[x[0] + di, x[1] + dj]);
        assert!(
            residual_at(&perturbed) >= baseline - 1e-12,
            "perturbation ({di},{dj}) decreased residual below optimum"
        );
    }
}
