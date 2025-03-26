use nalgebra::{Const, DVector, Dyn, MatrixViewMut, U1};

/// Updates an LDL^T factorization for a rank-1 modification of the form LDL^T + alp*uu^T
///
/// Computes the updated LDL^T factorization when a symmetric rank-1 update
/// is applied to a matrix. Given the current factorization A = LDL^T, this function
/// computes the factorization of A + alp*uu^T.
///
/// # Arguments
/// * `L` - Mutable view into the lower triangular matrix with unit diagonal from the LDL^T factorization
/// * `d` - Mutable view containing the diagonal elements of D
/// * `alp` - Scalar multiplier for the rank-1 update (can be positive or negative)
/// * `u` - Vector used in the rank-1 update
///
/// # Returns
/// * A direction vector `p` of null or negative curvature if the update would make the
///   factorization indefinite (when alp < 0). Returns an empty vector if the update succeeds.
///
/// # Mathematical Background
/// The rank-1 update of an LDL^T factorization is a fundamental operation in numerical linear algebra.
/// When alp > 0, the update always succeeds and produces a positive definite matrix if the original
/// was positive definite.
///
/// When alp < 0, the update may cause the matrix to become indefinite. In this case, the function
/// returns a direction of negative curvature, which is useful in optimization algorithms to determine
/// descent directions.
///
/// # Note
/// This function does not work for matrices of dimension 0.
pub fn ldlrk1<const N: usize>(
    L: &mut MatrixViewMut<f64, Dyn, Dyn, Const<1>, Const<{ N }>>,
    d: &mut MatrixViewMut<f64, Dyn, U1, Const<1>, Const<{ N }>>,
    mut alp: f64,
    u: &mut DVector<f64>,
) ->
    DVector<f64> // p
{
    if alp == 0.0 {
        return DVector::zeros(0);
    }

    let n: usize = u.len();
    let neps: f64 = n as f64 * f64::EPSILON;

    // save old factorization
    let (L0, d0) = (L.clone_owned(), d.clone_owned());

    // Very important to allocate memory gere as loop modifies u
    let u_non_zero_indices: Vec<usize> = u.iter()
        .enumerate()
        .filter_map(|(idx, &val)| (val != 0.0).then_some(idx))
        .collect();

    for &k in u_non_zero_indices.iter() {
        let delta: f64 = d[k] + alp * u[k].powi(2);

        if alp < 0.0 && delta <= neps {
            let mut p = DVector::zeros(n);
            p[k] = 1.0;

            // Solve the system for the first k+1 elements
            // Note: In Matlab, p(1:k) refers to the first k elements
            // In Rust with 0-indexing, we need to use ..=k for the same range
            let p_slice = p.rows_mut(0, k + 1);

            // This is equivalent to the Matlab L(1:k,1:k)'\p(1:k)
            let solution = L.view_range(0..k + 1, 0..k + 1).transpose().lu().solve(&p_slice).unwrap(); // TODO: not sure about .transpose() here. No such in Python version

            // Update the first k+1 elements of p with the solution
            p.rows_mut(0, k + 1).copy_from(&solution);

            // Note: The original code has "restore original factorization"
            L.copy_from(&L0); // This will copy the data from L0 back to L
            d.copy_from(&d0);

            return p;
        }

        let q = d[k] / delta;
        d[k] = delta;

        // helpers for the loop below
        let uk = u[k];
        let alp_uk_div_delta = alp * uk / delta;

        // ind=k+1:n;
        // c=L(ind,k)*u(k);
        // L(ind,k)=L(ind,k)*q+(alp*u(k)/del)*u(ind,1);
        // u(ind,1)=u(ind,1)-c;
        for index in (k + 1)..n {
            let ui = u[index];
            u[index] -= L[(index, k)] * uk;
            L[(index, k)] = L[(index, k)] * q + alp_uk_div_delta * ui;
        }

        alp *= q;
        if alp == 0.0 { break; }
    }
    DVector::zeros(0)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    static TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_coverage_0() {
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1., -2., 3.,
            4., -5., 6.,
            -7., 8., 9.
        ]);
        let mut d = DVector::from_row_slice(&[0.1, 0.2, 0.3]);
        let alp = 4.4;
        let mut u = DVector::from_row_slice(&[1.23, -1.0, 1.0]);

        // Create mutable views for L and d
        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, -2.0, 3.0, -0.7417756439476909, -5.0, 6.0, 0.6973756652596808, -0.8479315955240974, 9.0]));

        assert_eq!(d, DVector::<f64>::from_row_slice(&[6.75676, 2.482220472534174, 17.329279155304317]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));

        assert_eq!(alp, 4.4);
        assert_eq!(u, DVector::from_row_slice(&[1.23, -5.92, 56.97]));
    }

    #[test]
    fn test_coverage_1() {
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.
        ]);
        let mut d = DVector::from_row_slice(&[0.1, 0.2, 0.3]);
        let alp = -0.4;
        let mut u = DVector::from_row_slice(&[1.23, -1.0, 1.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));

        assert_eq!(d, DVector::from_row_slice(&[0.1, 0.2, 0.3]));
        assert_eq!(p, DVector::from_row_slice(&[1., 0., 0.]));

        assert_eq!(alp, -0.4);
        assert_eq!(u, DVector::from_row_slice(&[1.23, -1., 1.]));
    }


    #[test]
    fn test_coverage_break_case() {
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1., -2., 3.,
            4., -5., 6.,
            -7., 8., 9.
        ]);
        let mut d = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
        let alp = 0.01;
        let mut u = DVector::from_row_slice(&[1.23, -1.0, 1.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_relative_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, -2.0, 3.0, -0.8130081300813009, -5.0, 6.0, 0.8130081300813009, 8.0, 9.0]), epsilon = TOLERANCE);

        assert_eq!(d, DVector::from_row_slice(&[0.015129, 0., 0.]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));

        assert_eq!(alp, 0.01);
        assert_eq!(u, DVector::from_row_slice(&[1.23, -5.92, 9.61]));
    }

    #[test]
    fn test_real() {
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1.0, 0.0, -1.45,
            -0.6, -1.0, 4.0,
            -1.6, 1.0, 0.01
        ]);
        let mut d = DVector::from_row_slice(&[0.01, 0.002, 0.03]);
        let alp = -10.0;
        let mut u = DVector::from_row_slice(&[1.23, -10.0, 5.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[
            1.0, 0.0, -1.45,
            -0.6, -1.0, 4.0,
            -1.6, 1.0, 0.01
        ]));

        assert_eq!(d, DVector::from_row_slice(&[0.01, 0.002, 0.03]));
        assert_eq!(p, DVector::from_row_slice(&[1.0, 0.0, 0.0]));

        assert_eq!(alp, -10.0);
    }


    #[test]
    fn test_0() {
        let mut L = DMatrix::<f64>::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[2.0, 3.0]);
        let alp = 1.0;
        let mut u = DVector::from_row_slice(&[1.0, 1.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<2>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.6666666666666666, 1.0,
        ]));

        assert_eq!(d, DVector::<f64>::from_row_slice(&[3.0, 3.1666666666666665]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
        assert_eq!(alp, 1.0);
    }

    #[test]
    fn test_1() {
        let mut L = DMatrix::<f64>::from_row_slice(1, 1, &[1.0]);
        let mut d = DVector::from_row_slice(&[2.0]);
        let alp = 0.0;
        let mut u = DVector::from_row_slice(&[1.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<1>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(1, 1, &[1.0]));
        assert_eq!(d, DVector::from_row_slice(&[2.0]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
    }

    #[test]
    fn test_2() {
        let mut L = DMatrix::<f64>::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[1.0, 1.0]);
        let alp = -0.5;
        let mut u = DVector::from_row_slice(&[1.0, 1.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<2>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, 1.0,
        ]));

        assert_eq!(d, DVector::from_row_slice(&[0.5, 0.75]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
    }

    #[test]
    fn test_3() {
        let mut L = DMatrix::<f64>::from_row_slice(1, 1, &[1.0]);
        let mut d = DVector::from_row_slice(&[2.0]);
        let alp = 1.0;
        let mut u = DVector::from_row_slice(&[2.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<1>(&mut L_view, &mut d_view, alp, &mut u);


        assert_eq!(L, DMatrix::<f64>::from_row_slice(1, 1, &[1.0]));
        assert_eq!(d, DVector::from_row_slice(&[6.0]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
    }

    #[test]
    fn test_4() {
        let mut L = DMatrix::<f64>::from_row_slice(1, 1, &[1.0]);
        let mut d = DVector::from_row_slice(&[2.0]);
        let alp = -1.0;
        let mut u = DVector::from_row_slice(&[2.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<1>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(1, 1, &[1.0]));
        assert_eq!(d, DVector::from_row_slice(&[2.0]));
        assert_eq!(p, DVector::from_row_slice(&[1.0]));
    }

    #[test]
    fn test_5() {
        let mut L = DMatrix::<f64>::from_row_slice(2, 2, &[0.0132305353535357669842, -0.4234767083209478906, 0.5000000000001, 1.50000034343434340000001]);
        let mut d = DVector::from_row_slice(&[2.3133333333334354666, 3.77777777909802358971908208]);
        let alp = 1.131313131313131313131313133146024;
        let mut u = DVector::from_row_slice(&[1., 1.]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<2>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(2, 2, &[0.013230535353535766, -0.4234767083209479, 0.6642132426251283, 1.5000003434343434]));
        assert_eq!(d, DVector::from_row_slice(&[3.444646464646567, 3.96771776306761]));
        assert_eq!(p, DVector::from_row_slice(&[]));
    }

    #[test]
    fn test_real_mistake() {
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0,
            0.3, 0.2, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[2.0, 3.0, 1.0]);
        let alp = 1.5;
        let mut u = DVector::from_row_slice(&[1.0, 0.0, 2.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.2857142857142857, 1.0, 0.0, 1.0285714285714285, 0.2, 1.0]));
        assert_eq!(d, DVector::from_row_slice(&[3.5, 3.0, 3.477142857142857]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
        assert_eq!(u, DVector::from_row_slice(&[1., -0.5, 1.7]))
    }

    #[test]
    fn test_zero_alpha_initial() {
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0,
            0.3, 0.2, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[2.0, 3.0, 1.0]);
        let alp = 0.0;
        let mut u = DVector::from_row_slice(&[1.0, 0.0, 2.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.3, 0.2, 1.0, ]));
        assert_eq!(d, DVector::from_row_slice(&[2.0, 3.0, 1.0]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
        assert_eq!(u, DVector::from_row_slice(&[1.0, 0.0, 2.0]))
    }

    #[test]
    fn test_alpha_becomes_zero() {
        // Test case where alpha becomes zero during execution (should trigger the break)
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0,
            0.3, 0.2, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[1.0, 1e-16, 1.0]);
        let alp = 1e-16;
        let mut u = DVector::from_row_slice(&[1.0, 1.0, 1.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1., 0., 0., 0.5000000000000001, 1., 0., 0.3000000000000001, 0.43999999999999995, 1.]));
        assert_eq!(d, DVector::from_row_slice(&[1.00, 1.25e-16, 1.00]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
        assert_eq!(u, DVector::from_row_slice(&[1., 0.5, 0.6]))
    }

    #[test]
    fn test_large_matrix() {
        // Test with a larger matrix
        let n = 5;
        let mut L = DMatrix::<f64>::identity(n, n);
        let mut d = DVector::from_element(n, 1.0);
        let alp = 2.0;
        let mut u = DVector::from_element(n, 1.0);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<5>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(5, 5, &[1., 0., 0., 0., 0., 0.6666666666666666, 1., 0., 0., 0., 0.6666666666666666, 0.4, 1., 0., 0., 0.6666666666666666, 0.4, 0.28571428571428575, 1., 0., 0.6666666666666666, 0.4, 0.28571428571428575, 0.22222222222222224, 1.]));
        assert_eq!(d, DVector::from_row_slice(&[3., 1.6666666666666665, 1.4, 1.2857142857142858, 1.2222222222222223]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
        assert_eq!(u, DVector::from_row_slice(&[1., 1., 1., 1., 1.]))
    }

    #[test]
    fn test_python_mistake() { // TODO: why python version is wrong?
        let mut L = DMatrix::<f64>::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.51, 1.0, 0.0,
            0.32, 0.23, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[2.0, 1e-10, 1.0]);
        let alp = -1.0;
        let mut u = DVector::from_row_slice(&[0.1, 1.0, 0.1]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0000, 0., 0., 0.5100, 1.0000, 0., 0.3200, 0.2300, 1.0000]));
        assert_eq!(d, DVector::from_row_slice(&[2.0, 1.00e-10, 1.00]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[-0.4623115577889447, 1.0000, 0.]));
        assert_eq!(u, DVector::from_row_slice(&[0.1, 0.949, 0.068]))
    }

    #[test]
    fn test_mixed_signs() {
        // Test with mixed signs in u vector
        let mut L = DMatrix::<f64>::from_row_slice(4, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0,
            0.1, 0.4, 0.6, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[2.0, 3.0, 1.0, 4.0]);
        let alp = 2.5;
        let mut u = DVector::from_row_slice(&[1.0, -2.0, 3.0, -4.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<4>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(4, 4, &[1.0, 0.0, 0.0, 0.0, -0.888888888888889, 1.0, 0.0, 0.0, 1.8, -0.693854748603352, 1.0, 0.0, -2.177777777777778, 1.265921787709497, -0.6148222838416939, 1.0]));
        assert_eq!(d, DVector::from_row_slice(&[4.5, 9.944444444444445, 4.432402234636872, 5.905752457776657]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[]));
        assert_eq!(u, DVector::from_row_slice(&[1., -2.5, 3.2, -5.02]))
    }

    #[test]
    fn test_borderline_negative_curvature() {
        let n = 3;
        let mut L = DMatrix::<f64>::identity(n, n);
        let mut d = DVector::from_element(n, 1.0);
        let alp = -1.0;
        let mut u = DVector::<f64>::from_row_slice(&[1.0, 1.0, 1.0]);

        // Set d[1] to a value that will make delta exactly at the threshold
        let neps = n as f64 * f64::EPSILON;
        d[1] = neps - alp * u[1].powi(2); // This will make delta exactly neps

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<3>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
        assert_eq!(d, DVector::from_row_slice(&[1., 1.0000000000000007, 1.]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[1., 0., 0.]));
        assert_eq!(u, DVector::from_row_slice(&[1., 1., 1.]))
    }

    #[test]
    fn test_hard() {
        let mut L = DMatrix::<f64>::from_row_slice(4, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0,
            0.1, 0.4, 0.6, 1.0,
        ]);
        let mut d = DVector::from_row_slice(&[2.0, 3.0, 1.0, 4.0]);
        let alp = -2.5;
        let mut u = DVector::from_row_slice(&[1.0, -2.0, 3.0, -4.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<4>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(4, 4, &[1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.3, 0.2, 1.0, 0.0, 0.1, 0.4, 0.6, 1.0]));
        assert_eq!(d, DVector::from_row_slice(&[2., 3., 1., 4.]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[1., 0., 0., 0.]));
        assert_eq!(u, DVector::from_row_slice(&[1., -2., 3., -4.]))
    }

    #[test]
    fn test_python_mistake_2() { // TODO: why python version is wrong?
        let mut L = DMatrix::<f64>::from_row_slice(4, 4, &[
            1.1, 0.0, 0.0, 0.0,
            -0.3, 1.0, 0.0, 0.0,
            0.4, 0.2, 1.0, 0.0,
            -0.5, 0.23, 0.12, 1.43,
        ]);
        let mut d = DVector::from_row_slice(&[1.1, -3.0, -1.0, -4.0]);
        let alp = -1.7;
        let mut u = DVector::from_row_slice(&[0.0, -2.0, -3.0, -4.0]);

        let mut L_view = MatrixViewMut::from(&mut L);
        let mut d_view = MatrixViewMut::from(&mut d);

        let p = ldlrk1::<4>(&mut L_view, &mut d_view, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(4, 4, &[1.1, 0.0, 0.0, 0.0, -0.3, 1.0, 0.0, 0.0, 0.4, 0.2, 1.0, 0.0, -0.5, 0.23, 0.12, 1.43, ]));
        assert_eq!(d, DVector::from_row_slice(&[1.1, -3.0, -1.0, -4.0]));
        assert_eq!(p, DVector::<f64>::from_row_slice(&[0.2727272727272727, 1.0000, 0., 0.]));
        assert_eq!(u, DVector::from_row_slice(&[0.0, -2.0, -3.0, -4.0]))
    }
}
