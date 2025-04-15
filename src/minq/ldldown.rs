use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{SMatrix, SVector};

/// Downdates an LDL^T factorization by replacing the j-th row and column with the j-th unit vector
///
/// This function modifies the LDL^T factorization of a matrix when the j-th row and column
/// are replaced by the j-th unit vector. This is a key operation in active-set methods for
/// quadratic programming.
///
/// # Arguments
/// * `L` - Lower triangular matrix with unit diagonal from the LDL^T factorization
/// * `d` - Vector containing the diagonal elements of D (must be positive)
/// * `j` - Index of the row/column to be replaced (0-based indexing)
///
/// # Mathematical Background
/// If A = LDL^T is the original factorization, this function computes the factorization
/// of a matrix A' which is identical to A except that the j-th row and column are replaced
/// by the j-th unit vector.
///
/// # Note
/// The function modifies both L and d in-place.
pub(super) fn ldldown<const N: usize>(
    L: &mut SMatrix<f64, N, N>,
    d: &mut SVector<f64, N>,
    j: usize,
) {
    let dj = d[j];

    // as L's region ((j + 1)..N, 0..j) has no chance to be modified, no need to SAVE it (like let LKI = L.view_range((j + 1)..N, 0..j).into_owned();)
    let LKj = L.view_range((j + 1)..N, j).into_owned();
    let mut LKK = L.view_range_mut((j + 1)..N, (j + 1)..N);
    let mut dK = d.view_range_mut((j + 1)..N, 0);

    ldlrk1(&mut LKK, &mut dK, dj, LKj);

    match j {
        0 => {
            L.fill_row(j, 0.0);
            L.fill_column(j, 0.0)
        }
        _ => {
            L.fill_row(j, 0.0);
            L.view_range_mut(j.., j).fill(0.0);
            // as L's region ((j + 1)..N, 0..j) has no chance to be modified, no need to RESTORE it (like L.view_range_mut((j + 1)..N, 0..j).copy_from(&LKI);)
        }
    }

    L[(j, j)] = 1.0;
    d[j] = 1.;
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_real_mistake() {
        let mut L = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1., 0., 0., 0., 0., 0.,
            0.018635196087984494, 1., 0., 0., 0., 0.,
            -0.00006744289208153283, -0.002280552111200885, 1., 0., 0., 0.,
            0.048606475490027994, -0.07510298414917427, -0.46396921208814274, 1., 0., 0.,
            0.00014656637804684393, 0.004372396897708983, 0.18136372644709786, 0.005054161476720696, 1., 0.,
            0.00934128801419616, -0.0016648858565015735, 0.4674067707226544, -0.017343086953306597, -0.5222650783107669, 1.
        ]);
        let mut d = SVector::<f64, 6>::from_row_slice(&[
            95.7784634229683, 44.981003582969294, 0.30827844591150716, 49.5399548476696, 0.6927734092514332, 79.93996295955547
        ]);
        let j = 2;

        let expected_L = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.018635196087984494, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.048606475490027994, -0.07510298414917427, 0.0, 1.0, 0.0, 0.0, 0.00014656637804684393, 0.004372396897708983, 0.0, 0.004524467461310853, 1.0, 0.0, 0.00934128801419616, -0.0016648858565015735, 0.0, -0.018667576757453973, -0.477600159127822, 1.0
        ]);
        let expected_d = SVector::<f64, 6>::from_row_slice(&[95.7784634229683, 44.981003582969294, 1.0, 49.60631715637313, 0.703163545389539, 80.03349478791684]);

        ldldown(&mut L, &mut d, j);

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_0() {
        let mut L = SMatrix::<f64, 3, 3>::from_row_slice(&[-1.0; 9]);
        let mut d = SVector::<f64, 3>::from_row_slice(&[2.0, 3.0, 4.0]);
        let j = 2;

        let expected_L = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -1., -1., -1.,
            -1., -1., -1.,
            0., 0., 1.
        ]);
        let expected_d = SVector::<f64, 3>::from_row_slice(&[2.0, 3.0, 1.0]);

        ldldown(&mut L, &mut d, j);

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_general_case() {
        let mut L = SMatrix::<f64, 3, 3>::from_row_slice(&[
            1.0, 0.5, 0.3,
            0.0, 1.0, 0.2,
            0.0, 0.0, 1.0
        ]);
        let mut d = SVector::<f64, 3>::from_row_slice(&[2.0, 3.0, 4.0]);
        let j = 1;

        let expected_L = SMatrix::<f64, 3, 3>::from_row_slice(&[
            1.0, 0.5, 0.3,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 3>::from_row_slice(&[2.0, 1.0, 4.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_min_j() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[2.0, 3.0]);
        let j = 0;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.0,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[1.0, 3.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_max_j() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[2.0, 3.0]);
        let j = 1;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[2.0, 1.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_large_values() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[1e10, 1e12]);
        let j = 0;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.0,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[1.0, 1e12]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_fractional_values() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[0.2, 0.5]);
        let j = 1;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[0.2, 1.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }


    #[test]
    fn test_ldldown_j_zero() {
        // Test case for j = 0
        let mut L = SMatrix::<f64, 3, 3>::new(
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0,
            0.3, 0.2, 1.0,
        );
        let mut d = SVector::<f64, 3>::new(2.0, 1.5, 1.0);

        let j = 0;

        let expected_L = SMatrix::<f64, 3, 3>::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.30000000000000004, 1.0,
        );
        let expected_d = SVector::<f64, 3>::new(1.0, 2.0, 1.06);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_ldldown_j_middle() {
        // Test case for j = 1 (middle of matrix)
        let mut L = SMatrix::<f64, 3, 3>::new(
            1.0, 0.01, 0.02,
            0.5, 1.0, 0.03,
            0.3, 0.2, 1.0,
        );
        let mut d = SVector::<f64, 3>::new(2.0, 1.5, 1.01);

        let j = 1;

        let expected_L = SMatrix::<f64, 3, 3>::new(
            1.0, 0.01, 0.02,
            0.0, 1.0, 0.0,
            0.3, 0.0, 1.0,
        );
        let expected_d = SVector::<f64, 3>::new(2.0, 1.0, 1.07);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_ldldown_j_last() {
        // Test case for j = n-1 (last element)
        let mut L = SMatrix::<f64, 3, 3>::new(
            1.0, 0.01, 0.02,
            0.5, 1.0, 0.03,
            0.3, 0.2, 1.0,
        );
        let mut d = SVector::<f64, 3>::new(2.0, 1.5, 1.01);

        let j = 2;

        let expected_L = SMatrix::<f64, 3, 3>::new(
            1.0, 0.01, 0.02,
            0.5, 1.0, 0.03,
            0.0, 0.0, 1.0,
        );
        let expected_d = SVector::<f64, 3>::new(2.0, 1.5, 1.);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_ldldown_2() {
        // Test with a larger matrix (5x5)
        let mut L = SMatrix::<f64, 5, 5>::new(
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0, 0.0,
            0.4, 0.3, 0.2, 1.0, 0.0,
            0.1, 0.2, 0.3, 0.4, 1.0,
        );
        let mut d = SVector::<f64, 5>::new(2.0, 1.5, 1.0, 0.8, 0.5);

        let j = 0;

        let expected_L = SMatrix::<f64, 5, 5>::new(
            1., 0., 0., 0., 0.,
            0., 1., 0., 0., 0.,
            0., 0.30000000000000004, 1., 0., 0.,
            0., 0.425, 0.25943396226415094, 1., 0.,
            0., 0.2, 0.28301886792452824, 0.3503801345512224, 1.,
        );
        let expected_d = SVector::<f64, 5>::new(1., 2., 1.06, 0.8624056603773586, 0.527220040474758);

        ldldown(&mut L, &mut d, j);
        assert_relative_eq!(L, expected_L, epsilon = TOLERANCE);
        assert_relative_eq!(d, expected_d, epsilon = TOLERANCE);
    }

    #[test]
    fn test_ldldown_3() {
        // Test with a larger matrix (5x5)
        let mut L = SMatrix::<f64, 5, 5>::new(
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0, 0.0,
            0.4, 0.3, 0.2, 1.0, 0.0,
            0.1, 0.2, 0.3, 0.4, 1.0,
        );
        let mut d = SVector::<f64, 5>::new(2.0, 1.5, 1.0, 0.8, 0.5);

        let j = 1;

        let expected_L = SMatrix::<f64, 5, 5>::new(
            1., 0., 0., 0., 0.,
            0., 1., 0., 0., 0.,
            0.3, 0., 1., 0., 0.,
            0.4, 0., 0.27358490566037735, 1., 0.,
            0.1, 0., 0.3396226415094339, 0.4147882873393723, 1.,
        );
        let expected_d = SVector::<f64, 5>::new(2., 1., 1.06, 0.8956603773584906, 0.5016380872129766);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }


    #[test]
    fn test_ldldown_larger_matrix() {
        // Test with a larger matrix (5x5)
        let mut L = SMatrix::<f64, 5, 5>::new(
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0, 0.0,
            0.4, 0.3, 0.2, 1.0, 0.0,
            0.1, 0.2, 0.3, 0.4, 1.0,
        );
        let mut d = SVector::<f64, 5>::new(2.0, 1.5, 1.0, 0.8, 0.5);

        let j = 2;

        let expected_L = SMatrix::<f64, 5, 5>::new(
            1., 0., 0., 0., 0.,
            0.5, 1., 0., 0., 0.,
            0., 0., 1., 0., 0.,
            0.4, 0.3, 0., 1., 0.,
            0.1, 0.2, 0., 0.45238095238095244, 1.,
        );
        let expected_d = SVector::<f64, 5>::new(2., 1.5, 1., 0.8400000000000001, 0.5460952380952381);

        ldldown(&mut L, &mut d, j);
        assert_relative_eq!(L, expected_L, epsilon = TOLERANCE);
        assert_relative_eq!(d, expected_d, epsilon = TOLERANCE);
    }

    #[test]
    fn test_ldldown_larger_last() {
        // Test with a larger matrix (5x5)
        let mut L = SMatrix::<f64, 5, 5>::new(
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0, 0.0,
            0.4, 0.3, 0.2, 1.0, 0.0,
            0.1, 0.2, 0.3, 0.4, 1.0,
        );
        let mut d = SVector::<f64, 5>::new(2.0, 1.5, 1.0, 0.8, 0.5);

        let j = 4;

        let expected_L = SMatrix::<f64, 5, 5>::new(
            1., 0., 0., 0., 0.,
            0.5, 1., 0., 0., 0.,
            0.3, 0.2, 1., 0., 0.,
            0.4, 0.3, 0.2, 1., 0.,
            0., 0., 0., 0., 1.,
        );
        let expected_d = SVector::<f64, 5>::new(2., 1.5, 1., 0.8, 1.);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }
}