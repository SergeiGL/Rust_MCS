use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};


pub fn ldldown<const N: usize>(
    L: &mut SMatrix<f64, N, N>,
    d: &mut SVector<f64, N>,
    j: usize,
) {
    let K: Vec<usize> = ((j + 1)..N).collect();

    let K_len = K.len();
    let mut LKK = DMatrix::<f64>::zeros(K_len, K_len);
    for (row, &i) in K.iter().enumerate() {
        for (col, &k) in K.iter().enumerate() {
            LKK[(row, col)] = L[(i, k)];
        }
    }

    let mut dK = DVector::<f64>::zeros(K_len);
    for (idx, &k) in K.iter().enumerate() {
        dK[idx] = d[k];
    }

    let mut LKj = DVector::<f64>::zeros(K_len);
    for (idx, &k) in K.iter().enumerate() {
        LKj[idx] = L[(k, j)];
    }

    ldlrk1(&mut LKK, &mut dK, d[j], &mut LKj);

    for (idx, &k) in K.iter().enumerate() {
        d[k] = dK[idx];
    }

    let mut new_L = DMatrix::<f64>::zeros(N, N);


    for i in 0..j {
        for col in 0..N {
            new_L[(i, col)] = L[(i, col)];
        }
    }


    // Assign r2 (a row of zeros)
    // It's already zeros in new_L, so no action needed

    // Construct r3 based on whether I is empty or not
    if j == 0 {
        // When I is empty, r3 consists of [zeros | LKK]
        for row in 0..K.len() {
            new_L[(j + 1 + row, 0)] = 0.0; // First column is zero
            for col in 0..K_len {
                new_L[(j + 1 + row, col + 1)] = LKK[(row, col)];
            }
        }
    } else {
        // When I is not empty, construct LKI and then r3
        let mut LKI = DMatrix::<f64>::zeros(K_len, j);
        for (row, &k) in K.iter().enumerate() {
            for i in 0..j {
                LKI[(row, i)] = L[(k, i)];
            }
        }

        if !K.is_empty() {
            for row in 0..K.len() {
                for i in 0..j {
                    new_L[(j + 1 + row, i)] = LKI[(row, i)];
                }
                // Assign the zero column
                new_L[(j + 1 + row, j)] = 0.0;
                // Assign LKK
                for col in 0..K_len {
                    new_L[(j + 1 + row, j + 1 + col)] = LKK[(row, col)];
                }
            }
        }
    }

    // Assign the top part (r1) remains unchanged if I is not empty
    // Assign the middle row (r2) is already zeros
    // Assign the lower part (r3) has been handled above

    // Finally, set L[j, j] = 1
    new_L[(j, j)] = 1.0;

    // Now, copy new_L back to L
    for i in 0..N {
        for k in 0..N {
            L[(i, k)] = new_L[(i, k)];
        }
    }

    d[j] = 1.;
}


#[cfg(test)]
mod tests {
    use super::*;

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
}