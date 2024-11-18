use crate::minq::ldlrk1::ldlrk1;

pub fn ldldown(mut L: Vec<Vec<f64>>, mut d: Vec<f64>, j: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = d.len();

    if j < n {
        // Create index vectors
        let I: Vec<usize> = (0..j).collect();
        let K: Vec<usize> = ((j + 1)..n).collect();

        if !K.is_empty() {
            // Extract LKK submatrix
            let mut LKK = vec![vec![0.0; K.len()]; K.len()];
            for (i_idx, &i) in K.iter().enumerate() {
                for (j_idx, &k) in K.iter().enumerate() {
                    LKK[i_idx][j_idx] = L[i][k];
                }
            }

            // Extract dK vector
            let mut dK = vec![0.0; K.len()];
            for (i_idx, &i) in K.iter().enumerate() {
                dK[i_idx] = d[i];
            }

            // Extract LKj vector
            let mut LKj = vec![0.0; K.len()];
            for (i_idx, &i) in K.iter().enumerate() {
                LKj[i_idx] = L[i][j];
            }

            // Call ldlrk1
            let (LKK_new, dK_new, _) = ldlrk1(LKK, dK, d[j], LKj);

            // Update d vector with new values
            for (i_idx, &i) in K.iter().enumerate() {
                d[i] = dK_new[i_idx];
            }

            // Create new L matrix
            let mut L_new = vec![vec![0.0; n]; n];

            // Copy original values for I rows if they exist
            if !I.is_empty() {
                for &i in &I {
                    for j in 0..n {
                        L_new[i][j] = L[i][j];
                    }
                }
            }

            // Set row j to zeros
            for k in 0..n {
                L_new[j][k] = 0.0;
            }

            if !K.is_empty() {
                // Copy LKI if I is not empty
                if !I.is_empty() {
                    for (k_idx, &k) in K.iter().enumerate() {
                        for &i in &I {
                            L_new[k][i] = L[k][i];
                        }
                    }
                }

                // Set the new LKK values
                for (i_idx, &i) in K.iter().enumerate() {
                    for (j_idx, &k) in K.iter().enumerate() {
                        L_new[i][k] = LKK_new[i_idx][j_idx];
                    }
                }
            }

            L = L_new;
        }

        L[j][j] = 1.0;
    } else {
        for i in 0..n - 1 {
            L[n - 1][i] = 0.0;
        }
    }
    d[j] = 1.0;

    (L, d)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    fn assert_matrix_eq(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].len(), b[0].len());
        for i in 0..a.len() {
            for j in 0..a[i].len() {
                assert_relative_eq!(a[i][j], b[i][j], epsilon = 1e-10);
            }
        }
    }

    fn assert_vector_eq(a: &Vec<f64>, b: &Vec<f64>) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_relative_eq!(a[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_general_case() {
        let L = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.0, 1.0, 0.2],
            vec![0.0, 0.0, 1.0]
        ];
        let d = vec![2.0, 3.0, 4.0];
        let j = 1;

        let expected_L = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0]
        ];
        let expected_d = vec![2.0, 1.0, 4.0];

        let (L_new, d_new) = ldldown(L, d, j);
        assert_matrix_eq(&L_new, &expected_L);
        assert_vector_eq(&d_new, &expected_d);
    }

    #[test]
    fn test_min_j() {
        let L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let d = vec![2.0, 3.0];
        let j = 0;

        let expected_L = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![1.0, 3.0];

        let (L_new, d_new) = ldldown(L, d, j);
        assert_matrix_eq(&L_new, &expected_L);
        assert_vector_eq(&d_new, &expected_d);
    }

    #[test]
    fn test_max_j() {
        let L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let d = vec![2.0, 3.0];
        let j = 1;

        let expected_L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![2.0, 1.0];

        let (L_new, d_new) = ldldown(L, d, j);
        assert_matrix_eq(&L_new, &expected_L);
        assert_vector_eq(&d_new, &expected_d);
    }

    #[test]
    fn test_large_values() {
        let L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let d = vec![1e10, 1e12];
        let j = 0;

        let expected_L = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![1.0, 1e12];

        let (L_new, d_new) = ldldown(L, d, j);
        assert_matrix_eq(&L_new, &expected_L);
        assert_vector_eq(&d_new, &expected_d);
    }

    #[test]
    fn test_fractional_values() {
        let L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let d = vec![0.2, 0.5];
        let j = 1;

        let expected_L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![0.2, 1.0];

        let (L_new, d_new) = ldldown(L, d, j);
        assert_matrix_eq(&L_new, &expected_L);
        assert_vector_eq(&d_new, &expected_d);
    }
}