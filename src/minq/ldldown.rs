use crate::minq::ldlrk1::ldlrk1;

pub fn ldldown(L: &mut Vec<Vec<f64>>, d: &mut Vec<f64>, j: usize) {
    let n = d.len();

    if j < n {
        // Create index vectors
        // let I: Vec<usize> = (0..j).collect();
        let K: Vec<usize> = ((j + 1)..n).collect();

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
        ldlrk1(&mut LKK, &mut dK, d[j], &mut LKj);

        // d[K] = dK
        for (i_idx, &i) in K.iter().enumerate() {
            d[i] = dK[i_idx];
        }

        // Set row j to zeros
        for k in 0..n {
            L[j][k] = 0.0;
        }

        if !K.is_empty() {
            // Set the new LKK values
            for (i_idx, &i) in K.iter().enumerate() {
                for (j_idx, &k) in K.iter().enumerate() {
                    L[i][k] = LKK[i_idx][j_idx];
                }
            }
        }

        L[j][j] = 1.0;
    } else {
        // if n >= 2 {
        //     L[n - 1][..n - 1].iter_mut().for_each(|x| *x = 0.0);
        // }
        panic!("Bad State")
    }
    d[j] = 1.0; // TODO: VERY STRANGE if j>n this will break
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let mut L = vec![vec![-1.0; 3]; 3];
        let mut d = vec![2.0, 3.0, 4.0];
        let j = 2;

        let expected_L = vec![
            vec![-1., -1., -1.],
            vec![-1., -1., -1.],
            vec![0., 0., 1.]];
        let expected_d = vec![2.0, 3.0, 1.0];

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_general_case() {
        let mut L = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.0, 1.0, 0.2],
            vec![0.0, 0.0, 1.0]
        ];
        let mut d = vec![2.0, 3.0, 4.0];
        let j = 1;

        let expected_L = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0]
        ];
        let expected_d = vec![2.0, 1.0, 4.0];

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_min_j() {
        let mut L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let mut d = vec![2.0, 3.0];
        let j = 0;

        let expected_L = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![1.0, 3.0];

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_max_j() {
        let mut L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let mut d = vec![2.0, 3.0];
        let j = 1;

        let expected_L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![2.0, 1.0];

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_large_values() {
        let mut L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let mut d = vec![1e10, 1e12];
        let j = 0;

        let expected_L = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![1.0, 1e12];

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_fractional_values() {
        let mut L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let mut d = vec![0.2, 0.5];
        let j = 1;

        let expected_L = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0]
        ];
        let expected_d = vec![0.2, 1.0];

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }
}