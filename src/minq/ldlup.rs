use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{DMatrix, DVector};

pub fn ldlup(
    mut L: Vec<Vec<f64>>,
    mut d: Vec<f64>,
    j: usize,
    g: Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let eps = 2.2204e-16;
    let n = d.len();

    // Create index vectors I and K
    let I: Vec<usize> = (0..j).collect();
    let K: Vec<usize> = (j + 1..n).collect();

    // Special case for j == 0
    if j == 0 {
        let delta = g[j];

        if delta <= (n as f64) * eps {
            let mut p = vec![0.0; n];
            p[0] = 1.0;
            return (L, d, p);
        }

        // Clear row j (equivalent to L[j,I] = v.T in Python)
        for i in I {
            L[0][i] = 0.0;
        }
        d[0] = delta;

        return (L, d, vec![]);
    }

    // Extract LII submatrix
    let mut lii_data = Vec::new();
    for &i in &I {
        for &j in &I {
            lii_data.push(L[i][j]);
        }
    }
    let LII = DMatrix::from_vec(I.len(), I.len(), lii_data);

    // Extract gI vector
    let gI: Vec<f64> = I.iter().map(|&i| g[i]).collect();
    let gI = DVector::from_vec(gI);

    // Solve LII * u = gI
    let u = LII.clone().lu().solve(&gI).unwrap();

    // Calculate v
    let dI: Vec<f64> = I.iter().map(|&i| d[i]).collect();
    let v: DVector<f64> = u.component_div(&DVector::from_vec(dI));

    // Calculate delta
    let delta = g[j] - u.dot(&v);

    if delta <= (n as f64) * eps {
        let v_solve = LII.transpose().clone().lu().solve(&v).unwrap();
        let mut p = vec![0.0; n];
        for (i, val) in v_solve.iter().enumerate() {
            p[i] = *val;
        }
        p[j] = -1.0;
        return (L, d, p);
    }

    let mut q = Vec::new();

    if !K.is_empty() {
        // Extract LKI submatrix
        let mut lki_data = Vec::new();
        for &k in &K {
            for &i in &I {
                lki_data.push(L[k][i]);
            }
        }
        let LKI = DMatrix::from_vec(K.len(), I.len(), lki_data);

        // Calculate w
        let gK: Vec<f64> = K.iter().map(|&k| g[k]).collect();
        let gK = DVector::from_vec(gK);
        let w = (gK - &LKI * &u) / delta;

        // Extract LKK submatrix
        let mut lkk_data = Vec::new();
        for &k1 in &K {
            for &k2 in &K {
                lkk_data.push(L[k1][k2]);
            }
        }
        let LKK = DMatrix::from_vec(K.len(), K.len(), lkk_data);

        let dK: Vec<f64> = K.iter().map(|&k| d[k]).collect();

        // Call ldlrk1
        let (lkk_new, dk_new, q_new) = ldlrk1(
            LKK.as_slice().chunks(K.len()).map(|row| row.to_vec()).collect(),
            dK,
            -delta,
            w.as_slice().to_vec(),
        );

        // Update L and d with results
        for (i, &k1) in K.iter().enumerate() {
            for (j, &k2) in K.iter().enumerate() {
                L[k1][k2] = lkk_new[i][j];
            }
            d[k1] = dk_new[i];
        }

        q = q_new;
    }

    if q.is_empty() {
        // Create new rows for L matrix
        let mut new_L = vec![vec![0.0; n]; n];

        // Copy original rows for I indices
        for &i in &I {
            new_L[i] = L[i].clone();
        }

        // Update row j
        for i in 0..n {
            new_L[j][i] = if i < j {
                v[i]
            } else if i == j {
                1.0
            } else {
                L[j][i]
            };
        }

        // Update rows for K indices
        for &k in &K {
            for i in 0..n {
                if i <= j {
                    new_L[k][i] = if i < j {
                        L[k][i]
                    } else {
                        g[k] / delta
                    };
                } else {
                    new_L[k][i] = L[k][i];
                }
            }
        }

        L = new_L;
        d[j] = delta;
        (L, d, vec![])
    } else {
        // Create new L matrix
        let mut new_L = vec![vec![0.0; n]; n];

        // Copy rows 0 to j
        for i in 0..=j {
            new_L[i] = L[i].clone();
        }

        // Update remaining rows
        for (idx, &k) in K.iter().enumerate() {
            // Copy LKI part
            for (j_idx, &i) in I.iter().enumerate() {
                new_L[k][i] = L[k][i];
            }
            // Set column j
            new_L[k][j] = g[k] / delta;
            // Copy LKK part
            for (j_idx, &k2) in K.iter().enumerate() {
                new_L[k][k2] = L[k][k2];
            }
        }

        L = new_L;

        let w = DVector::from_vec(K.iter().map(|&k| g[k] / delta).collect());
        let q_vec = DVector::from_vec(q.clone());

        let pi = w.dot(&q_vec);
        let piv = &v * pi;

        let lki_data: Vec<f64> = K.iter().flat_map(|&k| I.iter().map({
            let value = L.clone();
            move |&i| value[k][i]
        })).collect();
        let LKI = DMatrix::from_vec(K.len(), I.len(), lki_data);

        let lki_q = &LKI.transpose() * &q_vec;
        let piv_lki_q = piv - lki_q;

        let p_solve = LII.transpose().clone().lu().solve(&piv_lki_q).unwrap();

        let mut p = Vec::with_capacity(n);
        p.extend(p_solve.iter());
        p.push(-pi);
        p.extend(q.iter());

        (L, d, p)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_j_start_row() {
        // Initialize L as a 2x2 matrix
        let L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];

        // Initialize vectors
        let d = vec![1.0, 2.0];
        let j = 0;
        let g = vec![1.0, 0.0];

        // Call the function
        let (L_new, d_new, p) = ldlup(L, d, j, g);
        let L_expected = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];
        let d_expected = vec![1.0, 2.0];
        let p_expected: Vec<f64> = vec![]; // Empty vector

        // Assert results using relative comparison for floating point numbers
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    L_new[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d_new[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }

    #[test]
    fn test_j_last_row() {
        // Initialize L as a 2x2 matrix
        let L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];

        // Initialize vectors
        let d = vec![1.0, 2.0];
        let j = 1;
        let g = vec![0.5, 2.0];

        // Call the function
        let (L_new, d_new, p) = ldlup(L, d, j, g);

        // Expected results
        let L_expected = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];
        let d_expected = vec![1.0, 1.75];
        let p_expected: Vec<f64> = vec![];

        // Assert results using relative comparison for floating point numbers
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    L_new[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d_new[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }

    #[test]
    fn test_large_matrix_size() {
        // Initialize L as a 2x2 matrix
        let L = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0, 0.0],
            vec![0.3, 0.6, 1.0, 0.0],
            vec![0.4, 0.7, 0.8, 1.0],
        ];

        // Initialize vectors
        let d = vec![1.0, 2.0, 3.0, 4.0];
        let j = 2;
        let g = vec![0.0, 0.0, 3.0, 0.0];

        // Call the function
        let (L_new, d_new, p) = ldlup(L, d, j, g);

        // Expected results
        let L_expected = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0, 0.0],
            vec![0.3, 0.6, 1.0, 0.0],
            vec![0.4, 0.7, 0.8, 1.0],
        ];
        let d_expected = vec![1.0, 2.0, 3.0, 4.0];
        let p_expected: Vec<f64> = vec![];

        // Assert results using relative comparison for floating point numbers
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    L_new[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d_new[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }

    #[test]
    fn test_3() {
        // Initialize L as a 2x2 matrix
        let L = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0, 0.0],
            vec![0.3, 0.6, 1.0, 0.0],
            vec![0.4, 0.7, 0.8, 1.0],
        ];

        // Initialize vectors
        let d = vec![1.0, 2.0, 3.0, 4.0];
        let j = 2;
        let g = vec![-100.0, 0.0, 3.0, 0.0];

        // Call the function
        let (L_new, d_new, p) = ldlup(L, d, j, g);

        // Expected results
        let L_expected = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0, 0.0],
            vec![0.3, 0.6, 1.0, 0.0],
            vec![0.4, 0.7, 0.8, 1.0],
        ];
        let d_expected = vec![1.0, 2.0, 3.0, 4.0];
        let p_expected: Vec<f64> = vec![-102.0, 10.0, -1.0, 0.0];

        // Assert results using relative comparison for floating point numbers
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    L_new[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d_new[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }
}