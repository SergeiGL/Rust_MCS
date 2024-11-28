use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{DMatrix, DVector};

pub fn ldlup(
    L: &mut Vec<Vec<f64>>,
    d: &mut Vec<f64>,
    j: usize,
    g: &[f64],
) ->
    Vec<f64>
{
    let mut p = vec![];
    let eps = 2.2204e-16;
    let n = d.len();

    // Create index vectors I and K
    let I: Vec<usize> = (0..j).collect();
    let K: Vec<usize> = ((j + 1)..n).collect();

    // Declare variables at function scope
    let mut q = Vec::new();
    let mut w_vec = Vec::new();
    let mut dK = Vec::new();
    let delta;
    let u;
    let v;
    let LII;
    let mut LKI = DMatrix::zeros(0, 0);

    // Special case for j == 0
    if j == 0 {
        delta = g[j];

        if delta <= (n as f64) * eps {
            p = vec![0.0; n];
            p[0] = 1.0;
            return p;
        }

        for i in I {
            L[0][i] = 0.0;
        }
        d[0] = delta;
        return p;
    }


    LII = DMatrix::from_fn(j, j, |row, col| L[row][col]);

    let gI = DVector::from_fn(j, |i, _| g[i]);

    // Solve LII * u = gI
    u = LII.clone().lu().solve(&gI).unwrap();

    // Calculate v
    v = u.component_div(&DVector::from_fn(j, |i, _| d[i]));

    // Calculate delta
    delta = g[j] - u.dot(&v);

    if delta <= (n as f64) * eps {
        p = vec![0.0; n];
        for (i, val) in LII.clone()
            .transpose()
            .lu()
            .solve(&v)
            .unwrap()
            .iter()
            .enumerate() {
            p[i] = *val;
        }
        p[j] = -1.0;
        return p;
    }

    if !K.is_empty() {
        LKI = DMatrix::from_fn(K.len(), I.len(), |row, col| L[K[row]][col]);

        // Extract gK vector
        let gK = DVector::from_fn(K.len(), |i, _| g[K[i]]);

        // Calculate w
        let w = (gK - &LKI * u).scale(1.0 / delta);
        w_vec = w.data.as_vec().to_vec();


        let mut LKK = Vec::with_capacity(K.len());
        for &k1 in &K {
            LKK.push(K.iter().map(|&k2| L[k1][k2]).collect::<Vec<_>>());
        }

        // Extract dK
        dK = K.iter().map(|&k| d[k]).collect();

        // Call ldlrk1
        q = ldlrk1(&mut LKK, &mut dK, -delta, &mut w_vec);

        // Update d
        for (i, &k) in K.iter().enumerate() {
            d[k] = dK[i];
        }
    } else {
        q = vec![];
    }

    if !K.is_empty() && q.is_empty() { //TODO: strange
        // Update second row (j-th row)
        let v_vec = v.data.as_vec();
        for (idx, &i) in I.iter().enumerate() {
            L[j][i] = v_vec[idx];
        }
        L[j][j] = 1.0;

        // Update L[K,j] with w values
        for (k_idx, &k) in K.iter().enumerate() {
            L[k][j] = w_vec[k_idx];
        }

        d[j] = delta;
    } else if !K.is_empty() { //TODO: strange
        // Calculate final p vector
        let w = DVector::from_vec(w_vec);
        let q_vec = DVector::from_vec(q);
        let pi = w.dot(&q_vec);
        let piv = v.scale(pi);
        let lki_q = &LKI.transpose() * &q_vec;
        let piv_lki_q = piv - lki_q;
        let pi_solve = LII.transpose().lu().solve(&piv_lki_q).unwrap();

        p = Vec::with_capacity(n);
        p.extend(pi_solve.iter());
        p.push(-pi);
        p.extend(q_vec.iter());
    } else { //TODO: strange
        let v_vec = v.data.as_vec();
        for (idx, &i) in I.iter().enumerate() {
            L[j][i] = v_vec[idx];
        }
        L[j][j] = 1.0;
        d[j] = delta;
    }

    p
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mistake_0() {
        let mut L = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ];
        let mut d = vec![23.127986525843045, 18.57643856292412, 1.0, 1.0, 1.0, 1.0];
        let j = 2;
        let g = vec![1.7538162952525622, -0.5909985456367551, 24.55657908379219, 0.0, 0.0, 0.0];

        let p = ldlup(&mut L, &mut d, j, &g);

        assert_eq!(L, vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.07583091132005203, -0.03214451416643742, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]);

        assert_eq!(d, vec![23.127986525843045, 18.57643856292412, 24.40439112304386, 1.0, 1.0, 1.0]);
        assert_eq!(p, vec![]);
    }

    #[test]
    fn test_j_start_row() {
        let mut L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];

        let mut d = vec![1.0, 2.0];
        let j = 0;
        let g = vec![1.0, 0.0];

        let p = ldlup(&mut L, &mut d, j, &g);

        let L_expected = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];
        let d_expected = vec![1.0, 2.0];
        let p_expected: Vec<f64> = vec![]; // Empty vector

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    L[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }

    #[test]
    fn test_j_last_row() {
        // Initialize L as a 2x2 matrix
        let mut L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];

        // Initialize vectors
        let mut d = vec![1.0, 2.0];
        let j = 1;
        let g = vec![0.5, 2.0];

        // Call the function
        let p = ldlup(&mut L, &mut d, j, &g);

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
                    L[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }

    #[test]
    fn test_large_matrix_size() {
        // Initialize L as a 2x2 matrix
        let mut L = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0, 0.0],
            vec![0.3, 0.6, 1.0, 0.0],
            vec![0.4, 0.7, 0.8, 1.0],
        ];

        // Initialize vectors
        let mut d = vec![1.0, 2.0, 3.0, 4.0];
        let j = 2;
        let g = vec![0.0, 0.0, 3.0, 0.0];

        // Call the function
        let p = ldlup(&mut L, &mut d, j, &g);

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
                    L[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }

    #[test]
    fn test_3() {
        // Initialize L as a 2x2 matrix
        let mut L = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0, 0.0],
            vec![0.3, 0.6, 1.0, 0.0],
            vec![0.4, 0.7, 0.8, 1.0],
        ];

        // Initialize vectors
        let mut d = vec![1.0, 2.0, 3.0, 4.0];
        let j = 2;
        let g = vec![-100.0, 0.0, 3.0, 0.0];

        // Call the function
        let p = ldlup(&mut L, &mut d, j, &g);

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
                    L[i][j],
                    L_expected[i][j],
                    epsilon = 1e-10
                );
            }
        }

        for i in 0..2 {
            assert_relative_eq!(
                d[i],
                d_expected[i],
                epsilon = 1e-10
            );
        }

        assert_eq!(p.len(), p_expected.len());
    }
}