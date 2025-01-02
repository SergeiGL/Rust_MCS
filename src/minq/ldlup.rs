use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

pub fn ldlup<const N: usize>(
    L: &mut SMatrix<f64, N, N>,
    d: &mut SVector<f64, N>,
    j: usize,
    g: &SVector<f64, N>,
) ->
    Option<SVector<f64, N>> // p
{
    let mut p_option: Option<SVector<f64, N>> = None;
    let eps = 2.2204e-16;

    // Create index vectors I and K
    let I: Vec<usize> = (0..j).collect();
    let K: Vec<usize> = ((j + 1)..N).collect();

    // Special case for j == 0
    if j == 0 {
        let delta = g[j];
        if delta <= (N as f64) * eps {
            p_option = Some(SVector::<f64, N>::zeros());
            if let Some(ref mut p_vect) = p_option {
                p_vect[0] = 1.0;
            }
            return p_option;
        }

        for &i in &I {
            L[(0, i)] = 0.0;
        }
        d[0] = delta;
        return p_option;
    }

    let LII = DMatrix::from_fn(j, j, |row, col| L[(row, col)]);
    let gI = DVector::from_fn(j, |i, _| g[i]);
    let u = LII.clone().lu().solve(&gI).unwrap();

    let v = u.component_div(&DVector::from_fn(j, |i, _| d[i]));

    let delta = g[j] - u.dot(&v);

    if delta <= (N as f64) * eps {
        // Negative or zero curvature
        p_option = Some(SVector::<f64, N>::zeros());
        let p1 =
            LII.transpose()
                .lu()
                .solve(&v)
                .unwrap()
                .iter()
                .cloned()
                .collect::<Vec<f64>>();
        if let Some(ref mut p_vect) = p_option {
            for (i, &val) in p1.iter().enumerate() {
                p_vect[i] = val;
            }
            p_vect[j] = -1.0;
        }

        return p_option;
    }


    let (q, w, LKI) = if !K.is_empty() {
        let LKI = DMatrix::from_fn(K.len(), I.len(), |row, col| L[(K[row], col)]);

        let gK = DVector::from_fn(K.len(), |i, _| g[K[i]]);

        let mut w = (gK - (&LKI * &u)).scale(1.0 / delta);

        let mut LKK = DMatrix::from_fn(K.len(), K.len(), |row, col| L[(K[row], K[col])]);

        let mut dK = DVector::from_fn(K.len(), |i, _| d[K[i]]);

        let q = ldlrk1(&mut LKK, &mut dK, -delta, &mut w);

        for (i, &k) in K.iter().enumerate() {
            d[k] = dK[i];
        }
        (q, w, LKI)
    } else {
        (
            DVector::<f64>::zeros(0),
            DVector::<f64>::zeros(0),
            DMatrix::<f64>::zeros(0, 0),
        )
    };

    // TODO: strange
    if !K.is_empty() && q.is_empty() {
        for (idx, &i) in I.iter().enumerate() {
            L[(j, i)] = v[idx];
        }
        L[(j, j)] = 1.0;

        // Update L[K,j] with w values
        for (k_idx, &k) in K.iter().enumerate() {
            L[(k, j)] = w[k_idx];
        }

        d[j] = delta;
    } else if !K.is_empty() {
        let pi = w.dot(&q);
        let piv = v.scale(pi);
        let lki_q = &LKI.transpose() * &q;
        let piv_lki_q = piv - lki_q;
        let pi_solve = LII.transpose().lu().solve(&piv_lki_q).unwrap();

        // Construct the SVector from the combined iterator
        p_option = Some(SVector::from_iterator(
            pi_solve.iter().copied()
                .chain(std::iter::once(-pi))
                .chain(q.iter().copied())
        ));
    } else {
        for (idx, &i) in I.iter().enumerate() {
            L[(j, i)] = v[idx];
        }
        L[(j, j)] = 1.0;
        d[j] = delta;
    }
    p_option
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_0() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.0, 0.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            1.23, -1.0, 1.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.0, 0.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            f64::INFINITY, f64::NEG_INFINITY, -1.0,
        ]));
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 2);
    }

    #[test]
    fn test_coverage_1() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.2,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.3, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.2,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            2.1111111111111103, -6.666666666666663, 0.1111111111111111,
        ]));
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_coverage_2() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.2,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.3, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -0.28888888888888886, -0.08943089430894309, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.1420234869015357,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 2);
    }


    #[test]
    fn test_coverage_3() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 3.0,
            4.0, -1.0, 6.0,
            -8.0, 8.0, 8.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 0.0, 0.2,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 3.0,
            4.0, -1.0, 6.0,
            -8.0, 8.0, 8.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 0.0, 0.2,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            1.0, -0.375, 0.125,
        ]));
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 1);
    }


    #[test]
    fn test_coverage_4() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let j = 0;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            1.0, 0.0, 0.0,
        ]));
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 0);
    }
    #[test]
    fn test_coverage_5() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            -0.3333333333333333, -1.0, 0.3333333333333333,
        ]));
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_coverage_6() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            0.06666666666666667, 0.1, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.2833333333333333,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 2);
    }


    #[test]
    fn test_mistake_0() {
        const N: usize = 6;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            23.127986525843045, 18.57643856292412, 1.0, 1.0, 1.0, 1.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            1.7538162952525622, -0.5909985456367551, 24.55657908379219, 0.0, 0.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.07583091132005203, -0.03214451416643742, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            23.127986525843045, 18.57643856292412, 24.40439112304386, 1.0, 1.0, 1.0,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }


    #[test]
    fn test_j_start_row() {
        const N: usize = 2;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0,
        ]);
        let j = 0;
        let g = SVector::<f64, N>::from_row_slice(&[
            1.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 0);
    }


    #[test]
    fn test_j_last_row() {
        const N: usize = 2;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.5, 2.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 1.75,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }


    #[test]
    fn test_large_matrix_size() {
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.3, 0.6, 1.0, 0.0,
            0.4, 0.7, 0.8, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.0, 3.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.4, 0.7, 0.0, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(g, SVector::<f64, N>::from_row_slice(&[
            0.0, 0.0, 3.0, 0.0,
        ]));
    }


    #[test]
    fn test_3() {
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.3, 0.6, 1.0, 0.0,
            0.4, 0.7, 0.8, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            -100.0, 0.0, 3.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.3, 0.6, 1.0, 0.0,
            0.4, 0.7, 0.8, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            -102.0, 10.0, -1.0, 0.0,
        ]));
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }
}