use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{SMatrix, SVector};


pub fn ldlup<const N: usize>(
    L: &mut SMatrix<f64, N, N>,
    d: &mut SVector<f64, N>,
    j: usize,
    g: &SVector<f64, N>,
) ->
    Option<SVector<f64, N>> // p
{
    let N_times_EPS = (N as f64) * f64::EPSILON;

    // Special case for j == 0
    if j == 0 {
        if g[0] <= N_times_EPS {
            let mut p = SVector::<f64, N>::zeros();
            p[0] = 1.0;
            return Some(p);
        }
        d[0] = g[0];
        return None;
    }

    let LII = L.view((0, 0), (j, j));
    let LII_T = LII.transpose();

    let gI = g.rows(0, j);

    let u = LII.lu().solve(&gI).unwrap();
    let v = u.component_div(&d.rows(0, j));

    let delta = g[j] - u.dot(&v);

    if delta <= N_times_EPS {
        let mut p = SVector::<f64, N>::zeros();
        let p_head = LII_T.lu().solve(&v).unwrap();

        p.rows_mut(0, j).copy_from(&p_head);
        p[j] = -1.0;
        return Some(p);
    }


    if j + 1 < N {
        let k_size = N - (j + 1);
        let LKI = L.view((j + 1, 0), (k_size, j));
        let LKI_T = LKI.transpose();
        let gK = g.rows(j + 1, k_size);

        let mut w = (&gK - &(LKI * &u)).scale(1.0 / delta);
        let mut LKK = L.view_mut((j + 1, j + 1), (k_size, k_size));
        let mut dK = d.rows_mut(j + 1, k_size);

        let q = ldlrk1(&mut LKK, &mut dK, -delta, &mut w);

        return match q.is_empty() {
            false => {
                let pi = w.dot(&q);
                let piv = v.scale(pi);
                let lki_q = LKI_T * &q;
                let piv_lki_q = piv - lki_q;

                let p_head = LII_T.lu().solve(&piv_lki_q).unwrap();

                let mut p = SVector::<f64, N>::zeros();
                p.rows_mut(0, j).copy_from(&p_head);
                p[j] = -pi;
                p.rows_mut(j + 1, k_size).copy_from(&q);

                Some(p)
            }
            true => {
                L.view_mut((j, 0), (1, j)).copy_from(&v.transpose());
                L[(j, j)] = 1.0;
                L.view_mut((j + 1, j), (k_size, 1)).copy_from(&w);
                d[j] = delta;
                None
            }
        };
    } else {
        // Last row case - direct update without K partition
        L.view_mut((j, 0), (1, j)).copy_from(&v.transpose());
        L[(j, j)] = 1.0;
        d[j] = delta;

        return None;
    }
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