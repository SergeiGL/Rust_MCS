use nalgebra::{Const, DVector, Dyn, MatrixViewMut, U1};

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

    for k in 0..n {
        if u[k] == 0.0 { continue; };
        let delta: f64 = d[k] + alp * u[k].powi(2);

        if alp < 0.0 && delta <= neps {
            let mut p0K = DVector::zeros(k + 1);
            p0K[0] = 1.0;
            let p0K = L.view_range(..k + 1, ..k + 1).lu().solve(&p0K).unwrap();

            return DVector::from_fn(n, |i, _| {
                if i < k + 1 {
                    p0K[i]
                } else {
                    0.
                }
            });
        }

        let inv_delta = 1.0 / delta;
        let q = d[k] * inv_delta;
        d[k] = delta;

        let uk = u[k];
        let alp_uk_div_delta = alp * uk * inv_delta;

        for index in (k + 1)..n {
            let ui = u[index];
            u[index] = ui - L[(index, k)] * uk;
            L[(index, k)] = L[(index, k)] * q + alp_uk_div_delta * ui;
        }

        alp *= q;
        if alp == 0.0 {
            break;
        }
    }
    DVector::zeros(0)
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

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

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, -2.0, 3.0, -0.8130081300813009, -5.0, 6.0, 0.8130081300813009, 8.0, 9.0]));

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
}