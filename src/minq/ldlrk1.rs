use nalgebra::{DMatrix, DVector};

pub fn ldlrk1(
    L: &mut DMatrix<f64>,
    d: &mut DVector<f64>,
    mut alp: f64,
    u: &mut DVector<f64>,
) ->
    DVector<f64> // p
{
    let mut p: DVector<f64> = DVector::zeros(0);

    if alp == 0.0 {
        return p;
    }

    let eps: f64 = 2.2204e-16;
    let n: usize = u.len();
    let neps: f64 = n as f64 * eps;

    for k in 0..n {
        if u[k] == 0.0 { continue; };
        let delta: f64 = d[k] + alp * u[k].powi(2);

        if alp < 0.0 && delta <= neps {
            let p0K = DVector::from_iterator(k + 1, std::iter::once(1.).chain(std::iter::repeat(0.).take(k)));
            let L0K = DMatrix::from_fn(k + 1, k + 1, |row, col| L[(row, col)]);
            let p0K = L0K.lu().solve(&p0K).unwrap();

            p = DVector::from_fn(n, |i, _| {
                if i < k + 1 {
                    p0K[i]
                } else {
                    0.
                }
            });

            return p;
        }

        let q = d[k] / delta;
        d[k] = delta;

        let LindK: Vec<f64> = ((k + 1)..n).map(|i| L[(i, k)]).collect();
        let c: Vec<f64> = LindK.iter().map(|&lk| lk * u[k]).collect();

        for (i, index) in ((k + 1)..n).enumerate() {
            L[(index, k)] = LindK[i] * q + (alp * u[k] / delta) * u[index];
            u[index] -= c[i];
        }

        alp *= q;
        if alp == 0.0 {
            break;
        }
    }
    p
}


#[cfg(test)]
mod tests {
    use super::*;

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(3, 3, &[1.0, -2.0, 3.0, -0.8130081300813008, -5.0, 6.0, 0.8130081300813008, 8.0, 9.0]));

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);


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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

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

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

        assert_eq!(L, DMatrix::<f64>::from_row_slice(2, 2, &[0.013230535353535766, -0.4234767083209479, 0.6642132426251283, 1.5000003434343434]));
        assert_eq!(d, DVector::from_row_slice(&[3.444646464646567, 3.96771776306761]));
        assert_eq!(p, DVector::from_row_slice(&[]));
    }
}