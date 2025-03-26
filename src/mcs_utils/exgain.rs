use crate::mcs_utils::{init_func::subint, polint::polint, quadratic_func::quadmin, quadratic_func::quadpol};
use nalgebra::{Matrix3xX, SVector};


// l is always full of 1;
// L is always full of 2
pub fn exgain<const N: usize>(
    n0: &SVector<usize, N>,
    x: &SVector<f64, N>,
    y: &SVector<f64, N>,
    x1: &SVector<f64, N>,
    x2: &SVector<f64, N>,
    fx: f64,
    f0: &Matrix3xX<f64>,
    f1: &SVector<f64, N>,
    f2: &SVector<f64, N>,
) -> (
    SVector<f64, N>,  // e
    isize,     // isplit
    f64        // splval
) {
    let mut e = SVector::<f64, N>::zeros();
    let mut emin = f64::INFINITY;
    let mut isplit = 0;
    let mut splval = f64::INFINITY;

    for i in 0..N {
        if n0[i] == 0 {
            e[i] = f0.column(i).iter().fold(f64::INFINITY, |acc, &new_val| acc.min(new_val)) - f0[(1, i)];

            if e[i] < emin {
                emin = e[i];
                isplit = i;
                splval = f64::INFINITY;
            }
        } else {
            let z1 = [x[i], x1[i], x2[i]];
            let z2 = [0.0, f1[i] - fx, f2[i] - fx];

            let d = polint(&z1, &z2);
            let (eta1, eta2) = subint::<1000>(x[i], y[i]);
            let z = quadmin(
                eta1.min(eta2),
                eta1.max(eta2),
                &d,
                &z1,
            );
            e[i] = quadpol(z, &d, &z1);

            if e[i] < emin {
                emin = e[i];
                isplit = i;
                splval = z;
            }
        }
    }
    (e, isplit as isize, splval)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let n0 = SVector::<usize, 6>::from_row_slice(&[1, 0, 0, 0, 0, 0]);
        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let y = SVector::<f64, 6>::from_row_slice(&[0.3, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let x1 = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let x2 = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9, -0.38, -0.05,
            -0.5, -0.62, -0.8, -0.8, -0.9, -0.9,
            -0.08, -0.09, -0.32, -0.025, -0.65, -0.37,
        ]);
        let f1 = SVector::<f64, 6>::from_row_slice(&[-0.62, -0.8, -0.5, -0.9, -0.38, -0.05]);
        let f2 = SVector::<f64, 6>::from_row_slice(&[-0.08, -0.09, -0.32, -0.025, -0.65, -0.37]);

        let (e, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e.as_slice(), [-0.0832, -0.18000000000000005, 0.0, -0.09999999999999998, 0.0, 0.0]);
        assert_eq!(isplit, 1);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_1() {
        let n0 = SVector::<usize, 4>::from_row_slice(&[1, 0, 0, 0]);
        let x = SVector::<f64, 4>::from_row_slice(&[0.5, 0.5, 0.5, 0.5]);
        let y = SVector::<f64, 4>::from_row_slice(&[0.3, 1.0, 1.0, 1.0]);
        let x1 = SVector::<f64, 4>::from_row_slice(&[0.0, 0.0, 0.0, 0.0]);
        let x2 = SVector::<f64, 4>::from_row_slice(&[1.0, 1.0, 1.0, 1.0]);
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9,
            -0.5, -0.62, -0.8, -0.8,
            -0.08, -0.09, -0.32, -0.025,
        ]);
        let f1 = SVector::<f64, 4>::from_row_slice(&[-0.62, -0.8, -0.5, -0.9]);
        let f2 = SVector::<f64, 4>::from_row_slice(&[-0.08, -0.09, -0.32, -0.025]);

        let (e, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e.as_slice(), [-0.0832, -0.18000000000000005, 0.0, -0.09999999999999998]);
        assert_eq!(isplit, 1);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_2() {
        let n0 = SVector::<usize, 6>::from_row_slice(&[1, 0, 1, 0, 1, 0]);
        let x = SVector::<f64, 6>::from_row_slice(&[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]);
        let y = SVector::<f64, 6>::from_row_slice(&[0.3, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let x1 = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let x2 = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9, -0.38, -0.05,
            -0.5, -0.62, -0.8, -0.8, -0.9, -0.9,
            -0.08, -0.09, -0.32, -0.025, -0.65, -0.37,
        ]);
        let f1 = SVector::<f64, 6>::from_row_slice(&[0.62, 0.8, 0.5, 0.9, 0.38, 0.05]);
        let f2 = SVector::<f64, 6>::from_row_slice(&[0.08, 0.09, 0.32, 0.025, 0.65, 0.37]);

        let (e, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e.as_slice(), [0.24249600000000007, -0.18000000000000005, 0.37815, -0.09999999999999998, 0.31800000000000006, 0.0]);
        assert_eq!(isplit, 1);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_3() {
        let n0 = SVector::<usize, 6>::from_row_slice(&[1; 6]);
        let x = SVector::<f64, 6>::from_row_slice(&[-0.5; 6]);
        let y = SVector::<f64, 6>::from_row_slice(&[0.3, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let x1 = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let x2 = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9, -0.38, -0.05,
            -0.5, -0.62, -0.8, -0.8, -0.9, -0.9,
            -0.08, -0.09, -0.32, -0.025, -0.65, -0.37,
        ]);
        let f1 = SVector::<f64, 6>::from_row_slice(&[0.62, 0.8, 0.5, 0.9, 0.38, 0.05]);
        let f2 = SVector::<f64, 6>::from_row_slice(&[0.08, 0.09, 0.32, 0.025, 0.65, 0.37]);

        let (e, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e.as_slice(), [0.24249600000000007, 0.5077000000000002, 0.37815, 0.5300000000000002, 0.31800000000000006, 0.19415000000000004]);
        assert_eq!(isplit, 5);
        assert_eq!(splval, -0.35);
    }
}