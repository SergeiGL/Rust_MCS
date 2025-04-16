use crate::mcs_utils::{polint::polint, quadratic_func::quadmin, quadratic_func::quadpol, subint::subint};
use nalgebra::{Matrix3xX, SVector};


// l is always full of 1;
// L is always full of 2
#[inline]
pub(crate) fn exgain<const N: usize>(
    n0: &[usize; N],
    x: &SVector<f64, N>,
    y: &[f64; N],
    x1: &[f64; N],
    x2: &[f64; N],
    fx: f64,
    f0: &Matrix3xX<f64>,
    f1: &[f64; N],
    f2: &[f64; N],
) -> (
    f64,       // emin = e.min()
    isize,     // isplit
    f64        // splval
) {
    let mut emin = f64::INFINITY;
    let mut isplit = 1; // cannot be 0;
    let mut splval = f64::INFINITY;

    for i in 0..N { // i: -1 from Matlab
        if n0[i] == 0 {
            let new_e = f0.column(i).iter().fold(f64::INFINITY, |acc, &new_val| acc.min(new_val)) - f0[(1, i)];

            if new_e < emin {
                emin = new_e;
                isplit = i + 1; // i: -1 from Matlab; isplit should be as in Matlab
                splval = f64::INFINITY;
            }
        } else {
            let z1 = [x[i], x1[i], x2[i]];
            let z2 = [0.0, f1[i] - fx, f2[i] - fx];

            let d = polint(&z1, &z2);
            let (eta1, eta2) = subint(x[i], y[i]);
            let z = quadmin(
                eta1.min(eta2),
                eta1.max(eta2),
                &d,
                &z1,
            );
            let new_e = quadpol(z, &d, &z1);

            if new_e < emin {
                emin = new_e;
                isplit = i + 1; // i: -1 from Matlab; isplit should be as in Matlab
                splval = z;
            }
        }
    }
    (emin, isplit as isize, splval)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let n0 = [1, 0, 0, 0, 0, 0];
        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let y = [0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9, -0.38, -0.05,
            -0.5, -0.62, -0.8, -0.8, -0.9, -0.9,
            -0.08, -0.09, -0.32, -0.025, -0.65, -0.37,
        ]);
        let f1 = [-0.62, -0.8, -0.5, -0.9, -0.38, -0.05];
        let f2 = [-0.08, -0.09, -0.32, -0.025, -0.65, -0.37];

        let (e_min, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e_min, -0.18000000000000005);
        assert_eq!(isplit, 2);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_1() {
        let n0 = [1, 0, 0, 0];
        let x = SVector::<f64, 4>::from_row_slice(&[0.5, 0.5, 0.5, 0.5]);
        let y = [0.3, 1.0, 1.0, 1.0];
        let x1 = [0.0, 0.0, 0.0, 0.0];
        let x2 = [1.0, 1.0, 1.0, 1.0];
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9,
            -0.5, -0.62, -0.8, -0.8,
            -0.08, -0.09, -0.32, -0.025,
        ]);
        let f1 = [-0.62, -0.8, -0.5, -0.9];
        let f2 = [-0.08, -0.09, -0.32, -0.025];

        let (e_min, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e_min, -0.18000000000000005);
        assert_eq!(isplit, 2);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_2() {
        let n0 = [1, 0, 1, 0, 1, 0];
        let x = SVector::<f64, 6>::from_row_slice(&[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]);
        let y = [0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9, -0.38, -0.05,
            -0.5, -0.62, -0.8, -0.8, -0.9, -0.9,
            -0.08, -0.09, -0.32, -0.025, -0.65, -0.37,
        ]);
        let f1 = [0.62, 0.8, 0.5, 0.9, 0.38, 0.05];
        let f2 = [0.08, 0.09, 0.32, 0.025, 0.65, 0.37];

        let (e_min, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e_min, -0.18000000000000005);
        assert_eq!(isplit, 2);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_3() {
        let n0 = [1; 6];
        let x = SVector::<f64, 6>::from_row_slice(&[-0.5; 6]);
        let y = [0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x1 = [0.0; 6];
        let x2 = [1.0; 6];
        let fx = -0.505;
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.8, -0.5, -0.9, -0.38, -0.05,
            -0.5, -0.62, -0.8, -0.8, -0.9, -0.9,
            -0.08, -0.09, -0.32, -0.025, -0.65, -0.37,
        ]);
        let f1 = [0.62, 0.8, 0.5, 0.9, 0.38, 0.05];
        let f2 = [0.08, 0.09, 0.32, 0.025, 0.65, 0.37];

        let (e_min, isplit, splval) = exgain(&n0, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);

        assert_eq!(e_min, 0.19415000000000004);
        assert_eq!(isplit, 6);
        assert_eq!(splval, -0.35);
    }
}