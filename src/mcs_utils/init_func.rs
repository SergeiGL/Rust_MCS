use crate::mcs_utils::{polint::polint, quadratic_func::quadmin, quadratic_func::quadpol, sign::sign};
use itertools::Itertools;
use nalgebra::{Matrix2xX, Matrix3xX, SMatrix, SVector};


pub fn subint(mut x1: f64, mut x2: f64) ->
(
    f64, // x1 new
    f64  // x2 new
) {
    let f: f64 = 1000_f64;
    if f * x1.abs() < 1.0 && x2.abs() > f {
        x2 = sign(x2)
    } else if x2.abs() > f {
        x2 = 10.0 * sign(x2) * x1.abs();
    }
    x1 += (x2 - x1) / 10.0;
    (x1, x2)
}


// l is always full of 1;
// L is always full of 2
pub fn init<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x0: &SMatrix<f64, N, 3>,
) -> (
    Matrix3xX<f64>, // f0 = [[f64; N];3]
    [usize; N],    //  istar
    usize          //  ncall
) {
    let mut ncall = 0_usize;

    let mut x: SVector<f64, N> = x0.column(1).into_owned();

    let mut f1 = func(&x);
    ncall += 1;

    let mut f0 = Matrix3xX::<f64>::repeat(N, 0.0); // L[0] is always = 3
    f0[(1, 0)] = f1;

    let mut istar = [1_usize; N];

    for i in 0..N {
        for j in 0..3 {
            if j == 1 {
                if i != 0 { f0[(j, i)] = f0[(istar[i - 1], i - 1)]; }
            } else {
                x[i] = x0[(i, j)];
                f0[(j, i)] = func(&x);
                ncall += 1;

                if f0[(j, i)] < f1 {
                    f1 = f0[(j, i)];
                    istar[i] = j;
                }
            }
        }

        // Update x[i] to the best found value
        x[i] = x0[(i, istar[i])];
    }

    (f0, istar, ncall)
}


// l is always full of 1;
// L is always full of 2
pub fn initbox<const N: usize>(
    x0: &SMatrix<f64, N, 3>,
    f0: &Matrix3xX<f64>,
    istar: &[usize; N],
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    isplit: &mut Vec<isize>,
    level: &mut Vec<usize>,
    ipar: &mut Vec<Option<usize>>,
    ichild: &mut Vec<isize>,
    f: &mut Matrix2xX<f64>,
    nboxes: &mut usize,
) -> (
    SVector<usize, N>, // p
    SVector<f64, N>,   // xbest
    f64                // fbest
) {
    ipar[0] = None; // parent of root box is -1
    level[0] = 1;  // root box level is 1
    ichild[0] = 1; // root has one child initially
    f[(0, 0)] = f0[(1, 0)];

    let mut par = 0_usize; // parent index is 0
    let mut var = SVector::<f64, N>::repeat(0.0); // variability for dimensions initialized to 0

    // Iterate over each dimension
    for i in 0..N {
        // Set negative index for split information
        isplit[par] = -(i as isize) - 1;
        let mut nchild = 0;

        // If x0's left endpoint is greater than the lower bound `u[i]`, create an extra box
        if x0[(i, 0)] > u[i] {
            *nboxes += 1; // generate a new box
            nchild += 1;
            ipar[*nboxes] = Some(par); // parent index
            level[*nboxes] = level[par] + 1; // Increment level for the child
            ichild[*nboxes] = -nchild; // update child information with negative value
            f[(0, *nboxes)] = f0[(0, i)]; // set function value
        }

        let v1 = v[i];

        // Perform polynomial interpolation and quadratic minimization
        let x0_arr: [f64; 3] = [
            x0[(i, 0)],
            x0[(i, 1)],
            x0[(i, 2)],
        ];

        let mut d = polint(&x0_arr, (&f0.column(i)).as_ref());
        let mut xl = quadmin(u[i], v1, &d, &x0_arr); // left bound minimization
        let mut fl = quadpol(xl, &d, &x0_arr); // compute function value

        let mut xu = quadmin(u[i], v1, &[-d[0], -d[1], -d[2]], &x0_arr); // right bound minimization
        let mut fu = quadpol(xu, &d, &x0_arr);  // compute function value

        // Track which box the coordinate belongs to
        let mut par1 = 0;

        if istar[i] == 0 {
            if xl < x0[(i, 0)] {
                par1 = *nboxes;
            } else {
                par1 = *nboxes + 1;
            }
        }

        // Iterate over L[i], generate new boxes for splitting
        for j in 0..2 {
            *nboxes += 1;
            nchild += 1;

            // Decide the level increment, based on function value comparisons
            let s = if f0[(j, i)] <= f0[(j + 1, i)] { 1 } else { 2 };
            ipar[*nboxes] = Some(par);
            level[*nboxes] = level[par] + s;
            ichild[*nboxes] = -nchild;
            f[(0, *nboxes)] = f0[(j, i)];  // assign function value

            if j >= 1 {
                if istar[i] == j {
                    if xl <= x0[(i, j)] {
                        par1 = *nboxes - 1; // nboxes is at least 1
                    } else {
                        par1 = *nboxes;
                    }
                }
                if j == 0 { // j can only be 0, not less (WTF)
                    let x0_arr: [f64; 3] = [
                        x0[(i, j)],     // First column
                        x0[(i, j + 1)], // Second column
                        x0[(i, j + 2)], // Third column
                    ];
                    d = polint(&x0_arr, (&f0.column(i)).as_ref());
                    let u1 = v[i]; // j is always < 0

                    xl = quadmin(x0[(i, j)], u1, &d, &x0_arr);
                    fl = fl.min(quadpol(xl, &d, &x0_arr));

                    xu = quadmin(x0[(i, j)], u1, &[-d[0], -d[1], -d[2]], &x0_arr);
                    fu = fu.max(quadpol(xu, &d, &x0_arr));
                }
            }

            *nboxes += 1;
            nchild += 1;
            ipar[*nboxes] = Some(par);
            level[*nboxes] = level[par] + 3 - s;
            ichild[*nboxes] = -nchild;
            f[(0, *nboxes)] = f0[(j + 1, i)]; // update function value for the next box
        }

        // If the upper end of x0 is below v, generate a final box
        if x0[(i, 2)] < v[i] {
            *nboxes += 1;
            nchild += 1;
            ipar[*nboxes] = Some(par);
            level[*nboxes] = level[par] + 1;
            ichild[*nboxes] = -nchild;
            f[(0, *nboxes)] = f0[(2, i)];
        }

        if istar[i] == 2 {
            if x0[(i, 2)] < v[i] {
                if xl <= x0[(i, 2)] {
                    par1 = *nboxes - 1; // nboxes is at least 1
                } else {
                    par1 = *nboxes;
                }
            } else {
                par1 = *nboxes;
            }
        }

        // Variability across the ith component
        var[i] = fu - fl;

        // Mark the parent box as split
        level[par] = 0;
        par = par1;
    }

    // Finding the best function value
    let fbest = f0[(istar[N - 1], N - 1)];

    let mut p = SVector::<usize, N>::zeros(); // stores the indices of best points
    let mut xbest = SVector::<f64, N>::zeros(); // best point values

    for i in 0..N {
        p[i] = var.iter().position_max_by(|a, b| a.total_cmp(b)).unwrap(); // find the maximum in var
        var[p[i]] = -1.0; // mark as used
        xbest[i] = x0[(i, istar[i])];  // store the best value of x at that index
    }

    (p, xbest, fbest)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;
    use nalgebra::Matrix2xX;

    #[test]
    fn initbox_test_0() {
        const N: usize = 30;

        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62323147, -0.86038255, -0.5152638, -0.98834122, -0.37914019, -0.05313547,
            -0.50531499, -0.62323147, -0.86038255, -0.86038255, -0.98834122, -0.98834122,
            -0.08793206, -0.09355143, -0.32139218, -0.0251343, -0.65273189, -0.37750674,
        ]);
        let istar = [0, 0, 1, 0, 1, 1];
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut isplit = vec![0_isize; N];
        let mut level = vec![0_usize; N];
        let mut ipar = vec![Some(0_usize); N];
        let mut ichild = vec![0_isize; N];
        let mut f = Matrix2xX::<f64>::zeros(N);
        let mut nboxes = 0_usize;

        let (p, xbest, fbest) = initbox(&x0, &f0, &istar, &u, &v, &mut isplit, &mut level, &mut ipar, &mut ichild, &mut f, &mut nboxes);

        let expected_ipar = [
            None, Some(0), Some(0), Some(0), Some(0), Some(1), Some(1), Some(1), Some(1), Some(5), Some(5), Some(5), Some(5),
            Some(10), Some(10), Some(10), Some(10), Some(13), Some(13), Some(13), Some(13), Some(19), Some(19), Some(19), Some(19), Some(0),
            Some(0), Some(0), Some(0), Some(0)];

        let expected_level = [
            0, 0, 3, 2, 3, 0, 4, 3, 4, 5, 0, 4, 5, 0, 6, 5, 6,
            7, 6, 0, 7, 8, 7, 7, 8, 0, 0, 0, 0, 0];
        let expected_ichild = [
            1, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
            -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, 0,
            0, 0, 0, 0,
        ];
        let expected_f = Matrix2xX::<f64>::from_row_slice(&[
            -0.50531499, -0.62323147, -0.50531499, -0.50531499, -0.08793206, -0.86038255, -0.62323147, -0.62323147, -0.09355143, -0.5152638, -0.86038255, -0.86038255, -0.32139218, -0.98834122, -0.86038255, -0.86038255, -0.0251343, -0.37914019, -0.98834122, -0.98834122, -0.65273189, -0.05313547, -0.98834122, -0.98834122, -0.37750674, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);
        let expected_isplit = [
            -1, -2, 0, 0, 0, -3, 0, 0, 0, 0, -4, 0, 0,
            -5, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];
        let expected_p = [3, 5, 1, 4, 2, 0];
        let expected_xbest = [0.0, 0.0, 0.5, 0.0, 0.5, 0.5];
        let expected_fbest = -0.98834122;
        let expected_nboxes = 24;


        assert_eq!(ipar, expected_ipar);
        assert_eq!(level, expected_level);
        assert_eq!(ichild, expected_ichild);
        assert_eq!(isplit, expected_isplit);
        assert_eq!(p.as_slice(), expected_p);
        assert_eq!(xbest.as_slice(), expected_xbest);
        assert_eq!(fbest, expected_fbest);
        assert_eq!(f, expected_f);
        assert_eq!(nboxes, expected_nboxes);
    }

    #[test]
    fn initbox_test_1() {
        const N: usize = 30;

        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[
            0., 1., 2.,
            10., 11., 12.,
            20., 21., 22.,
            30., 31., 32.,
            40., 41., 42.,
            50., 51., 52.,
        ]);

        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.1, -1.1, -11.1, 0.1, 1.1, 11.1,
            -0.2, -2.2, -21.2, 0.2, 2.2, 21.2,
            -0.3, -3.3, -31.3, 0.3, 3.3, 31.3
        ]);

        let istar = [0, 1, 2, 0, 1, 2];
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut isplit = vec![-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let mut level = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
        let mut ipar = vec![Some(0_usize); N];
        ipar[0] = None;
        let mut ichild = vec![-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let mut f = Matrix2xX::<f64>::from_row_slice(&[
            0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
            -1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4
        ]);
        let mut nboxes = 0_usize;

        let (p, xbest, fbest) =
            initbox(
                &x0, &f0, &istar, &u, &v,
                &mut isplit, &mut level, &mut ipar, &mut ichild,
                &mut f,
                &mut nboxes,
            );

        let expected_ipar = [
            None, Some(0), Some(0), Some(0), Some(0), Some(1), Some(1), Some(1), Some(1), Some(1), Some(7), Some(7), Some(7), Some(7), Some(7), Some(14), Some(14), Some(14), Some(14), Some(14), Some(15), Some(15), Some(15), Some(15), Some(15), Some(22), Some(22), Some(22), Some(22), Some(22)
        ];
        let expected_level = [
            0, 0, 2, 3, 2, 4, 5, 0, 5, 4, 5, 6, 5, 6, 0, 0, 6, 7, 6, 7, 7, 7, 0, 7, 8, 9, 9, 10, 9, 10
        ];
        let expected_ichild = [
            1, -1, -2, -3, -4, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5
        ];
        let expected_f = Matrix2xX::<f64>::from_row_slice(&[
            -0.2, -0.1, -0.2, -0.2, -0.3, -1.1, -1.1, -2.2, -2.2, -3.3, -11.1, -11.1, -21.2, -21.2, -31.3, 0.1, 0.1, 0.2, 0.2, 0.3, 1.1, 1.1, 2.2, 2.2, 3.3, 11.1, 11.1, 21.2, 21.2, 31.3,
            -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4
        ]);
        let expected_isplit = [
            -1, -2, -13, -12, -11, -10, -9, -3, -7, -6, -5, -4, -3, -2, -4, -5, 1, 2, 3, 4, 5, 6, -6, 8, 9, 10, 11, 12, 13, 14
        ];
        let expected_p = [2, 5, 4, 1, 3, 0];
        let expected_xbest = [0.0, 11.0, 22.0, 30.0, 41.0, 52.0];
        let expected_fbest = 31.3;
        let expected_nboxes = 29;


        assert_eq!(ipar, expected_ipar);
        assert_eq!(level, expected_level);
        assert_eq!(ichild, expected_ichild);
        assert_eq!(isplit, expected_isplit);
        assert_eq!(p.as_slice(), expected_p);
        assert_eq!(xbest.as_slice(), expected_xbest);
        assert_eq!(fbest, expected_fbest);
        assert_eq!(f, expected_f);
        assert_eq!(nboxes, expected_nboxes);
    }


    #[test]
    fn subint_test_0() {
        let x1 = 1.0;
        let x2 = 2.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 1.1);
        assert_eq!(x2_new, 2.0);
    }

    #[test]
    fn subint_test_1() {
        let x1 = 0.0000001;
        let x2 = 2000.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 0.10000009000000001);
        assert_eq!(x2_new, 1.0);
    }


    #[test]
    fn subint_test_2() {
        let x1 = -1.0;
        let x2 = 50.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 4.1);
        assert_eq!(x2_new, 50.0);
    }

    #[test]
    fn subint_test_3() {
        let x1 = -0.00000001;
        let x2 = 132455.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 0.099999991);
        assert_eq!(x2_new, 1.0);
    }

    //---------------------------------------------------
    #[test]
    fn init_test_base_case() {
        let x0 = SMatrix::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ]);

        let (f0, istar, ncall) = init(hm6, &x0);

        let expected_f0 = Matrix3xX::<f64>::from_row_slice(&[-0.6232314675898112, -0.8603825505022568, -0.5152637963551447, -0.9883412202327723, -0.3791401895175917, -0.05313547352279423, -0.5053149917022333, -0.6232314675898112, -0.8603825505022568, -0.8603825505022568, -0.9883412202327723, -0.9883412202327723, -0.08793206431638863, -0.09355142624190396, -0.3213921800858628, -0.025134295008083094, -0.6527318901582629, -0.37750674173452126]);
        let expected_istar = [0, 0, 1, 0, 1, 1];

        assert_eq!(f0, expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_same_x0_values() {
        let x0 = SMatrix::from_row_slice(&[
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);

        let (f0, istar, ncall) = init(hm6, &x0);

        let expected_f0 = Matrix3xX::<f64>::from_row_slice(&[-3.408539273427753e-05; 18]);
        let expected_istar = [1, 1, 1, 1, 1, 1];

        assert_eq!(f0, expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }
}

