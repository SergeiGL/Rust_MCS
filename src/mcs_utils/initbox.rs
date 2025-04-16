use crate::mcs_utils::polint::polint;
use crate::mcs_utils::quadratic_func::{quadmin, quadpol};
use nalgebra::{Matrix3xX, SMatrix, SVector};

pub(crate) fn initbox<const N: usize>(
    x0: &SMatrix<f64, N, 3>,
    f0: &Matrix3xX<f64>,
    istar: &[usize; N], // -1 from Matlab
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    isplit: &mut Vec<isize>,
    level: &mut Vec<usize>,
    ipar: &mut Vec<Option<usize>>,
    ichild: &mut Vec<isize>,
    f: &mut [Vec<f64>; 2],
    nboxes: &mut usize, // as in Matlab;
) -> (
    SVector<usize, N>, // p
    SVector<f64, N>,   // xbest
    f64                // fbest
) {
    // nglob, xglob are used only for prints

    ipar[0] = Some(0); // as in Matlab
    level[0] = 1;  // as in Matlab; root box level is 1
    ichild[0] = 1; // as in Matlab; root has one child initially
    f[0][0] = f0[(1, 0)];

    let mut par = 1_usize; // parent index is 1
    let mut var = [(0_usize, f64::NAN); N]; // variability for dimensions initialized to 0

    for i in 0..N {
        // Set negative index for split information
        isplit[par - 1] = -(i as isize) - 1; //   isplit(par) = -i;  i = 1:n
        let mut nchild = 0;

        // If x0's left endpoint is greater than the lower bound `u[i]`, create an extra box
        if x0[(i, 0)] > u[i] {
            nchild += 1;
            // Update before incrementing nboxes as index in Rust is precisely 1 lower
            ipar[*nboxes] = Some(par); // parent index
            level[*nboxes] = level[par - 1] + 1; // Increment level for the child
            ichild[*nboxes] = -nchild; // update child information with negative value
            f[0][*nboxes] = f0[(0, i)]; // set function value
            *nboxes += 1; // generate a new box
        }

        let v1 = v[i];

        // Perform polynomial interpolation and quadratic minimization
        let x0_arr: [f64; 3] = [
            x0[(i, 0)],
            x0[(i, 1)],
            x0[(i, 2)],
        ];

        let d = polint(&x0_arr, (&f0.column(i)).as_ref());
        let xl = quadmin(u[i], v1, &d, &x0_arr); // left bound minimization
        let fl = quadpol(xl, &d, &x0_arr); // compute function value
        let xu = quadmin(u[i], v1, &[-d[0], -d[1], -d[2]], &x0_arr); // right bound minimization
        let fu = quadpol(xu, &d, &x0_arr);  // compute function value

        // Track which box the coordinate belongs to
        let mut par1 = 0;
        if istar[i] == 0 {
            par1 = if xl < x0[(i, 0)] {
                *nboxes
            } else {
                *nboxes + 1
            }
        }

        // Iterate over L[i], generate new boxes for splitting
        for j in 0..2 {
            // Decide the level increment, based on function value comparisons
            let s = if f0[(j, i)] <= f0[(j + 1, i)] { 1 } else { 2 };
            nchild += 1;
            // Update before incrementing nboxes as index in Rust is precisely 1 lower
            ipar[*nboxes] = Some(par);
            level[*nboxes] = level[par - 1] + s;
            ichild[*nboxes] = -nchild;
            f[0][*nboxes] = f0[(j, i)];
            *nboxes += 1;

            if j >= 1 {
                if istar[i] == j {
                    par1 = if xl <= x0[(i, j)] {
                        *nboxes - 1 // nboxes is at least 1
                    } else {
                        *nboxes
                    }
                }
                // Matlab: j <= L(i)-2; L(3)===3 --> j <= 1,
                // but upper in Matlab there is j >= 2
                // thus, j <= 1 && j >= 2 never reached
            }

            nchild += 1;
            ipar[*nboxes] = Some(par);
            level[*nboxes] = level[par - 1] + 3 - s; // s is either 1 or 2
            ichild[*nboxes] = -nchild;
            f[0][*nboxes] = f0[(j + 1, i)]; // update function value for the next box
            *nboxes += 1;
        }

        // If the upper end of x0 is below v, generate a final box
        if x0[(i, 2)] < v[i] {
            nchild += 1;
            // Update before incrementing nboxes as index in Rust is precisely 1 lower
            ipar[*nboxes] = Some(par);
            level[*nboxes] = level[par - 1] + 1;
            ichild[*nboxes] = -nchild;
            f[0][*nboxes] = f0[(2, i)];
            *nboxes += 1;
        }

        if istar[i] == 2 { // Matlab: L===3; here istar -1 from Matlab, so ==2
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
        var[i] = (i, fu - fl);

        // Mark the parent box as split
        level[par - 1] = 0;
        par = par1;

        // j1 sets splval, but it is used only for prints
    }

    // Equivalent to
    // for i = 1:n
    //   [var0,p(i)] = max(var);
    //   var(p(i)) = -1;
    //   xbest(i) = x0(i,istar(i));  % best point after the init. procedure
    // end
    var.sort_unstable_by(|(_, var_i), (_, var_j)| var_j.total_cmp(var_i));

    (
        SVector::<usize, N>::from_fn(|row, _| var[row].0),   // p
        SVector::<f64, N>::from_fn(|row, _| x0[(row, istar[row])]), // xbest
        f0[(istar[N - 1], N - 1)]
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3xX, SMatrix, SVector};

    #[test]
    fn initbox_test() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // global nboxes;
        // nboxes = 1;
        // isplit = zeros(1,1);
        // level = zeros(1,1);
        // ipar = zeros(1,1);
        // ichild = zeros(1,1);
        //
        // x0 = [
        // [1110.0, 1.0, 10.0];
        // [12323.0, 0.0, 1.0];
        // [1011.0, 1.0, 20.0];
        // [-14.0, -1.0, -10.0];
        // [111.0, 5.0, 10.0];
        // [201.0, -10.0, 10.0];
        // ];
        // f0 = [
        //     [-0.62, -0.86, 1., -1., -0.3, -0.05];
        //    [ -1.50, -0., 0.8, -0.86, -0.984122, -0.9];
        //     [0.0, 0.09, 0.32, 0.02, 1.652, 0.377];
        // ];
        // l = ones(100)*2; % always 2
        // L = ones(100)*3; % always 3
        // istar = [1, 1, 1, 2, 3, 2];
        // u = [-10.0, -1.0, 10.0, -14.0, 1.0, 20.0];
        // v = [10.0, 1.0, 20.0, -10.0, 10.0, 10.0];
        // prt = 0;
        //
        // format long g;
        // [ipar_out,level_out,ichild_out,f_out,isplit_out,p_out,xbest_out,fbest_out]=initbox(x0,f0,l,L,istar,u,v,prt);
        // sprintf('%.16g,', ipar_out)
        // sprintf('%.16g,', level_out)
        // sprintf('%.16g,', ichild_out)
        // sprintf('%.16g,', isplit_out)
        // sprintf('%.16g,', p_out)
        // sprintf('%.16g,', xbest_out)
        // fbest_out
        // sprintf('%.16g,', f_out)
        // nboxes

        const INIT_VEC_CAPACITY: usize = 30;

        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[
            1110.0, 1.0, 10.0,
            12323.0, 0.0, 1.0,
            1011.0, 1.0, 20.0,
            -14.0, -1.0, -10.0,
            111.0, 5.0, 10.0,
            201.0, -10.0, 10.0,
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.62, -0.86, 1., -1., -0.3, -0.05,
            -1.50, -0., 0.8, -0.86, -0.984122, -0.9,
            0.0, 0.09, 0.32, 0.02, 1.652, 0.377,
        ]);
        let istar = [0, 0, 0, 1, 2, 1];
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0, -1.0, 10.0, -14.0, 1.0, 20.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[10.0, 1.0, 20.0, -10.0, 10.0, 10.0]);

        let mut isplit = vec![0_isize; INIT_VEC_CAPACITY];
        let mut level = vec![0_usize; INIT_VEC_CAPACITY];
        let mut ipar = vec![Some(0); INIT_VEC_CAPACITY];
        let mut ichild = vec![0_isize; INIT_VEC_CAPACITY];
        let mut f = [vec![0_f64; INIT_VEC_CAPACITY], vec![0_f64; INIT_VEC_CAPACITY]];
        let mut nboxes = 1_usize;

        let (p, xbest, fbest) = initbox(&x0, &f0, &istar, &u, &v, &mut isplit, &mut level, &mut ipar, &mut ichild, &mut f, &mut nboxes);

        let expected_ipar = [Some(0), Some(1), Some(1), Some(1), Some(1), Some(1), Some(2), Some(2), Some(2), Some(2), Some(2), Some(7), Some(7), Some(7), Some(7), Some(7), Some(12), Some(12), Some(12), Some(12), Some(18), Some(18), Some(18), Some(18), Some(18), Some(25), Some(25), Some(25), Some(25), Some(25)];
        let expected_level = [0, 0, 3, 2, 2, 3, 0, 3, 4, 3, 4, 0, 5, 4, 5, 4, 5, 0, 5, 6, 7, 8, 7, 7, 0, 9, 10, 9, 9, 10];
        let expected_ichild = [1, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, ];
        let expected_isplit = [-1, -2, 0, 0, 0, 0, -3, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0];
        // -1 from Matlab
        let expected_p = [4, 0, 3, 5, 2, 1];
        let expected_xbest = [1110., 12323., 1011., -1., 10., -10.];
        let expected_fbest = -0.9;
        let expected_f = [-1.5, -0.62, -0.62, -1.5, -1.5, 0., -0.86, -0.86, -0., -0., 0.09, 1., 1., 0.8, 0.8, 0.32, -1., -0.86, -0.86, 0.02, -0.3, -0.3, -0.9841220000000001, -0.9841220000000001, 1.652, -0.05, -0.05, -0.9, -0.9, 0.377];
        let expected_nboxes = 30;

        assert_eq!(ipar, expected_ipar);
        assert_eq!(level, expected_level);
        assert_eq!(ichild, expected_ichild);
        assert_eq!(isplit, expected_isplit);
        assert_eq!(p.as_slice(), expected_p);
        assert_eq!(xbest.as_slice(), expected_xbest);
        assert_eq!(fbest, expected_fbest);
        assert_eq!(f[0].as_slice(), expected_f);
        assert_eq!(nboxes, expected_nboxes);
    }

    #[test]
    fn initbox_test_0() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // global nboxes;
        // nboxes = 1;
        // isplit = zeros(1,1);
        // level = zeros(1,1);
        // ipar = zeros(1,1);
        // ichild = zeros(1,1);
        //
        // x0 = [
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // ];
        // f0 = [
        //     [-0.62323147, -0.86038255, -0.5152638, -0.98834122, -0.37914019, -0.05313547];
        //     [-0.50531499, -0.62323147, -0.86038255, -0.86038255, -0.98834122, -0.98834122];
        //     [-0.08793206, -0.09355143, -0.32139218, -0.0251343, -0.65273189, -0.37750674];
        // ];
        // l = ones(100)*2; % always 2
        // L = ones(100)*3; % always 3
        // istar = [1, 1, 2, 1, 2, 2]; % +1 from Rust
        // u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // prt = 0;
        //
        // format long g;
        // [ipar_out,level_out,ichild_out,f_out,isplit_out,p_out,xbest_out,fbest_out]=initbox(x0,f0,l,L,istar,u,v,prt);
        // sprintf('%.16g,', ipar_out)
        // sprintf('%.16g,', level_out)
        // sprintf('%.16g,', ichild_out)
        // sprintf('%.16g,', isplit_out)
        // sprintf('%.16g,', p_out)
        // sprintf('%.16g,', xbest_out)
        // fbest_out
        // sprintf('%.16g,', f_out)
        // nboxes

        const INIT_VEC_CAPACITY: usize = 25;

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
        let mut isplit = vec![0_isize; INIT_VEC_CAPACITY];
        let mut level = vec![0_usize; INIT_VEC_CAPACITY];
        let mut ipar = vec![Some(0_usize); INIT_VEC_CAPACITY];
        let mut ichild = vec![0_isize; INIT_VEC_CAPACITY];
        let mut f = [vec![0_f64; INIT_VEC_CAPACITY], vec![0_f64; INIT_VEC_CAPACITY]];
        let mut nboxes = 1_usize;

        let (p, xbest, fbest) = initbox(&x0, &f0, &istar, &u, &v, &mut isplit, &mut level, &mut ipar, &mut ichild, &mut f, &mut nboxes);

        let expected_ipar = [Some(0), Some(1), Some(1), Some(1), Some(1), Some(2), Some(2), Some(2), Some(2), Some(6), Some(6), Some(6), Some(6), Some(11), Some(11), Some(11), Some(11), Some(14), Some(14), Some(14), Some(14), Some(20), Some(20), Some(20), Some(20)];
        let expected_level = [0, 0, 3, 2, 3, 0, 4, 3, 4, 5, 0, 4, 5, 0, 6, 5, 6, 7, 6, 0, 7, 8, 7, 7, 8];
        let expected_ichild = [1, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4];
        let expected_f = [-0.50531499, -0.62323147, -0.50531499, -0.50531499, -0.08793206, -0.86038255, -0.62323147, -0.62323147, -0.09355143, -0.5152638, -0.86038255, -0.86038255, -0.32139218, -0.98834122, -0.86038255, -0.86038255, -0.0251343, -0.37914019, -0.98834122, -0.98834122, -0.65273189, -0.05313547, -0.98834122, -0.98834122, -0.37750674];
        let expected_isplit = [-1, -2, 0, 0, 0, -3, 0, 0, 0, 0, -4, 0, 0, -5, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0];
        let expected_p = [3, 5, 1, 4, 2, 0];
        let expected_xbest = [0.0, 0.0, 0.5, 0.0, 0.5, 0.5];
        let expected_fbest = -0.98834122;
        let expected_nboxes = 25;

        assert_eq!(ipar, expected_ipar);
        assert_eq!(level, expected_level);
        assert_eq!(ichild, expected_ichild);
        assert_eq!(isplit, expected_isplit);
        assert_eq!(p.as_slice(), expected_p);
        assert_eq!(xbest.as_slice(), expected_xbest);
        assert_eq!(fbest, expected_fbest);
        assert_eq!(f[0].as_slice(), expected_f);
        assert_eq!(nboxes, expected_nboxes);
    }

    #[test]
    fn initbox_test_1() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // global nboxes;
        // nboxes = 1;
        // isplit = [-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        // level = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
        // ipar = zeros(1,1);
        // ichild = [-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        //
        // x0 = [
        //     [0., 1., 2.];
        //     [10., 11., 12.];
        //     [20., 21., 22.];
        //     [30., 31., 32.];
        //     [40., 41., 42.];
        //     [50., 51., 52.];
        // ];
        // f0 = [
        //     [-0.1, -1.1, -11.1, 0.1, 1.1, 11.1];
        //     [-0.2, -2.2, -21.2, 0.2, 2.2, 21.2];
        //     [-0.3, -3.3, -31.3, 0.3, 3.3, 31.3]
        // ];
        // l = ones(100)*2; % always 2
        // L = ones(100)*3; % always 3
        // istar = [1, 2, 3, 1, 2, 3]; % +1 from Rust
        // u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // prt = 0;
        //
        // format long g;
        // [ipar_out,level_out,ichild_out,f_out,isplit_out,p_out,xbest_out,fbest_out]=initbox(x0,f0,l,L,istar,u,v,prt);
        // sprintf('%.16g,', ipar_out)
        // sprintf('%.16g,', level_out)
        // sprintf('%.16g,', ichild_out)
        // sprintf('%.16g,', isplit_out)
        // sprintf('%.16g,', p_out)
        // sprintf('%.16g,', xbest_out)
        // fbest_out
        // sprintf('%.16g,', f_out)
        // nboxes

        const INIT_VEC_CAPACITY: usize = 30;

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
        let mut ipar = vec![Some(0_usize); INIT_VEC_CAPACITY];
        let mut ichild = vec![-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let mut f = [
            vec![0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, -1.5],
            vec![-1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4]
        ];
        let mut nboxes = 1_usize;

        let (p, xbest, fbest) = initbox(&x0, &f0, &istar, &u, &v, &mut isplit, &mut level, &mut ipar, &mut ichild, &mut f, &mut nboxes);

        let expected_ipar = [Some(0), Some(1), Some(1), Some(1), Some(1), Some(2), Some(2), Some(2), Some(2), Some(2), Some(8), Some(8), Some(8), Some(8), Some(8), Some(15), Some(15), Some(15), Some(15), Some(15), Some(16), Some(16), Some(16), Some(16), Some(16), Some(23), Some(23), Some(23), Some(23), Some(23)];
        let expected_level = [0, 0, 2, 3, 2, 4, 5, 0, 5, 4, 5, 6, 5, 6, 0, 0, 6, 7, 6, 7, 7, 7, 0, 7, 8, 9, 9, 10, 9, 10];
        let expected_ichild = [1, -1, -2, -3, -4, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5];
        let expected_f = [-0.2, -0.1, -0.2, -0.2, -0.3, -1.1, -1.1, -2.2, -2.2, -3.3, -11.1, -11.1, -21.2, -21.2, -31.3, 0.1, 0.1, 0.2, 0.2, 0.3, 1.1, 1.1, 2.2, 2.2, 3.3, 11.1, 11.1, 21.2, 21.2, 31.3, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];
        let expected_isplit = [-1, -2, -13, -12, -11, -10, -9, -3, -7, -6, -5, -4, -3, -2, -4, -5, 1, 2, 3, 4, 5, 6, -6, 8, 9, 10, 11, 12, 13, 14];
        let expected_p = [2, 5, 4, 1, 3, 0];
        let expected_xbest = [0.0, 11.0, 22.0, 30.0, 41.0, 52.0];
        let expected_fbest = 31.3;
        let expected_nboxes = 30;

        assert_eq!(ipar, expected_ipar);
        assert_eq!(level, expected_level);
        assert_eq!(ichild, expected_ichild);
        assert_eq!(isplit, expected_isplit);
        assert_eq!(p.as_slice(), expected_p);
        assert_eq!(xbest.as_slice(), expected_xbest);
        assert_eq!(fbest, expected_fbest);
        assert_eq!(f.iter().flatten().cloned().collect::<Vec<f64>>().as_slice(), expected_f);
        assert_eq!(nboxes, expected_nboxes);
    }
}