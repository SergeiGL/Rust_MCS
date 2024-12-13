use crate::feval::feval;
use crate::sign::sign;
use crate::{polint::polint, quadratic_func::quadmin, quadratic_func::quadpol};
use nalgebra::SMatrix;

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

pub fn init<const N: usize>(x0: &SMatrix<f64, N, 3>, l: &[usize; N], L: &[usize; N]) -> (
    [[f64; N]; 3], // f0
    [usize; N],    // istar
    usize          // ncall
) {
    let mut ncall = 0_usize;

    // Initialize x vector with zeros
    let mut x = [0.0; N];

    // Fetch initial point x0
    for i in 0..N {
        x[i] = x0[(i, l[i])];  // value at l[i] is the index of midpoint
    }

    // Compute initial function value
    let mut f1 = feval(&x);
    ncall += 1;

    // Initialize f0 matrix with zeros
    let mut f0 = [[0.0_f64; N]; 3]; // TODO: check L[0] is always = 3
    f0[l[0]][0] = f1;  // computing f(x) at initial point x0

    // Initialize istar vector for storing optimal indices
    let mut istar = [0_usize; N];

    // For all coordinates k (in this case i) in dim n
    for i in 0..N {
        istar[i] = l[i];  // set i* to midpoint

        // Iterate through possible values for current coordinate
        for j in 0..=L[i] {  // using =L[i] for inclusive range
            if j == l[i] {
                if i != 0 {
                    f0[j][i] = f0[istar[i - 1]][i - 1];
                }
            } else {
                x[i] = x0[(i, j)];
                f0[j][i] = feval(&x);
                ncall += 1;

                if f0[j][i] < f1 {
                    f1 = f0[j][i];
                    istar[i] = j;
                }
            }
        }

        // Update x[i] to the best found value
        x[i] = x0[(i, istar[i])];
    }

    (f0, istar, ncall)
}


pub fn initbox<const N: usize>(
    x0: &SMatrix<f64, N, 3>,
    f0: &[[f64; N]; 3],
    l: &[usize; N],
    L: &[usize; N],
    istar: &[usize; N],
    u: &[f64; N],
    v: &[f64; N],
    isplit: &mut Vec<isize>,
    level: &mut Vec<usize>,
    ipar: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    f: &mut [Vec<f64>; 2],
    nboxes: &mut usize,
) -> (
    [usize; N],   // p
    [f64; N],     // xbest
    f64           // fbest
) {
    ipar[0] = usize::MAX; // parent of root box is -1
    level[0] = 1;  // root box level is 1
    ichild[0] = 1; // root has one child initially
    f[0][0] = f0[l[0]][0];  // first element set to initial function values corresponding to l[0] and first function values

    let mut par = 0_usize; // parent index is 0
    let mut var = [0.0_f64; N]; // variability for dimensions initialized to 0

    // Iterate over each dimension
    for i in 0..N {
        // Set negative index for split information
        isplit[par] = -(i as isize) - 1;
        let mut nchild = 0;

        // If x0's left endpoint is greater than the lower bound `u[i]`, create an extra box
        if x0[(i, 0)] > u[i] {
            *nboxes += 1; // generate a new box
            nchild += 1;
            ipar[*nboxes] = par; // parent index
            level[*nboxes] = level[par] + 1; // Increment level for the child
            ichild[*nboxes] = -nchild; // update child information with negative value
            f[0][*nboxes] = f0[0][i]; // set function value
        }

        // Preparation for interpolation
        let v1 = if L[i] == 2 { v[i] } else { x0[(i, 2)] };

        // Perform polynomial interpolation and quadratic minimization
        let x0_arr: [f64; 3] = [
            x0[(i, 0)],
            x0[(i, 1)],
            x0[(i, 2)],
        ];
        let f0_arr: [f64; 3] = f0[0..3].iter().map(|r| r[i]).collect::<Vec<f64>>()
            .try_into().expect("Expected exactly 3 elements in f0");

        let mut d = polint(&x0_arr, &f0_arr);
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
        for j in 0..L[i] {
            *nboxes += 1;
            nchild += 1;

            // Decide the level increment, based on function value comparisons
            let s = if f0[j][i] <= f0[j + 1][i] { 1 } else { 2 };
            ipar[*nboxes] = par;
            level[*nboxes] = level[par] + s;
            ichild[*nboxes] = -nchild;
            f[0][*nboxes] = f0[j][i];  // assign function value

            if j >= 1 {
                if istar[i] == j {
                    if xl <= x0[(i, j)] {
                        par1 = *nboxes - 1;
                    } else {
                        par1 = *nboxes;
                    }
                }
                if j <= L[i] - 2 {
                    let x0_arr: [f64; 3] = [
                        x0[(i, j)],     // First column
                        x0[(i, j + 1)], // Second column
                        x0[(i, j + 2)], // Third column
                    ];
                    let f0_arr: [f64; 3] = f0[j..j + 3].iter().map(|r| r[i]).collect::<Vec<f64>>()
                        .try_into().expect("Expected exactly 3 elements in f0");
                    d = polint(&x0_arr, &f0_arr);
                    let u1 = if j < L[i] - 2 { x0[(i, j + 1)] } else { v[i] };

                    xl = quadmin(x0[(i, j)], u1, &d, &x0_arr);
                    fl = fl.min(quadpol(xl, &d, &x0_arr));

                    xu = quadmin(x0[(i, j)], u1, &[-d[0], -d[1], -d[2]], &x0_arr);
                    fu = fu.max(quadpol(xu, &d, &x0_arr));
                }
            }

            *nboxes += 1;
            nchild += 1;
            ipar[*nboxes] = par;
            level[*nboxes] = level[par] + 3 - s;
            ichild[*nboxes] = -nchild;
            f[0][*nboxes] = f0[j + 1][i]; // update function value for the next box
        }

        // If the upper end of x0 is below v, generate a final box
        if x0[(i, L[i])] < v[i] {
            *nboxes += 1;
            nchild += 1;
            ipar[*nboxes] = par;
            level[*nboxes] = level[par] + 1;
            ichild[*nboxes] = -nchild;
            f[0][*nboxes] = f0[L[i]][i];
        }

        if istar[i] == L[i] {
            if x0[(i, L[i])] < v[i] {
                if xl <= x0[(i, L[i])] {
                    par1 = *nboxes - 1;
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
    let fbest = f0[istar[N - 1]][N - 1];

    let mut p = [0_usize; N]; // stores the indices of best points
    let mut xbest = [0.0; N]; // best point values

    for i in 0..N {
        p[i] = var.iter().copied().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0; // find the maximum in var
        var[p[i]] = -1.0; // mark as used
        xbest[i] = x0[(i, istar[i])];  // store the best value of x at that index
    }

    (p, xbest, fbest)
}


#[cfg(test)]
mod tests {
    use super::*;

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

        let f0 = [
            [-0.62323147, -0.86038255, -0.5152638, -0.98834122, -0.37914019, -0.05313547],
            [-0.50531499, -0.62323147, -0.86038255, -0.86038255, -0.98834122, -0.98834122],
            [-0.08793206, -0.09355143, -0.32139218, -0.0251343, -0.65273189, -0.37750674],
        ];

        let l = [1, 1, 1, 1, 1, 1];
        let L = [2, 2, 2, 2, 2, 2];
        let istar = [0, 0, 1, 0, 1, 1];
        let u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];


        let mut isplit = vec![0_isize; N];
        let mut level = vec![0_usize; N];
        let mut ipar = vec![0_usize; N];
        let mut ichild = vec![0_isize; N];
        let mut f = [vec![0.0; N], vec![0.0; N]];
        let mut nboxes = 0_usize;

        let (p, xbest, fbest) =
            initbox(
                &x0, &f0, &l, &L, &istar, &u, &v,
                &mut isplit, &mut level, &mut ipar, &mut ichild,
                &mut f,
                &mut nboxes,
            );

        let expected_ipar = [
            usize::MAX, 0, 0, 0, 0, 1, 1, 1, 1, 5, 5, 5, 5,
            10, 10, 10, 10, 13, 13, 13, 13, 19, 19, 19, 19, 0,
            0, 0, 0, 0];
        let expected_level = [
            0, 0, 3, 2, 3, 0, 4, 3, 4, 5, 0, 4, 5, 0, 6, 5, 6,
            7, 6, 0, 7, 8, 7, 7, 8, 0, 0, 0, 0, 0];
        let expected_ichild = [
            1, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
            -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, 0,
            0, 0, 0, 0,
        ];
        let expected_f = [
            vec![-0.50531499, -0.62323147, -0.50531499, -0.50531499, -0.08793206, -0.86038255, -0.62323147, -0.62323147, -0.09355143, -0.5152638, -0.86038255, -0.86038255, -0.32139218, -0.98834122, -0.86038255, -0.86038255, -0.0251343, -0.37914019, -0.98834122, -0.98834122, -0.65273189, -0.05313547, -0.98834122, -0.98834122, -0.37750674, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ];
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
        assert_eq!(p, expected_p);
        assert_eq!(xbest, expected_xbest);
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
        let l = [1, 1, 1, 1, 1, 1];
        let L = [2, 2, 2, 2, 2, 2];

        let (f0, istar, ncall) = init(&x0, &l, &L);

        let expected_f0 = [[-0.6232314675898112, -0.8603825505022568, -0.5152637963551447, -0.9883412202327723, -0.3791401895175917, -0.05313547352279423], [-0.5053149917022333, -0.6232314675898112, -0.8603825505022568, -0.8603825505022568, -0.9883412202327723, -0.9883412202327723], [-0.08793206431638863, -0.09355142624190396, -0.3213921800858628, -0.025134295008083094, -0.6527318901582629, -0.37750674173452126]];
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
        let l = [1, 1, 1, 1, 1, 1];
        let L = [2, 2, 2, 2, 2, 2];

        let (f0, istar, ncall) = init(&x0, &l, &L);

        let expected_f0 = [[-3.408539273427753e-05; 6]; 3];
        let expected_istar = [1, 1, 1, 1, 1, 1];

        assert_eq!(f0, expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_edge_case_smallest_largest_indices() {
        let x0 = SMatrix::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ]);
        let l = [0, 0, 0, 0, 0, 0];
        let L = [2, 2, 2, 2, 2, 2];

        let (f0, istar, ncall) = init(&x0, &l, &L);

        let expected_f0 = [[-0.00508911288366444, -0.00508911288366444, -0.006592384416474927, -0.008816885534788802, -0.16446561922697936, -0.16473052321056048], [-0.004986271984857385, -0.005653858548782868, -0.008816885534788802, -0.16446561922697936, -0.16473052321056048, -0.09355142624190396], [-0.0010186628537568049, -0.006592384416474927, -0.006725151389454303, -0.028179437932502046, -0.15324289513062755, -0.036909774274296356]];
        let expected_istar = [0, 2, 1, 1, 1, 0];

        assert_eq!(f0, expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_0() {
        let x0 = SMatrix::from_row_slice(&[
            -10.0, 0.5, 1.0,
            -10.0, 0.5, 1.0,
            -10.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ]);
        let l = [0, 1, 2, 0, 1, 0];
        let L = [2, 0, 2, 2, 0, 0];

        let (f0, istar, ncall) = init(&x0, &l, &L);

        let expected_f0 = [[-8.535767023601033e-10, -8.531765565729684e-139, -8.201051614973535e-05, -0.0491552890369112, -0.7882313661790343, 0.0], [-0.035667757842579986, 0.0, -0.0491552890369112, -0.7886441250514713, 0.0, 0.0], [-0.0011675413792504375, 0.0, 0.0, -0.13332552218667648, 0.0, 0.0]];
        let expected_istar = [1, 1, 1, 1, 1, 0];

        assert_eq!(&f0, &expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 9);
    }
}

