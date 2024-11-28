use crate::feval::feval;
use crate::sign::sign;
use crate::{polint::polint, quadratic_func::quadmin, quadratic_func::quadpol};

pub fn subint(x1: &mut f64, x2: &mut f64) {
    let f: f64 = 1000_f64;

    if f * x1.abs() < 1.0 && x2.abs() > f {
        *x2 = sign(*x2)
    } else if x2.abs() > f {
        *x2 = 10.0 * sign(*x2) * x1.abs();
    }
    *x1 += (*x2 - *x1) / 10.0;
}

pub fn init(x0: &Vec<Vec<f64>>, l: Vec<usize>, L: Vec<usize>, n: usize) -> (Vec<Vec<f64>>, Vec<usize>, usize) {
    // Initialize function call counter
    let mut ncall = 0_usize;

    // Initialize x vector with zeros
    let mut x: Vec<f64> = vec![0.0; n];

    // Fetch initial point x0
    for i in 0..n {
        x[i] = x0[i][l[i]];  // value at l[i] is the index of mid point
    }

    // Compute initial function value
    let mut f1 = feval(&x);
    ncall += 1;

    // Initialize f0 matrix with zeros
    let mut f0: Vec<Vec<f64>> = vec![vec![0.0; n]; L[0] + 1];
    f0[l[0]][0] = f1;  // computing f(x) at initial point x0

    // Initialize istar vector for storing optimal indices
    let mut istar: Vec<usize> = vec![0; n];

    // For all coordinates k (in this case i) in dim n
    for i in 0..n {
        istar[i] = l[i];  // set i* to mid point

        // Iterate through possible values for current coordinate
        for j in 0..=L[i] {  // using =L[i] for inclusive range
            if j == l[i] {
                if i != 0 {
                    f0[j][i] = f0[istar[i - 1]][i - 1];
                }
            } else {
                x[i] = x0[i][j];
                f0[j][i] = feval(&x);
                ncall += 1;

                if f0[j][i] < f1 {
                    f1 = f0[j][i];
                    istar[i] = j;
                }
            }
        }

        // Update x[i] to the best found value
        x[i] = x0[i][istar[i]];
    }

    (f0, istar, ncall)
}


pub fn initbox(
    x0: &[&[f64]], f0: &[&[f64]], l: &[usize], L: &[usize], istar: &[usize],
    u: &[f64], v: &[f64], isplit: &mut [i32], level: &mut [i32],
    ipar: &mut [i32], ichild: &mut [i32], f: &mut [&mut [f64]],
    nboxes: &mut usize,
) -> (Vec<usize>, Vec<f64>, f64) {
    let n = u.len(); // number of dimensions / coordinates

    ipar[0] = -1; // parent of root box is -1
    level[0] = 1;  // root box level is 1
    ichild[0] = 1; // root has one child initially
    f[0][0] = f0[l[0]][0];  // first element set to initial function values corresponding to l[0] and first function values

    let mut par = 0; // parent index is 0
    let mut var: Vec<f64> = vec![0.0; n]; // variability for dimensions initialized to 0

    // Iterate over each dimension
    for i in 0..n {
        // Set negative index for split information
        isplit[par] = -(i as i32) - 1;
        let mut nchild = 0;

        // If x0's left endpoint is greater than the lower bound `u[i]`, create an extra box
        if x0[i][0] > u[i] {
            *nboxes += 1; // generate a new box
            nchild += 1;
            ipar[*nboxes] = par as i32; // parent index
            level[*nboxes] = level[par] + 1; // Increment level for the child
            ichild[*nboxes] = -(nchild as i32); // update child information with negative value
            f[0][*nboxes] = f0[0][i]; // set function value
        }

        // Preparation for interpolation
        let v1 = if L[i] == 2 { v[i] } else { x0[i][2] };

        // Perform polynomial interpolation and quadratic minimization
        let x0_arr: [f64; 3] = x0[i][0..3].iter().copied().collect::<Vec<f64>>()
            .try_into().expect("Expected exactly 3 elements in x0");
        let f0_arr: [f64; 3] = f0[0..3].iter().map(|r| r[i]).collect::<Vec<f64>>()
            .try_into().expect("Expected exactly 3 elements in f0");

        let mut d = polint(&x0_arr, &f0_arr);
        let mut xl = quadmin(u[i], v1, &d, <&[f64; 3]>::try_from(&x0[i][0..3]).unwrap()); // left bound minimization
        let mut fl = quadpol(xl, &d, &x0_arr); // compute function value

        let mut xu = quadmin(u[i], v1, &[-d[0], -d[1], -d[2]], &x0_arr); // right bound minimization
        let mut fu = quadpol(xu, &d, &x0_arr);  // compute function value

        // Track which box the coordinate belongs to
        let mut par1 = 0;

        if istar[i] == 0 {
            if xl < x0[i][0] {
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
            ipar[*nboxes] = par as i32;
            level[*nboxes] = level[par] + s;
            ichild[*nboxes] = -(nchild as i32);
            f[0][*nboxes] = f0[j][i];  // assign function value

            if j >= 1 {
                if istar[i] == j {
                    if xl <= x0[i][j] {
                        par1 = *nboxes - 1;
                    } else {
                        par1 = *nboxes;
                    }
                }
                if j <= L[i] - 2 {
                    let x0_arr: [f64; 3] = x0[i][j..j + 1].iter().copied().collect::<Vec<f64>>()
                        .try_into().expect("Expected exactly 3 elements in x0");
                    let f0_arr: [f64; 3] = f0[j..j + 1].iter().map(|r| r[i]).collect::<Vec<f64>>()
                        .try_into().expect("Expected exactly 3 elements in f0");
                    d = polint(&x0_arr, &f0_arr);
                    let u1 = if j < L[i] - 2 { x0[i][j + 1] } else { v[i] };

                    xl = quadmin(x0[i][j], u1, &d, &x0_arr);
                    fl = fl.min(quadpol(xl, &d, &x0_arr));

                    xu = quadmin(x0[i][j], u1, &[-d[0], -d[1], -d[2]], &x0_arr);
                    fu = fu.max(quadpol(xu, &d, &x0_arr));
                }
            }

            *nboxes += 1;
            nchild += 1;
            ipar[*nboxes] = par as i32;
            level[*nboxes] = (level[par] + 3 - s) as i32;
            ichild[*nboxes] = -(nchild as i32);
            f[0][*nboxes] = f0[j + 1][i]; // update function value for the next box
        }

        // If the upper end of x0 is below v, generate a final box
        if x0[i][L[i]] < v[i] {
            *nboxes += 1;
            nchild += 1;
            ipar[*nboxes] = par as i32;
            level[*nboxes] = level[par] + 1;
            ichild[*nboxes] = -(nchild as i32);
            f[0][*nboxes] = f0[L[i]][i];
        }

        if istar[i] == L[i] {
            if x0[i][L[i]] < v[i] {
                if xl <= x0[i][L[i]] {
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
    let fbest = f0[istar[n - 1]][n - 1];

    let mut p: Vec<usize> = vec![0; n]; // stores the indices of best points
    let mut xbest: Vec<f64> = vec![0.0; n]; // best point values

    for i in 0..n {
        p[i] = var.iter().copied().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0; // find the maximum in var
        var[p[i]] = -1.0; // mark as used
        xbest[i] = x0[i][istar[i]];  // store the best value of x at that index
    }

    (p, xbest, fbest)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn initbox_test_minimal_valid_input() {
        let n = 30;

        // Inputs
        let x0: Vec<Vec<f64>> = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
        ];

        let f0: Vec<Vec<f64>> = vec![
            vec![-0.62323147, -0.86038255, -0.5152638, -0.98834122, -0.37914019, -0.05313547],
            vec![-0.50531499, -0.62323147, -0.86038255, -0.86038255, -0.98834122, -0.98834122],
            vec![-0.08793206, -0.09355143, -0.32139218, -0.0251343, -0.65273189, -0.37750674],
        ];

        let l: Vec<usize> = vec![1, 1, 1, 1, 1, 1];
        let L: Vec<usize> = vec![2, 2, 2, 2, 2, 2];
        let istar: Vec<usize> = vec![0, 0, 1, 0, 1, 1];
        let u: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Mutable arrays
        let mut isplit = vec![0_i32; n];
        let mut level = vec![0_i32; n];
        let mut ipar = vec![0_i32; n];
        let mut ichild = vec![0_i32; n];
        let mut f: Vec<Vec<f64>> = vec![vec![0.0; n], vec![0.0; n]];
        let mut nboxes = 0_usize;

        // Call the Rust function
        let (p, xbest, fbest) = initbox(
            &x0.iter().map(|v| &v[..]).collect::<Vec<&[f64]>>(),
            &f0.iter().map(|v| &v[..]).collect::<Vec<&[f64]>>(),
            &l, &L, &istar, &u, &v,
            &mut isplit, &mut level, &mut ipar, &mut ichild,
            &mut f.iter_mut().map(|v| &mut v[..]).collect::<Vec<&mut [f64]>>(),
            &mut nboxes,
        );

        let expected_ipar = vec![
            -1, 0, 0, 0, 0, 1, 1, 1, 1, 5, 5, 5, 5,
            10, 10, 10, 10, 13, 13, 13, 13, 19, 19, 19, 19, 0,
            0, 0, 0, 0];
        let expected_level = vec![
            0, 0, 3, 2, 3, 0, 4, 3, 4, 5, 0, 4, 5, 0, 6, 5, 6,
            7, 6, 0, 7, 8, 7, 7, 8, 0, 0, 0, 0, 0];
        let expected_ichild = vec![
            1, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
            -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, 0,
            0, 0, 0, 0,
        ];
        let expected_f = vec![
            vec![
                -0.50531499, -0.62323147, -0.50531499, -0.50531499, -0.08793206,
                -0.86038255, -0.62323147, -0.62323147, -0.09355143, -0.5152638,
                -0.86038255, -0.86038255, -0.32139218, -0.98834122, -0.86038255,
                -0.86038255, -0.0251343, -0.37914019, -0.98834122, -0.98834122,
                -0.65273189, -0.05313547, -0.98834122, -0.98834122, -0.37750674,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];
        let expected_isplit = vec![
            -1, -2, 0, 0, 0, -3, 0, 0, 0, 0, -4, 0, 0,
            -5, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];
        let expected_p = vec![3, 5, 1, 4, 2, 0];
        let expected_xbest = vec![0.0, 0.0, 0.5, 0.0, 0.5, 0.5];
        let expected_fbest = -0.98834122;
        let expected_nboxes = 24;

        for (a, b) in ipar.iter().zip(expected_ipar.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in level.iter().zip(expected_level.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in ichild.iter().zip(expected_ichild.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in isplit.iter().zip(expected_isplit.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in p.iter().zip(expected_p.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in xbest.iter().zip(expected_xbest.iter()) {
            assert_eq!(a, b);
        }
        assert_relative_eq!(fbest, -0.98834122);

        assert_eq!(nboxes, expected_nboxes);
    }

    #[test]
    fn subint_test_0() {
        let mut x1 = 1.0;
        let mut x2 = 2.0;
        subint(&mut x1, &mut x2);
        assert_relative_eq!(x1, 1.1);
        assert_relative_eq!(x2, 2.0);
    }

    #[test]
    fn subint_test_1() {
        let mut x1 = 0.0000001;
        let mut x2 = 2000.0;
        subint(&mut x1, &mut x2);
        assert_relative_eq!(x1, 0.10000009000000001);
        assert_relative_eq!(x2, 1.0);
    }


    #[test]
    fn subint_test_2() {
        let mut x1 = -1.0;
        let mut x2 = 50.0;
        subint(&mut x1, &mut x2);
        assert_relative_eq!(x1, 4.1);
        assert_relative_eq!(x2, 50.0);
    }

    #[test]
    fn subint_test_3() {
        let mut x1 = -0.00000001;
        let mut x2 = 132455.0;
        subint(&mut x1, &mut x2);
        assert_relative_eq!(x1, 0.099999991);
        assert_relative_eq!(x2, 1.0);
    }

    //---------------------------------------------------
    // Helper function to check if two 2D vectors are approximately equal
    fn assert_vec2d_relative_eq(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) {
        assert_eq!(a.len(), b.len(), "Vectors have different outer lengths");
        for (row_a, row_b) in a.iter().zip(b.iter()) {
            assert_eq!(row_a.len(), row_b.len(), "Rows have different lengths");
            for (val_a, val_b) in row_a.iter().zip(row_b.iter()) {
                assert_relative_eq!(*val_a, *val_b, max_relative = 1e-5);
            }
        }
    }

    #[test]
    fn init_test_base_case() {
        let x0 = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
        ];
        let l = vec![1, 1, 1, 1, 1, 1];
        let L = vec![2, 2, 2, 2, 2, 2];
        let n = 6;

        let (f0, istar, ncall) = init(&x0, l, L, n);

        let expected_f0 = vec![
            vec![-0.62323147, -0.86038255, -0.5152638, -0.98834122, -0.37914019, -0.05313547],
            vec![-0.50531499, -0.62323147, -0.86038255, -0.86038255, -0.98834122, -0.98834122],
            vec![-0.08793206, -0.09355143, -0.32139218, -0.0251343, -0.65273189, -0.37750674],
        ];
        let expected_istar = vec![0, 0, 1, 0, 1, 1];

        assert_vec2d_relative_eq(&f0, &expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_same_x0_values() {
        let x0 = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let l = vec![1, 1, 1, 1, 1, 1];
        let L = vec![2, 2, 2, 2, 2, 2];
        let n = 6;

        let (f0, istar, ncall) = init(&x0, l, L, n);

        let expected_f0 = vec![vec![-3.40853927e-5; 6]; 3];
        let expected_istar = vec![1, 1, 1, 1, 1, 1];

        assert_vec2d_relative_eq(&f0, &expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_edge_case_smallest_largest_indices() {
        let x0 = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
        ];
        let l = vec![0, 0, 0, 0, 0, 0];
        let L = vec![2, 2, 2, 2, 2, 2];
        let n = 6;

        let (f0, istar, ncall) = init(&x0, l, L, n);

        let expected_f0 = vec![
            vec![-0.00508911, -0.00508911, -0.00659238, -0.00881689, -0.16446562, -0.16473052],
            vec![-0.00498627, -0.00565386, -0.00881689, -0.16446562, -0.16473052, -0.09355143],
            vec![-0.00101866, -0.00659238, -0.00672515, -0.02817944, -0.1532429, -0.03690977],
        ];
        let expected_istar = vec![0, 2, 1, 1, 1, 0];

        assert_vec2d_relative_eq(&f0, &expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_0() {
        let x0 = vec![
            vec![-10.0, 0.5, 1.0],
            vec![-10.0, 0.5, 1.0],
            vec![-10.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
        ];
        let l = vec![0, 1, 2, 0, 1, 0];
        let L = vec![2, 0, 2, 2, 0, 0];
        let n = 6;

        let (f0, istar, ncall) = init(&x0, l, L, n);

        let expected_f0 = vec![
            vec![-8.53576702e-10, -8.53176557e-139, -8.20105161e-5, -4.91552890e-2, -7.88231366e-1, 0.0],
            vec![-0.03566776, 0.0, -0.04915529, -0.78864413, 0.0, 0.0],
            vec![-0.00116754, 0.0, 0.0, -0.13332552, 0.0, 0.0],
        ];
        let expected_istar = vec![1, 1, 1, 1, 1, 0];

        assert_vec2d_relative_eq(&f0, &expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 9);
    }
}

