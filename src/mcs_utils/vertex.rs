use crate::mcs_utils::{split::split1, updtf::updtf};
use nalgebra::{Matrix2xX, Matrix3xX, SMatrix, SVector};

// l is always full of 1 (2 in Matlab);
// L is always full of 2 (3 in Matlab)
#[inline]
pub(crate) fn vertex<const N: usize>(
    par: usize, // -1 from Matlab
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    v1: &SVector<f64, N>,
    x0: &SMatrix<f64, N, 3>,
    f0: &Matrix3xX<f64>,
    ipar: &Vec<Option<usize>>,
    isplit: &Vec<isize>,
    ichild: &Vec<isize>,
    z: &Matrix2xX<f64>,
    f: &Matrix2xX<f64>,
    n0: &mut [usize; N],
    x: &mut SVector<f64, N>,
    y: &mut [f64; N],
    x1: &mut [f64; N],
    x2: &mut [f64; N],
    f1: &mut [f64; N],
    f2: &mut [f64; N],
) {
    n0.fill(0);
    x.fill(f64::INFINITY);
    y.fill(f64::INFINITY);
    x1.fill(f64::INFINITY);
    x2.fill(f64::INFINITY);
    f1.fill(0.0);
    f2.fill(0.0);

    let mut fold = f[(0, par)];
    let mut m = par;

    while m != 0 {  // -1 from Matlab as j is -1 from Matlab
        debug_assert!(ipar[m].is_some());
        debug_assert!(ipar[m].unwrap() >= 1);

        let ipar_m = ipar[m].unwrap() - 1; // -1 as we will use at as index; Rust index is -1 from Matlab

        debug_assert!(isplit[ipar_m].abs() >= 1);
        let i = isplit[ipar_m].abs() as usize - 1; // -1 as we will use at as index; Rust index is -1 from Matlab
        n0[i] += 1;

        let ichild_m = ichild[m]; // as in Matlab
        if ichild_m == 1 {
            if x[i] == f64::INFINITY || x[i] == z[(0, ipar_m)] {
                vert1(z[(1, ipar_m)], f[(1, ipar_m)], &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
                x[i] = z[(0, ipar_m)];
            } else {
                fold = updtf(i, &x1, &x2, f1, f2, fold, f[(0, ipar_m)]);
                vert2(x[i], z[(0, ipar_m)], z[(1, ipar_m)], f[(0, ipar_m)], f[(1, ipar_m)], &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            }
        } else if ichild_m >= 2 {
            fold = updtf(i, &x1, &x2, f1, f2, fold, f[(0, ipar_m)]);
            if x[i] == f64::INFINITY || x[i] == z[(1, ipar_m)] {
                vert1(z[(0, ipar_m)], f[(0, ipar_m)], &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
                x[i] = z[(1, ipar_m)];
            } else {
                vert2(x[i], z[(1, ipar_m)], z[(0, ipar_m)], f[(1, ipar_m)], f[(0, ipar_m)], &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            }
        };

        if 1 <= ichild[m] && ichild[m] <= 2 && y[i] == f64::INFINITY {
            y[i] = split1(
                z[(0, ipar_m)],
                z[(1, ipar_m)],
                f[(0, ipar_m)],
                f[(1, ipar_m)],
            );
        };

        if ichild_m < 0 {
            let abs_ichild_m = ichild_m.abs() as usize;
            let half_abs_ichild_m = abs_ichild_m as f64 / 2.;
            let floor = half_abs_ichild_m.floor() as usize;
            let ceil = half_abs_ichild_m.ceil() as usize;
            // j2: as in Matlab
            // j1, j1_plus_j3, j1_plus_2_j3, j1_minus_j3: -1 from matlab
            let (j1, j2, j1_plus_j3, j1_plus_2_j3, j1_minus_j3) = if u[i] < x0[(i, 0)] {
                let j1 = ceil;
                let j2 = floor;
                if (half_abs_ichild_m < (j1 as f64) && j1 > 1) || (j1 == 3) {
                    // (j2, j1 - 1 - 1, j1 - 2 - 1, j1 + 1 - 1)
                    (j1 - 1, j2, j1 - 2, j1 - 3, j1)
                } else {
                    // (j2, j1 + 1 - 1, j1 + 2 - 1, j1 - 1 - 1)
                    (j1 - 1, j2, j1, j1 + 1, j1 - 2)
                }
            } else {
                let j1 = floor + 1;
                let j2 = ceil;
                if ((half_abs_ichild_m + 1.) > (j1 as f64)) && (j1 < 3) {
                    // (j2, j1 + 1 - 1, j1 + 2 - 1, j1 - 1 - 1)
                    if j1 < 2 {
                        (j1 - 1, j2, j1, j1 + 1, 0) // TODO: FIX this
                    } else {
                        (j1 - 1, j2, j1, j1 + 1, j1 - 2)
                    }
                } else {
                    // (j2, j1 - 1 - 1, j1 - 2 - 1, j1 + 1 - 1)
                    if j1 < 3 {
                        (j1 - 1, j2, j1 - 2, 0, j1)  // TODO: FIX this
                    } else {
                        (j1 - 1, j2, j1 - 2, j1 - 3, j1)
                    }
                }
            };
            // As we will use k as index, need -1 as Rust indexes are -1 from Matlab;
            let k = if isplit[ipar_m] < 0 {
                i
            } else {
                debug_assert!(z[(1, ipar_m)].abs() >= 1.);
                z[(1, ipar_m)] as usize - 1
            };

            if j1 != 1 || (x[i] != f64::INFINITY && x[i] != x0[(i, 1)]) {
                fold = updtf(i, &x1, &x2, f1, f2, fold, f0[(1, k)]);
            }

            if x[i] == f64::INFINITY || x[i] == x0[(i, j1)] {
                x[i] = x0[(i, j1)];
                match (x1[i], x2[i]) {
                    (f64::INFINITY, _) => {
                        vert3(j1, &x0, &f0, i, k, &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
                    }
                    (_, f64::INFINITY) => {
                        if x1[i] != x0[(i, j1_plus_j3)] {
                            x2[i] = x0[(i, j1_plus_j3)];
                            f2[i] += f0[(j1_plus_j3, k)];
                        } else if j1 != 0 && j1 != 2 {
                            x2[i] = x0[(i, j1_minus_j3)];
                            f2[i] += f0[(j1_minus_j3, k)];
                        } else {
                            x2[i] = x0[(i, j1_plus_2_j3)];
                            f2[i] += f0[(j1_plus_2_j3, k)];
                        }
                    }
                    _ => {}
                }
            } else {
                debug_assert!(i < N && j1 < 3 && j1_plus_j3 < 3 && j1_plus_j3 < N);

                match (x1[i], x2[i]) {
                    (f64::INFINITY, _) => {
                        x1[i] = x0[(i, j1)];
                        f1[i] += f0[(j1, k)];
                        if x[i] != x0[(i, j1_plus_j3)] {
                            x2[i] = x0[(i, j1_plus_j3)];
                            f2[i] += f0[(j1_plus_j3, k)];
                        }
                    }
                    (_, f64::INFINITY) => {
                        if x1[i] != x0[(i, j1)] {
                            x2[i] = x0[(i, j1)];
                            f2[i] += f0[(j1, k)];
                        } else if x[i] != x0[(i, j1_plus_j3)] {
                            x2[i] = x0[(i, j1_plus_j3)];
                            f2[i] += f0[(j1_plus_j3, k)];
                        } else if j1 != 0 && j1 != 2 {
                            x2[i] = x0[(i, j1_plus_j3)];
                            f2[i] += f0[(j1_plus_j3, k)];
                        } else {
                            x2[i] = x0[(i, j1_plus_2_j3)];
                            f2[i] += f0[(j1_plus_2_j3, k)];
                        }
                    }
                    _ => {}
                }
            }
            if y[i] == f64::INFINITY {
                y[i] = match j2 { // as in Matlab
                    0 => u[i],
                    3 => v[i],
                    _ => split1(x0[(i, j2 - 1)], x0[(i, j2)], f0[(j2 - 1, k)], f0[(j2, k)]),
                };
            }
        }
        m = match ipar[m] { // ipar[m] is same as in Matlab, BUT m is -1 from Matlab! => need to -1
            None | Some(0) | Some(1) => 0, // stop the cycle
            Some(two_or_more) => two_or_more - 1
        };
    }

    for i in 0..N {
        if x[i] == f64::INFINITY {
            x[i] = x0[(i, 1)];
            vert3(1, &x0, &f0, i, i, &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
        }

        if y[i] == f64::INFINITY {
            y[i] = v1[i];
        }
    }
}


#[inline]
fn vert1(z_j: f64, f_j: f64, x1: &mut f64, x2: &mut f64, f1: &mut f64, f2: &mut f64) {
    match (*x1, *x2) {
        (f64::INFINITY, _) => {
            *x1 = z_j;
            *f1 += f_j;
        }
        (x_1, f64::INFINITY) if x_1 != z_j => {
            *x2 = z_j;
            *f2 += f_j;
        }
        _ => {}
    }
}


#[inline]
fn vert2(x: f64, z_j: f64, z_j1: f64, f_j: f64, f_j1: f64,
         x1: &mut f64, x2: &mut f64,
         f1: &mut f64, f2: &mut f64) {
    match (*x1, *x2) {
        (f64::INFINITY, _) => {
            *x1 = z_j;
            *f1 += f_j;
            if x != z_j1 {
                *x2 = z_j1;
                *f2 += f_j1;
            }
        }
        (_, f64::INFINITY) => {
            if *x1 != z_j {
                *x2 = z_j;
                *f2 += f_j;
            } else {
                *x2 = z_j1;
                *f2 += f_j1;
            }
        }
        _ => {}
    }
}


// L is always 3 in Matlab
#[inline]
fn vert3<const N: usize>(
    j: usize, // -1 from Matlab
    x0: &SMatrix<f64, N, 3>,
    f0: &Matrix3xX<f64>,
    i: usize, // -1 from Matlab
    k: usize,
    x1: &mut f64,
    x2: &mut f64,
    f1: &mut f64,
    f2: &mut f64,
) {
    let (k1, k2) = match j {
        0 => (1, 2),
        2 => (0, 1), // (3-2-1, 3-1-1)
        _ => (j - 1, j + 1)
    };
    *x1 = x0[(i, k1)];
    *x2 = x0[(i, k2)];
    *f1 += f0[(k1, k)];
    *f2 += f0[(k2, k)];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // l = ones(100)*2; % constant; do not touch
        // L = ones(100)*3; % constant; do not touch
        // n = 6; % const N in Rust
        // j = 1; % +1 from Rust value
        // u = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]; % column vector
        // v = [10.0; 10.0; 10.0; 10.0; 10.0; 10.0]; % column vector
        // v1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // x0 = [
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // [0.0, 0.5, 1.0];
        // ];
        // f0 = [
        // [-0.5, -0.8, -0.3, -0.5, -0.8, -0.3];
        // [-0.6, -0.9, -0.4, -0.6, -0.9, -0.4];
        // [-0.7, -1.0, -0.5, -0.7, -1.0, -0.5];
        // ];
        // ipar = [0];
        // isplit = [-1];
        // ichild = [1];
        // z = [[0.0]; [inf]];
        // f = [[-0.5]; [0.0]];
        //
        // n0 = [0,0,0,0,0,0];
        // x = [0,0,0,0,0,0];
        // y = [0,0,0,0,0,0];
        // x1 = [0,0,0,0,0,0];
        // x2 = [0,0,0,0,0,0];
        // f1 = [0,0,0,0,0,0];
        // f2 = [0,0,0,0,0,0];
        //
        // [n0,x,y,x1,x2,f1,f2] = vertex(j,n,u,v,v1,x0,f0,ipar,isplit,ichild,z,f,l,L)

        let j: usize = 0; // -1 from Matlab
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let v1 = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let x0 = SMatrix::<f64, 6, 3>::from_row_slice(&[0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.5, -0.8, -0.3, -0.5, -0.8, -0.3,
            -0.6, -0.9, -0.4, -0.6, -0.9, -0.4,
            -0.7, -1.0, -0.5, -0.7, -1.0, -0.5,
        ]);
        let ipar = vec![Some(0)];
        let isplit = vec![-1];
        let ichild = vec![1];
        let z = Matrix2xX::<f64>::from_row_slice(&[0.0, f64::INFINITY]);
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, 0.0]);
        let mut n0 = [0; 6];
        let mut x = SVector::<f64, 6>::zeros();
        let mut y = [0.; 6];
        let mut x1 = [0.; 6];
        let mut x2 = [0.; 6];
        let mut f1 = [0.; 6];
        let mut f2 = [0.; 6];

        vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0, [0, 0, 0, 0, 0, 0]);
        assert_eq!(x.as_slice(), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        assert_eq!(y.as_slice(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(x1.as_slice(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(x2.as_slice(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(f1.as_slice(), [-0.5000, -0.8000, -0.3000, -0.5000, -0.8000, -0.3000]);
        assert_eq!(f2.as_slice(), [-0.7000, -1.0000, -0.5000, -0.7000, -1.0000, -0.5000]);
    }

    #[test]
    fn test_1() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // l = ones(100)*2; % constant; do not touch
        // L = ones(100)*3; % constant; do not touch
        // n = 4; % const N in Rust
        // j = 2; % +1 from Rust value
        // u = [1.0; 2.0; 3.0; 4.0];
        // v = [5.0; 6.0; 7.0; 8.0];
        // v1 = [0.5, 1.5, 2.5, 3.5];
        // x0 = [
        //     [0.0, 0.3, 0.6];
        //     [0.1, 0.4, 0.7];
        //     [0.2, 0.5, 0.8];
        //     [0.3, 0.6, 0.9];
        // ];
        // f0 = [
        //     [-0.1, -0.2, -0.3, -0.4];
        //     [-0.5, -0.6, -0.7, -0.8];
        //     [-0.9, -1.0, -1.1, -1.2];
        // ];
        // ipar = [3, 1];
        // isplit = [2, -3];
        // ichild = [-1, 2];
        // z = [[0.2, 0.3]; [0.4, 0.5]];
        // f = [[-0.3, -0.4]; [-0.5, -0.6]];
        //
        // n0 = [0,0,0,0];
        // x = [0,0,0,0];
        // y = [0,0,0,0];
        // x1 = [0,0,0,0];
        // x2 = [0,0,0,0];
        // f1 = [0,0,0,0];
        // f2 = [0,0,0,0];
        //
        // format long g;
        // [n0,x,y,x1,x2,f1,f2] = vertex(j,n,u,v,v1,x0,f0,ipar,isplit,ichild,z,f,l,L)

        let j: usize = 1; // -1 from Matlab
        let u = SVector::<f64, 4>::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
        let v = SVector::<f64, 4>::from_row_slice(&[5.0, 6.0, 7.0, 8.0]);
        let v1 = SVector::<f64, 4>::from_row_slice(&[0.5, 1.5, 2.5, 3.5]);
        let x0 = SMatrix::<f64, 4, 3>::from_row_slice(&[
            0.0, 0.3, 0.6,
            0.1, 0.4, 0.7,
            0.2, 0.5, 0.8,
            0.3, 0.6, 0.9
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.1, -0.2, -0.3, -0.4,
            -0.5, -0.6, -0.7, -0.8,
            -0.9, -1.0, -1.1, -1.2,
        ]);
        let ipar = vec![Some(3), Some(1)];
        let isplit = vec![2, -3];
        let ichild = vec![-1, 2];
        let z = Matrix2xX::<f64>::from_row_slice(&[0.2, 0.3, 0.4, 0.5]);
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.3, -0.4, -0.5, -0.6]);
        let mut n0 = [0; 4];
        let mut x = SVector::<f64, 4>::zeros();
        let mut y = [0.; 4];
        let mut x1 = [0.; 4];
        let mut x2 = [0.; 4];
        let mut f1 = [0.; 4];
        let mut f2 = [0.; 4];

        vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [0, 1, 0, 0]);
        assert_eq!(x.as_slice(), [0.3, 0.4, 0.5, 0.6]);
        assert_eq!(y.as_slice(), [0.5, 0.27639320225002106, 2.5, 3.5]);
        assert_eq!(x1.as_slice(), [0., 0.2, 0.2, 0.3, ]);
        assert_eq!(x2.as_slice(), [0.6, f64::INFINITY, 0.8, 0.9, ]);
        assert_eq!(f1.as_slice(), [-0.20000000000000004, -0.3, -0.4, -0.5000]);
        assert_eq!(f2.as_slice(), [-1.0, 0., -1.2000000000000002, -1.3, ]);
    }
    #[test]
    fn test_3() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // l = ones(100)*2; % constant; do not touch
        // L = ones(100)*3; % constant; do not touch
        // n = 5; % const N in Rust
        // j = 3; % +1 from Rust value
        // u = [0.2; 0.4; 0.6; 0.8; 1.0];
        // v = [0.3; 0.5; 0.7; 0.9; 1.1];
        // v1 = [0.25, 0.45, 0.65, 0.85, 1.05];
        // x0 = [
        //     [0.1, 0.2, 0.3];
        //     [0.3, 0.4, 0.5];
        //     [0.5, 0.6, 0.7];
        //     [0.7, 0.8, 0.9];
        //     [0.9, 1.0, 1.1];
        // ];
        // f0 = [
        //     [0.01, 0.02, 0.03, 0.04, 0.05];
        //     [0.11, 0.12, 0.13, 0.14, 0.15];
        //     [0.21, 0.22, 0.23, 0.24, 0.25];
        // ];
        // ipar = [4, 5, 1];
        // isplit = [1, 3, -5];
        // ichild = [-5, -3, -1];
        // z = [[2, 1, 2]; [3, 4, 5]];
        // f = [[2, 1, 2]; [3, 4, 5]];
        //
        // n0 = [0,0,0,0,0];
        // x = [0,0,0,0,0];
        // y = [0,0,0,0,0];
        // x1 = [0,0,0,0,0];
        // x2 = [0,0,0,0,0];
        // f1 = [0,0,0,0,0];
        // f2 = [0,0,0,0,0];
        //
        // [n0,x,y,x1,x2,f1,f2] = vertex(j,n,u,v,v1,x0,f0,ipar,isplit,ichild,z,f,l,L)

        let j: usize = 2; // -1 from Matlab
        let u = SVector::<f64, 5>::from_row_slice(&[0.2, 0.4, 0.6, 0.8, 1.0]);
        let v = SVector::<f64, 5>::from_row_slice(&[0.3, 0.5, 0.7, 0.9, 1.1]);
        let v1 = SVector::<f64, 5>::from_row_slice(&[0.25, 0.45, 0.65, 0.85, 1.05]);
        let x0 = SMatrix::<f64, 5, 3>::from_row_slice(&[
            0.1, 0.2, 0.3,
            0.3, 0.4, 0.5,
            0.5, 0.6, 0.7,
            0.7, 0.8, 0.9,
            0.9, 1.0, 1.1
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            0.01, 0.02, 0.03, 0.04, 0.05,
            0.11, 0.12, 0.13, 0.14, 0.15,
            0.21, 0.22, 0.23, 0.24, 0.25,
        ]);
        let ipar = vec![Some(4), Some(5), Some(1)];
        let isplit = vec![1, 3, -5];
        let ichild = vec![-5, -3, -1];
        let z = Matrix2xX::<f64>::from_row_slice(&[2., 1., 2., 3., 4., 5.]);
        let f = Matrix2xX::<f64>::from_row_slice(&[2., 1., 2., 3., 4., 5.]);
        let mut n0 = [0; 5];
        let mut x = SVector::<f64, 5>::zeros();
        let mut y = [0.; 5];
        let mut x1 = [0.; 5];
        let mut x2 = [0.; 5];
        let mut f1 = [0.; 5];
        let mut f2 = [0.; 5];

        vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [1, 0, 0, 0, 0]);
        assert_eq!(x.as_slice(), [0.1, 0.4, 0.6, 0.8, 1.]);
        assert_eq!(y.as_slice(), [0.1618033988749895, 0.45, 0.65, 0.85, 1.05]);
        assert_eq!(x1.as_slice(), [0.2, 0.3, 0.5, 0.7, 0.9]);
        assert_eq!(x2.as_slice(), [0.3, 0.5, 0.7, 0.9, 1.1]);
        assert_eq!(f1.as_slice(), [0.13, 1.8900000000000001, 1.9000000000000001, 1.9100000000000001, 1.9200000000000002]);
        assert_eq!(f2.as_slice(), [0.23, 2.0900000000000003, 2.1, 2.1100000000000003, 2.12]);
    }

    #[test]
    fn test_4() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // l = ones(100)*2; % constant; do not touch
        // L = ones(100)*3; % constant; do not touch
        // n = 2; % const N in Rust
        // j = 3; % +1 from Rust value
        // u = [0.3; 0.7];
        // v = [0.6; 0.8];
        // v1 = [0.4, 0.75];
        // x0 = [
        //     [0.2, 0.4, 0.6];
        //     [0.5, 0.7, 0.9];
        // ];
        // f0 = [
        //     [0.25, 0.45];
        //     [0.35, 0.55];
        //     [0.65, 0.85];
        // ];
        // ipar = [3, 2, 1];
        // isplit = [1, 0, 2];
        // ichild = [2, 1, 3];
        // z = [[0.25, 0.55, 0.75]; [0.35, 0.65, 0.85]];
        // f = [[0.30, 0.40, 0.50]; [0.35, 0.45, 0.55]];
        //
        // n0 = [0,0];
        // x = [0,0];
        // y = [0,0];
        // x1 = [0,0];
        // x2 = [0,0];
        // f1 = [0,0];
        // f2 = [0,0];
        //
        // [n0,x,y,x1,x2,f1,f2] = vertex(j,n,u,v,v1,x0,f0,ipar,isplit,ichild,z,f,l,L)

        let j: usize = 2; // -1 from Matlab
        let u = SVector::<f64, 2>::from_row_slice(&[0.3, 0.7]);
        let v = SVector::<f64, 2>::from_row_slice(&[0.6, 0.8]);
        let v1 = SVector::<f64, 2>::from_row_slice(&[0.4, 0.75]);
        let x0 = SMatrix::<f64, 2, 3>::from_row_slice(&[
            0.2, 0.4, 0.6,
            0.5, 0.7, 0.9
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            0.25, 0.45,
            0.35, 0.55,
            0.65, 0.85,
        ]);
        let ipar = vec![Some(3), Some(2), Some(1)];
        let isplit = vec![1, 0, 2];
        let ichild = vec![2, 1, 3];
        let z = Matrix2xX::<f64>::from_row_slice(&[0.25, 0.55, 0.75, 0.35, 0.65, 0.85]);
        let f = Matrix2xX::<f64>::from_row_slice(&[0.30, 0.40, 0.50, 0.35, 0.45, 0.55]);
        let mut n0 = [0; 2];
        let mut x = SVector::<f64, 2>::zeros();
        let mut y = [0.; 2];
        let mut x1 = [0.; 2];
        let mut x2 = [0.; 2];
        let mut f1 = [0.; 2];
        let mut f2 = [0.; 2];

        vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [1, 0]);
        assert_eq!(x.as_slice(), [0.35, 0.7]);
        assert_eq!(y.as_slice(), [0.4, 0.75]);
        assert_eq!(x1.as_slice(), [0.25, 0.5]);
        assert_eq!(x2.as_slice(), [f64::INFINITY, 0.9]);
        assert_eq!(f1.as_slice(), [0.3, 0.65]);
        assert_eq!(f2.as_slice(), [0., 1.05]);
    }

    #[test]
    fn test_5() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // l = ones(100)*2; % constant; do not touch
        // L = ones(100)*3; % constant; do not touch
        // n = 3; % const N in Rust
        // j = 2; % +1 from Rust value
        // u = [0.1; 0.2; 0.3];
        // v = [0.9; 0.8; 0.7];
        // v1 = [0.5, 0.5, 0.5];
        // x0 = [
        //     [0.0, 0.5, 1.0];
        //     [0.0, 0.5, 1.0];
        //     [0.0, 0.5, 1.0];
        // ];
        // f0 = [
        //     [-0.2, -0.3, -0.4];
        //     [-0.5, -0.6, -0.7];
        //     [-0.8, -0.9, -1.0];
        // ];
        // ipar = [3, 1, 0];
        // isplit = [-1, 2, 0];
        // ichild = [1, -4, 2];
        // z = [[0.1, 0.2, 0.3]; [0.4, 0.5, 0.6]];
        // f = [[-0.1, -0.2, -0.3]; [-0.4, -0.5, -0.6]];
        //
        // n0 = [0,0,0];
        // x = [0,0,0];
        // y = [0,0,0];
        // x1 = [0,0,0];
        // x2 = [0,0,0];
        // f1 = [0,0,0];
        // f2 = [0,0,0];
        //
        // [n0,x,y,x1,x2,f1,f2] = vertex(j,n,u,v,v1,x0,f0,ipar,isplit,ichild,z,f,l,L)

        let j: usize = 1; // -1 from Matlab
        let u = SVector::<f64, 3>::from_row_slice(&[0.1, 0.2, 0.3]);
        let v = SVector::<f64, 3>::from_row_slice(&[0.9, 0.8, 0.7]);
        let v1 = SVector::<f64, 3>::from_row_slice(&[0.5, 0.5, 0.5]);
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.2, -0.3, -0.4,
            -0.5, -0.6, -0.7,
            -0.8, -0.9, -1.0,
        ]);
        let ipar = vec![Some(3), Some(1), Some(0)];
        let isplit = vec![-1, 2, 0];
        let ichild = vec![1, -4, 2];
        let z = Matrix2xX::<f64>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]);
        let mut n0 = [0; 3];
        let mut x = SVector::<f64, 3>::zeros();
        let mut y = [0.; 3];
        let mut x1 = [0.; 3];
        let mut x2 = [0.; 3];
        let mut f1 = [0.; 3];
        let mut f2 = [0.; 3];

        vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [1, 0, 0]);
        assert_eq!(x.as_slice(), [1., 0.5, 0.5]);
        assert_eq!(y.as_slice(), [0.6909830056250525, 0.5, 0.5]);
        assert_eq!(x1.as_slice(), [0., 0., 0., ]);
        assert_eq!(x2.as_slice(), [0.5, 1., 1.]);
        assert_eq!(f1.as_slice(), [-0.2, 0., -0.10000000000000003]);
        assert_eq!(f2.as_slice(), [-0.5, -0.6000000000000001, -0.7]);
    }

    #[test]
    fn test_6() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // l = ones(100)*2; % constant; do not touch
        // L = ones(100)*3; % constant; do not touch
        // n = 7; % const N in Rust
        // j = 5; % +1 from Rust value
        // u = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7];
        // v = [1.1; 1.2; 1.3; 1.4; 1.5; 1.6; 1.7];
        // v1 = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        // x0 = [
        //     [0.0, 0.5, 1.0];
        //     [0.1, 0.6, 1.1];
        //     [0.2, 0.7, 1.2];
        //     [0.3, 0.8, 1.3];
        //     [0.4, 0.9, 1.4];
        //     [0.5, 1.0, 1.5];
        //     [0.6, 1.1, 1.6];
        // ];
        // f0 = [
        //     [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
        //     [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26];
        //     [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36];
        // ];
        // ipar = [2, 1, 3, 4, 2];
        // isplit = [1, 3, 2, 3, 1];
        // ichild = [3, 2, 3, 1, 5];
        // z = [[1, 2, 3, 4, 5]; [1, 2, 3, 4, 5]];
        // f = [[0.01, 0.02, 0.03, 0.04, 0.05]; [0.06, 0.07, 0.08, 0.09, 0.10]];
        //
        // n0 = [0,0,0,0,0,0,0];
        // x = [0,0,0,0,0,0,0];
        // y = [0,0,0,0,0,0,0];
        // x1 = [0,0,0,0,0,0,0];
        // x2 = [0,0,0,0,0,0,0];
        // f1 = [0,0,0,0,0,0,0];
        // f2 = [0,0,0,0,0,0,0];
        //
        // [n0,x,y,x1,x2,f1,f2] = vertex(j,n,u,v,v1,x0,f0,ipar,isplit,ichild,z,f,l,L)

        let j: usize = 4; // -1 from Matlab
        let u = SVector::<f64, 7>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        let v = SVector::<f64, 7>::from_row_slice(&[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]);
        let v1 = SVector::<f64, 7>::from_row_slice(&[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
        let x0 = SMatrix::<f64, 7, 3>::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.1, 0.6, 1.1,
            0.2, 0.7, 1.2,
            0.3, 0.8, 1.3,
            0.4, 0.9, 1.4,
            0.5, 1.0, 1.5,
            0.6, 1.1, 1.6
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
            0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
            0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
        ]);
        let ipar = vec![Some(2), Some(1), Some(3), Some(4), Some(2)];
        let isplit = vec![1, 3, 2, 3, 1];
        let ichild = vec![3, 2, 3, 1, 5];
        let z = Matrix2xX::<f64>::from_row_slice(&[1., 2., 3., 4., 5., 1., 2., 3., 4., 5., ]);
        let f = Matrix2xX::<f64>::from_row_slice(&[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]);
        let mut n0 = [0; 7];
        let mut x = SVector::<f64, 7>::zeros();
        let mut y = [0.; 7];
        let mut x1 = [0.; 7];
        let mut x2 = [0.; 7];
        let mut f1 = [0.; 7];
        let mut f2 = [0.; 7];

        vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [1, 0, 1, 0, 0, 0, 0]);
        assert_eq!(x.as_slice(), [1., 0.6, 2., 0.8, 0.9, 1., 1.1]);
        assert_eq!(y.as_slice(), [1., 0.7, 0.8, 0.9, 1., 1.1, 1.2]);
        assert_eq!(x1.as_slice(), [1., 0.1, 2., 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(x2.as_slice(), [f64::INFINITY, 1.1, f64::INFINITY, 1.3, 1.4, 1.5, 1.6]);
        assert_eq!(f1.as_slice(), [0.04, 0.15, 0.02, 0.17, 0.18000000000000002, 0.19, 0.2]);
        assert_eq!(f2.as_slice(), [0.030000000000000002, 0.35, 0.01, 0.37, 0.38, 0.38999999999999996, 0.39999999999999997]);
    }

    #[test]
    fn test_7() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // l = ones(100)*2; % constant; do not touch
        // L = ones(100)*3; % constant; do not touch
        // n = 7; % const N in Rust
        // j = 5; % +1 from Rust value
        // u = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7];
        // v = [1.1; 1.2; 1.3; 1.4; 1.5; 1.6; 1.7];
        // v1 = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        // x0 = [
        //     [0.0, 0.5, 1.0];
        //     [0.1, 0.6, 1.1];
        //     [0.2, 0.7, 1.2];
        //     [0.3, 0.8, 1.3];
        //     [0.4, 0.9, 1.4];
        //     [0.5, 1.0, 1.5];
        //     [0.6, 1.1, 1.6];
        // ];
        // f0 = [
        //     [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
        //     [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26];
        //     [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36];
        // ];
        // ipar = [2, 1, 3, 4, 2];
        // isplit = [1, 3, 2, 3, 1];
        // ichild = [-3, -2, -3, -1, -5];
        // z = [[1, 2, 3, 4, 5]; [1, 2, 3, 4, 5]];
        // f = [[0.01, 0.02, 0.03, 0.04, 0.05]; [0.06, 0.07, 0.08, 0.09, 0.10]];
        //
        // n0 = [0,0,0,0,0,0,0];
        // x = [0,0,0,0,0,0,0];
        // y = [0,0,0,0,0,0,0];
        // x1 = [0,0,0,0,0,0,0];
        // x2 = [0,0,0,0,0,0,0];
        // f1 = [0,0,0,0,0,0,0];
        // f2 = [0,0,0,0,0,0,0];
        //
        // [n0,x,y,x1,x2,f1,f2] = vertex(j,n,u,v,v1,x0,f0,ipar,isplit,ichild,z,f,l,L)

        let j: usize = 4; // -1 from Matlab
        let u = SVector::<f64, 7>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        let v = SVector::<f64, 7>::from_row_slice(&[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]);
        let v1 = SVector::<f64, 7>::from_row_slice(&[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
        let x0 = SMatrix::<f64, 7, 3>::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.1, 0.6, 1.1,
            0.2, 0.7, 1.2,
            0.3, 0.8, 1.3,
            0.4, 0.9, 1.4,
            0.5, 1.0, 1.5,
            0.6, 1.1, 1.6
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
            0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
            0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
        ]);
        let ipar = vec![Some(2), Some(1), Some(3), Some(4), Some(2)];
        let isplit = vec![1, 3, 2, 3, 1];
        let ichild = vec![-3, -2, -3, -1, -5];
        let z = Matrix2xX::<f64>::from_row_slice(&[1., 2., 3., 4., 5., 1., 2., 3., 4., 5., ]);
        let f = Matrix2xX::<f64>::from_row_slice(&[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]);
        let mut n0 = [0; 7];
        let mut x = SVector::<f64, 7>::zeros();
        let mut y = [0.; 7];
        let mut x1 = [0.; 7];
        let mut x2 = [0.; 7];
        let mut f1 = [0.; 7];
        let mut f2 = [0.; 7];

        vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [1, 0, 1, 0, 0, 0, 0]);
        assert_eq!(x.as_slice(), [0.5, 0.6, 1.2, 0.8, 0.9, 1., 1.1]);
        assert_eq!(y.as_slice(), [0.30901699437494745, 0.7, 1.3, 0.9, 1., 1.1, 1.2]);
        assert_eq!(x1.as_slice(), [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(x2.as_slice(), [1., 1.1, 0.7, 1.3, 1.4, 1.5, 1.6]);
        assert_eq!(f1.as_slice(), [-0.05999999999999997, -0.049999999999999975, 0.11, -0.02999999999999997, -0.019999999999999962, -0.009999999999999981, 2.7755575615628914e-17]);
        assert_eq!(f2.as_slice(), [0.14, 0.15000000000000002, 0.21, 0.17000000000000004, 0.18000000000000005, 0.19, 0.2]);
    }
}