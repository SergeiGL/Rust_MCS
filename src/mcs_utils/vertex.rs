use crate::mcs_utils::{split_func::split1, updtf::updtf};
use nalgebra::{Matrix2xX, Matrix3xX, SMatrix, SVector};


#[inline]
fn vert1(
    z_j: f64,
    f_j: f64,
    x1: &mut f64,
    x2: &mut f64,
    f1: &mut f64,
    f2: &mut f64,
) {
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
fn vert2(
    x: f64,
    z_j: f64,
    z_j1: f64,
    f_j: f64,
    f_j1: f64,
    x1: &mut f64,
    x2: &mut f64,
    f1: &mut f64,
    f2: &mut f64,
) {
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

#[inline]
const fn wrap_index(x: usize) -> usize {
    match x {
        0 => 2,
        1 => 0,
        2 => 1,
        3 => 2,  // Handle special case j1 == 3
        _ => unreachable!(),
    }
}

// l is always full of 1;
// L is always full of 2
pub fn vertex<const N: usize>(
    j: usize,
    u: &SVector<f64, N>,
    v1: &SVector<f64, N>,
    x0: &SMatrix<f64, N, 3>,
    f0: &Matrix3xX<f64>,
    ipar: &Vec<Option<usize>>,
    isplit: &Vec<isize>,
    ichild: &Vec<isize>,
    z: &Matrix2xX<f64>,
    f: &Matrix2xX<f64>,
    n0: &mut SVector<usize, N>,
    x: &mut SVector<f64, N>,
    y: &mut SVector<f64, N>,
    x1: &mut SVector<f64, N>,
    x2: &mut SVector<f64, N>,
    f1: &mut SVector<f64, N>,
    f2: &mut SVector<f64, N>,
) {
    n0.fill(0);
    x.fill(f64::INFINITY);
    y.fill(f64::INFINITY);
    x1.fill(f64::INFINITY);
    x2.fill(f64::INFINITY);
    f1.fill(0.0);
    f2.fill(0.0);

    let mut fold = f[(0, j)];
    let mut m = j;

    while m > 0 {
        let ipar_m = ipar[m].unwrap_or(f.ncols() - 1);
        let ichild_m = ichild[m];

        let i = match isplit[ipar_m] {
            val if val >= 0 => val as usize,
            0 => N - 1,
            val => val.abs() as usize - 1,
        };
        n0[i] += 1;

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

        if (ichild_m == 1 || ichild_m == 2) && y[i] == f64::INFINITY {
            y[i] = split1(
                z[(0, ipar_m)],
                z[(1, ipar_m)],
                f[(0, ipar_m)],
                f[(1, ipar_m)],
            );
        };

        if ichild_m < 0 {
            let abs_ichild_m = ichild_m.abs() as usize;
            let floor = abs_ichild_m >> 1;  // Faster than division by 2
            let ceil = (abs_ichild_m + 1) >> 1;

            let (j1, j2, j3) = if u[i] < x0[(i, 0)] {
                let j1 = ceil;
                let j3 = !(((abs_ichild_m & 1) == 0 && j1 != 0) || j1 == 3);
                (wrap_index(j1), wrap_index(floor), j3)
            } else {
                let j1 = floor;
                let j3 = (abs_ichild_m > j1 * 2) || (j1 < 2);
                (wrap_index(j1 + 1), wrap_index(ceil), j3)
            };

            let (j1_plus_j3, j1_plus_2_j3, j1_minus_j3) = if j3 {
                match j1 {
                    0 => (1, 2, 2),
                    1 => (2, 0, 0),
                    2 => (0, 1, 1),
                    _ => unreachable!(),
                }
            } else {
                match j1 {
                    0 => (2, 1, 1),
                    1 => (0, 2, 2),
                    2 => (1, 0, 0),
                    _ => unreachable!(),
                }
            };

            let k = if isplit[ipar_m] < 0 { i } else { z[(0, ipar_m)] as usize };

            if j1 != 1 || (x[i] != f64::INFINITY && x[i] != x0[(i, 1)]) {
                fold = updtf(i, &x1, &x2, f1, f2, fold, f0[(1, k)]);
            }

            if x[i] == f64::INFINITY || x[i] == x0[(i, j1)] {
                x[i] = x0[(i, j1)];
                match (x1[i], x2[i]) {
                    (f64::INFINITY, _) => {
                        let (k1, k2) = match j1 {
                            0 => (1, 2),
                            1 => (0, 2),
                            2 => (0, 1),
                            _ => unreachable!(),
                        };

                        x1[i] = x0[(i, k1)];
                        x2[i] = x0[(i, k2)];
                        f1[i] += f0[(k1, k)];
                        f2[i] += f0[(k2, k)];
                    }
                    (_, f64::INFINITY) => {
                        if x1[i] != x0[(i, j1_plus_j3)] {
                            x2[i] = x0[(i, j1_plus_j3)];
                            f2[i] += f0[(j1_plus_j3, k)];
                        } else if j1 != 1 && j1 != 2 {
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
                        } else if j1 != 1 && j1 != 2 {
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
                y[i] = match j2 {
                    2 => u[i],
                    _ => split1(x0[(i, j2)], x0[(i, j2 + 1)], f0[(j2, k)], f0[(j2 + 1, k)]),
                };
            }
        }
        m = ipar[m].unwrap_or(0); // Should be -1 but the loop will stop anyway but i need m to remain usize
    }

    for i in 0..N {
        if x[i] == f64::INFINITY {
            x[i] = x0[(i, 1)];
            x1[i] = x0[(i, 0)];
            x2[i] = x0[(i, 2)];
            f1[i] += f0[(0, i)];
            f2[i] += f0[(2, i)];
        }

        if y[i] == f64::INFINITY {
            y[i] = v1[i];
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let j: usize = 0;
        let u = SVector::<f64, 1>::from_row_slice(&[0.0]);
        let v1 = SVector::<f64, 1>::from_row_slice(&[1.0]);
        let x0 = SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 0.5, 1.0]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.5, -0.8, -0.3,
            -0.6, -0.9, -0.4,
            -0.7, -1.0, -0.5,
        ]);
        let ipar = vec![Some(0)];
        let isplit = vec![-1];
        let ichild = vec![1];
        let z = Matrix2xX::<f64>::from_row_slice(&[0.0, f64::INFINITY]);
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, 0.0]);
        let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) =
            (SVector::<usize, 1>::zeros(), SVector::<f64, 1>::repeat(f64::INFINITY), SVector::<f64, 1>::repeat(f64::INFINITY), SVector::<f64, 1>::repeat(f64::INFINITY), SVector::<f64, 1>::repeat(f64::INFINITY), SVector::<f64, 1>::zeros(), SVector::<f64, 1>::zeros());

        vertex(j, &u, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [0]);
        assert_eq!(x.as_slice(), [0.5]);
        assert_eq!(y.as_slice(), [1.0]);
        assert_eq!(x1.as_slice(), [0.0]);
        assert_eq!(x2.as_slice(), [1.0]);
        assert_eq!(f1.as_slice(), [-0.5]);
        assert_eq!(f2.as_slice(), [-0.7]);
    }

    #[test]
    fn test_1() {
        let j: usize = 0;
        let u = SVector::<f64, 3>::from_row_slice(&[0.0, 0.0, 0.0]);
        let v1 = SVector::<f64, 3>::from_row_slice(&[1.0, 1.0, 1.0]);
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ]);
        let f0 = Matrix3xX::<f64>::zeros(3);
        let ipar = vec![Some(1), Some(2), Some(3)];
        let isplit = vec![3, 2, 1];
        let ichild = vec![1, 2, 1];
        let z = Matrix2xX::<f64>::from_row_slice(&[f64::INFINITY; 6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[0.0; 6]);
        let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) =
            (SVector::<usize, 3>::zeros(), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::zeros(), SVector::<f64, 3>::zeros());

        vertex(j, &u, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [0, 0, 0]);
        assert_eq!(x.as_slice(), [0.5, 0.5, 0.5]);
        assert_eq!(y.as_slice(), [1.0, 1.0, 1.0]);
        assert_eq!(x1.as_slice(), [0.0, 0.0, 0.0]);
        assert_eq!(x2.as_slice(), [1.0, 1.0, 1.0]);
        assert_eq!(f1.as_slice(), [0.0, 0.0, 0.0]);
        assert_eq!(f2.as_slice(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_2() {
        let j: usize = 0;
        let u = SVector::<f64, 3>::from_row_slice(&[0.0, 0.0, 0.0]);
        let v1 = SVector::<f64, 3>::from_row_slice(&[-1.0, -1.0, -1.0]);
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -0.0, -0.5, -1.0,
            -0.0, -0.5, -1.0,
            -0.0, -0.5, -1.0,
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![Some(3), Some(2), Some(3)];
        let isplit = vec![1, 1, 1];
        let ichild = vec![1, 2, 1];
        let z = Matrix2xX::<f64>::from_row_slice(&[f64::INFINITY; 6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) =
            (SVector::<usize, 3>::zeros(), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::zeros(), SVector::<f64, 3>::zeros());

        vertex(j, &u, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [0, 0, 0]);
        assert_eq!(x.as_slice(), [-0.5, -0.5, -0.5]);
        assert_eq!(y.as_slice(), [-1.0, -1.0, -1.0]);
        assert_eq!(x1.as_slice(), [-0.0, -0.0, -0.0]);
        assert_eq!(x2.as_slice(), [-1.0, -1.0, -1.0]);
        assert_eq!(f1.as_slice(), [1.0, 1.0, 1.0]);
        assert_eq!(f2.as_slice(), [1.0, 1.0, 1.0]);
    }


    #[test]
    fn test_3() {
        let j: usize = 1;
        let u = SVector::<f64, 3>::from_row_slice(&[0.0, 0.0, 0.0]);
        let v1 = SVector::<f64, 3>::from_row_slice(&[-1.0, -2.0, -3.0]);
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -0.0, -0.5, -1.0,
            -0.1, -0.52, -1.2,
            0.2, 0.5, 2.0,
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![None, Some(2), Some(0)];
        let isplit = vec![-1, -2, -3];
        let ichild = vec![-3, -1, -1];
        let z = Matrix2xX::<f64>::from_row_slice(&[1.1, 1.3, 1.0, 2.2, 1.22, 2.23]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.11, 1.31, 1.01, 2.22, 1.222, 2.232]);
        let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) =
            (SVector::<usize, 3>::zeros(), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::zeros(), SVector::<f64, 3>::zeros());

        vertex(j, &u, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [1, 0, 1]);
        assert_eq!(x.as_slice(), [-0., -0.52, 0.2]);
        assert_eq!(y.as_slice(), [-0.30901699437494745, -2.0, 0.0]);
        assert_eq!(x1.as_slice(), [-0.5, -0.1, 0.5]);
        assert_eq!(x2.as_slice(), [-1.0, -1.2, 2.0]);
        assert_eq!(f1.as_slice(), [1.31, 1.31, 1.0]);
        assert_eq!(f2.as_slice(), [1.31, 1.31, 1.0]);
    }

    #[test]
    fn test_4() {
        let j: usize = 1;
        let u = SVector::<f64, 3>::from_row_slice(&[0.0, 0.0, 0.0]);
        let v1 = SVector::<f64, 3>::from_row_slice(&[3.0, 1.0, -3.0]);
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            3.0, 1.5, 1.0,
            1.0, 0.5, -1.0,
            -1.0, -1.5, 2.0
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![Some(2), Some(2), Some(0)];
        let isplit = vec![-1, -2, -3];
        let ichild = vec![-3, 1, -1];
        let z = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) =
            (SVector::<usize, 3>::zeros(), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::zeros(), SVector::<f64, 3>::zeros());

        vertex(j, &u, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [1, 0, 1]);
        assert_eq!(x.as_slice(), [3., 0.5, 1.]);
        assert_eq!(y.as_slice(), [0., 1., 1.]);
        assert_eq!(x1.as_slice(), [1.5, 1., 1.]);
        assert_eq!(x2.as_slice(), [1., -1., f64::INFINITY]);
        assert_eq!(f1.as_slice(), [1., 1., 1.]);
        assert_eq!(f2.as_slice(), [1., 1., 0.]);
    }


    #[test]
    fn test_5() {
        let j: usize = 1;
        let u = SVector::<f64, 3>::from_row_slice(&[0.0, 0.0, 0.0]);
        let v1 = SVector::<f64, 3>::from_row_slice(&[3.0, 1.0, -3.0]);
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            3.0, 1.5, 1.0,
            1.0, 0.5, -1.0,
            -1.0, -1.5, 2.0
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![Some(2), Some(2), Some(0)];
        let isplit = vec![0, 0, 0];
        let ichild = vec![1, 1, 1];
        let z = Matrix2xX::<f64>::from_row_slice(&[1., 2., 3., 5., 6., 7.]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) =
            (SVector::<usize, 3>::zeros(), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::repeat(f64::INFINITY), SVector::<f64, 3>::zeros(), SVector::<f64, 3>::zeros());

        vertex(j, &u, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        assert_eq!(n0.as_slice(), [2, 0, 0]);
        assert_eq!(x.as_slice(), [3., 0.5, -1.5]);
        assert_eq!(y.as_slice(), [5.47213595499958, 1., -3.]);
        assert_eq!(x1.as_slice(), [7., 1., -1.]);
        assert_eq!(x2.as_slice(), [1., -1., 2.]);
        assert_eq!(f1.as_slice(), [1., 1., 1.]);
        assert_eq!(f2.as_slice(), [1., 1., 1.]);
    }
}