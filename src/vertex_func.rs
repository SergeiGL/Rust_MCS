use crate::split_func::split1;
use crate::updtf::updtf;
use nalgebra::{Const, Matrix2xX, Matrix3xX, MatrixView, SMatrix, U1, U2, U3};

enum J3 {
    One,
    MinusOne,
}


fn vert1(
    j: usize,
    z: MatrixView<f64, U2, U1, U1, U2>,
    f: MatrixView<f64, U2, U1, U1, U2>,
    x1: &mut f64,
    x2: &mut f64,
    f1: &mut f64,
    f2: &mut f64,
) -> f64 {
    let j1 = if j == 0 { 1 } else { 0 };

    let x = z[j1];
    if *x1 == f64::INFINITY {
        *x1 = z[j];
        *f1 += f[j];
    } else if *x2 == f64::INFINITY && *x1 != z[j] {
        *x2 = z[j];
        *f2 += f[j];
    }
    x
}

fn vert2(
    j: usize,
    x: f64,
    z: MatrixView<f64, U2, U1, U1, U2>,
    f: MatrixView<f64, U2, U1, U1, U2>,
    x1: &mut f64,
    x2: &mut f64,
    f1: &mut f64,
    f2: &mut f64,
) {
    let j1 = if j == 0 { 1 } else { 0 };

    if *x1 == f64::INFINITY {
        *x1 = z[j];
        *f1 += f[j];
        if x != z[j1] {
            *x2 = z[j1];
            *f2 += f[j1];
        }
    } else if *x2 == f64::INFINITY && *x1 != z[j] {
        *x2 = z[j];
        *f2 += f[j];
    } else if *x2 == f64::INFINITY {
        *x2 = z[j1];
        *f2 += f[j1];
    }
}

fn vert3<const N: usize>(
    j: usize,
    x0: MatrixView<f64, U1, Const<3>, Const<1>, Const<{ N }>>,
    f0: MatrixView<f64, U3, U1, U1, U3>,
    L: usize,
    x1: &mut f64,
    x2: &mut f64,
    f1: &mut f64,
    f2: &mut f64,
) {
    let (k1, k2) = if j == 0 {
        (1, 2)
    } else if j == L {
        (L - 2, L - 1)
    } else {
        (j - 1, j + 1)
    };

    *x1 = x0[k1];
    *x2 = x0[k2];
    *f1 += f0[k1];
    *f2 += f0[k2];
}


pub fn vertex<const N: usize>(
    j: usize,
    u: &[f64; N],
    v: &[f64; N],
    v1: &[f64; N],
    x0: &SMatrix<f64, N, 3>,
    f0: &Matrix3xX<f64>,
    ipar: &Vec<usize>,
    isplit: &Vec<isize>,
    ichild: &Vec<isize>,
    z: &Matrix2xX<f64>,
    f: &Matrix2xX<f64>,
    l: &[usize; N],
    L: &[usize; N],
) -> (
    [usize; N],  // n0
    [f64; N],    // x
    [f64; N],    // y
    [f64; N],    // x1
    [f64; N],    // x2
    [f64; N],    // f1
    [f64; N],    // f2
) {
    let mut x = [f64::INFINITY; N];
    let mut y = [f64::INFINITY; N];
    let mut x1 = [f64::INFINITY; N];
    let mut x2 = [f64::INFINITY; N];
    let mut f1 = [0.0; N];
    let mut f2 = [0.0; N];
    let mut n0 = [0_usize; N];

    let mut fold = f[(0, j)];
    let mut m = j;

    while m > 0 {
        let split_val = isplit[ipar[m]];
        let i = if split_val < 0 {  // TODO: why here int()?
            match split_val.abs() as usize {
                0 => N - 1,
                val => val - 1,
            }
        } else {
            split_val.abs() as usize
        };

        n0[i] += 1;

        if ichild[m] == 1 {
            if x[i] == f64::INFINITY || x[i] == z[(0, ipar[m])] {
                x[i] = vert1(1, z.column(ipar[m]), f.column(ipar[m]), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            } else {
                fold = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f[(0, ipar[m])]);
                vert2(0, x[i], z.column(ipar[m]), f.column(ipar[m]), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            }
        } else if ichild[m] >= 2 {
            fold = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f[(0, ipar[m])]);
            if x[i] == f64::INFINITY || x[i] == z[(1, ipar[m])] {
                x[i] = vert1(0, z.column(ipar[m]), f.column(ipar[m]), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            } else {
                vert2(1, x[i], z.column(ipar[m]), f.column(ipar[m]), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            }
        };

        if ichild[m] >= 1 && ichild[m] <= 2 && y[i] == f64::INFINITY {
            y[i] = split1(
                z[(0, ipar[m])],
                z[(1, ipar[m])],
                f[(0, ipar[m])],
                f[(1, ipar[m])],
            );
        };

        if ichild[m] < 0 {
            let (mut j1, mut j2, j3) = if u[i] < x0[(i, 0)] {
                let j1 = (ichild[m].abs() as f64 / 2.0).ceil() as usize;
                let j2 = (ichild[m].abs() as f64 / 2.0).floor() as usize;
                let j3 = if ((ichild[m].abs() as f64 / 2.0) < (j1 as f64) && j1 > 0) || (j1 == L[i] + 1) {
                    J3::MinusOne
                } else {
                    J3::One
                };
                (j1, j2, j3)
            } else {
                let j1 = (ichild[m].abs() as f64 / 2.0).floor() as usize + 1;
                let j2 = (ichild[m].abs() as f64 / 2.0).ceil() as usize;
                let j3 = if (ichild[m].abs() as f64 / 2.0 + 1.0 > j1 as f64) || (j1 < (L[i] + 1)) {
                    J3::One
                } else {
                    J3::MinusOne
                };
                (j1, j2, j3)
            };

            // j1 -= 1
            j1 = match j1 { // j1 is index of array len 3
                0 => 2,
                j1 => j1 - 1
            };

            // j2 -= 1
            j2 = match j2 { // j2 is index of array len 3
                0 => 2,
                j2 => j2 - 1
            };

            let (j1_plus_j3, j1_plus_2_j3, j1_minus_j3) = match j3 {
                J3::One => (
                    j1 + 1, j1 + 2, match j1 {
                        0 => 2,
                        j1 => j1 - 1
                    }),
                J3::MinusOne => {
                    match j1 {
                        0 => (2, 1, 1),
                        1 => (0, 2, 2),
                        j1 => (j1 - 1, j1 - 2, j1 + 1)
                    }
                }
            };


            let k = if isplit[ipar[m]] < 0 {
                i
            } else {
                z[(0, ipar[m])] as usize
            };

            if j1 != l[i] || (x[i] != f64::INFINITY && x[i] != x0[(i, l[i])]) {
                fold = updtf(
                    i,
                    &x1,
                    &x2,
                    &mut f1,
                    &mut f2,
                    fold,
                    f0[(l[i], k)],
                );
            }

            if x[i] == f64::INFINITY || x[i] == x0[(i, j1)] {
                x[i] = x0[(i, j1)];
                if x1[i] == f64::INFINITY {
                    vert3(
                        j1,
                        x0.row(i),
                        f0.column(k),
                        L[i],
                        &mut x1[i],
                        &mut x2[i],
                        &mut f1[i],
                        &mut f2[i],
                    );
                } else if x2[i] == f64::INFINITY && x1[i] != x0[(i, j1_plus_j3)] {
                    x2[i] = x0[(i, j1_plus_j3)];
                    f2[i] += f0[(j1_plus_j3, k)];
                } else if x2[i] == f64::INFINITY {
                    if j1 != 1 && j1 != L[i] {
                        x2[i] = x0[(i, j1_minus_j3)];
                        f2[i] += f0[(j1_minus_j3, k)];
                    } else {
                        x2[i] = x0[(i, j1_plus_2_j3)];
                        f2[i] += f0[(j1_plus_2_j3, k)];
                    }
                }
            } else {
                if x1[i] == f64::INFINITY {
                    x1[i] = x0[(i, j1)];
                    f1[i] += f0[(j1, k)];
                    if x[i] != x0[(i, j1_plus_j3)] {
                        x2[i] = x0[(i, j1_plus_j3)];
                        f2[i] += f0[(j1_plus_j3, k)];
                    }
                } else if x2[i] == f64::INFINITY {
                    if x1[i] != x0[(i, j1)] {
                        x2[i] = x0[(i, j1)];
                        f2[i] += f0[(j1, k)];
                    } else if x[i] != x0[(i, j1_plus_j3)] {
                        x2[i] = x0[(i, j1_plus_j3)];
                        f2[i] += f0[(j1_plus_j3, k)];
                    } else {
                        if j1 != 1 && j1 != L[i] {
                            x2[i] = x0[(i, j1_plus_j3)];
                            f2[i] += f0[(j1_plus_j3, k)];
                        } else {
                            x2[i] = x0[(i, j1_plus_2_j3)];
                            f2[i] += f0[(j1_plus_2_j3, k)];
                        }
                    }
                }
            }

            if y[i] == f64::INFINITY {
                if j2 == 2 {
                    y[i] = u[i];
                } else if j2 == L[i] {
                    y[i] = v[i];
                } else {
                    y[i] = split1(
                        x0[(i, j2)],
                        x0[(i, j2 + 1)],
                        f0[(j2, k)],
                        f0[(j2 + 1, k)],
                    );
                }
            }
        }
        m = ipar[m];
    }

    for i in 0..N {
        if x[i] == f64::INFINITY {
            x[i] = x0[(i, l[i])];
            vert3(l[i], x0.row(i), f0.column(i), L[i], &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
        }

        if y[i] == f64::INFINITY {
            y[i] = v1[i];
        }
    }

    (n0, x, y, x1, x2, f1, f2)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let j: usize = 0;
        let u = [0.0];
        let v = [1.0];
        let v1 = [1.0];
        let x0 = SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 0.5, 1.0]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[
            -0.5, -0.8, -0.3,
            -0.6, -0.9, -0.4,
            -0.7, -1.0, -0.5,
        ]);
        let ipar = vec![0];
        let isplit = vec![-1];
        let ichild = vec![1];
        let z = Matrix2xX::<f64>::from_row_slice(&[0.0, f64::INFINITY]);
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, 0.0]);
        let l = [1];
        let L = [2];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &l, &L);

        assert_eq!(n0, [0]);
        assert_eq!(x, [0.5]);
        assert_eq!(y, [1.0]);
        assert_eq!(x1, [0.0]);
        assert_eq!(x2, [1.0]);
        assert_eq!(f1, [-0.5]);
        assert_eq!(f2, [-0.7]);
    }

    #[test]
    fn test_1() {
        let j: usize = 0;
        let u = [0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0];
        let v1 = [1.0, 1.0, 1.0];
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ]);
        let f0 = Matrix3xX::<f64>::zeros(3);
        let ipar = vec![1, 2, 3];
        let isplit = vec![3, 2, 1];
        let ichild = vec![1, 2, 1];
        let z = Matrix2xX::<f64>::from_row_slice(&[f64::INFINITY; 6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[0.0; 6]);
        let l = [1, 1, 1];
        let L = [2, 2, 2];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &l, &L);

        assert_eq!(n0, [0, 0, 0]);
        assert_eq!(x, [0.5, 0.5, 0.5]);
        assert_eq!(y, [1.0, 1.0, 1.0]);
        assert_eq!(x1, [0.0, 0.0, 0.0]);
        assert_eq!(x2, [1.0, 1.0, 1.0]);
        assert_eq!(f1, [0.0, 0.0, 0.0]);
        assert_eq!(f2, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_2() {
        let j: usize = 0;
        let u = [0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0];
        let v1 = [-1.0, -1.0, -1.0];
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -0.0, -0.5, -1.0,
            -0.0, -0.5, -1.0,
            -0.0, -0.5, -1.0,
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![3, 2, 3];
        let isplit = vec![1, 1, 1];
        let ichild = vec![1, 2, 1];
        let z = Matrix2xX::<f64>::from_row_slice(&[f64::INFINITY; 6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let l = [1, 1, 1];
        let L = [2, 2, 2];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &l, &L);

        assert_eq!(n0, [0, 0, 0]);
        assert_eq!(x, [-0.5, -0.5, -0.5]);
        assert_eq!(y, [-1.0, -1.0, -1.0]);
        assert_eq!(x1, [-0.0, -0.0, -0.0]);
        assert_eq!(x2, [-1.0, -1.0, -1.0]);
        assert_eq!(f1, [1.0, 1.0, 1.0]);
        assert_eq!(f2, [1.0, 1.0, 1.0]);
    }


    #[test]
    fn test_3() {
        let j: usize = 2;
        let u = [0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0];
        let v1 = [-1.0, -2.0, -3.0];
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -0.0, -0.5, -1.0,
            -0.0, -0.5, -1.0,
            -0.0, -0.5, -1.0,
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![2, 1, 0];
        let isplit = vec![-1, -2, -3];
        let ichild = vec![-3, -1, -1];
        let z = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let l = [2, 0, 1];
        let L = [1, 1, 2];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &l, &L);

        assert_eq!(n0, [1, 0, 0]);
        assert_eq!(x, [-0., -0., -0.5]);
        assert_eq!(y, [-0.30901699437494745, -2., -3.]);
        assert_eq!(x1, [-0.5, -0.5, -0.]);
        assert_eq!(x2, [-1., -1., -1.]);
        assert_eq!(f1, [1., 1., 1.]);
        assert_eq!(f2, [1., 1., 1.]);
    }

    #[test]
    fn test_4() {
        let j: usize = 1;
        let u = [0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0];
        let v1 = [3.0, 1.0, -3.0];
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            3.0, 1.5, 1.0,
            1.0, 0.5, -1.0,
            -1.0, -1.5, 2.0
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![2, 2, 0];
        let isplit = vec![-1, -2, -3];
        let ichild = vec![-3, 1, -1];
        let z = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let l = [0, 1, 1];
        let L = [2, 2, 3];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &l, &L);

        assert_eq!(n0, [1, 0, 1]);
        assert_eq!(x, [3., 0.5, 1.]);
        assert_eq!(y, [0., 1., 1.]);
        assert_eq!(x1, [1.5, 1., 1.]);
        assert_eq!(x2, [1., -1., f64::INFINITY]);
        assert_eq!(f1, [1., 1., 1.]);
        assert_eq!(f2, [1., 1., 0.]);
    }


    #[test]
    fn test_5() {
        let j: usize = 1;
        let u = [0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0];
        let v1 = [3.0, 1.0, -3.0];
        let x0 = SMatrix::<f64, 3, 3>::from_row_slice(&[
            3.0, 1.5, 1.0,
            1.0, 0.5, -1.0,
            -1.0, -1.5, 2.0
        ]);
        let f0 = Matrix3xX::<f64>::from_row_slice(&[1.; 9]); // 3x3 matrix of ones
        let ipar = vec![2, 2, 0];
        let isplit = vec![0, 0, 0];
        let ichild = vec![1, 1, 1];
        let z = Matrix2xX::<f64>::from_row_slice(&[1., 2., 3., 5., 6., 7.]);
        let f = Matrix2xX::<f64>::from_row_slice(&[1.0; 6]);
        let l = [0, 1, 1];
        let L = [2, 2, 3];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, &u, &v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &l, &L);

        assert_eq!(n0, [2, 0, 0]);
        assert_eq!(x, [3., 0.5, -1.5]);
        assert_eq!(y, [5.47213595499958, 1., -3.]);
        assert_eq!(x1, [7., 1., -1.]);
        assert_eq!(x2, [1., -1., 2.]);
        assert_eq!(f1, [1., 1., 1.]);
        assert_eq!(f2, [1., 1., 1.]);
    }
}