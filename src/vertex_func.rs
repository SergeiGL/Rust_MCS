use crate::split_func::split1;
use crate::updtf::updtf;

fn vert1(
    j: usize,
    z: &[f64],
    f: &[f64],
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
    z: &[f64],
    f: &[f64],
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

fn vert3(
    j: usize,
    x0: &[f64],
    f0: &[f64],
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

fn vert3_2(
    j: isize,
    x0: &Vec<f64>,
    f0: &Vec<f64>,
    L: usize,
    x1: &mut f64,
    x2: &mut f64,
    f1: &mut f64,
    f2: &mut f64,
) {
    let (k1, k2) = if j == 0 {
        (1, 2)
    } else if j == L as isize {
        (L as isize - 2, L as isize - 1)
    } else {
        (j - 1, j + 1)
    };

    match k1 {
        n if n < 0 => {
            *x1 = x0[(n + x0.len() as isize) as usize];
            *f1 += f0[(n + f0.len() as isize) as usize];
        }
        n => {
            *x1 = x0[n as usize];
            *f1 += f0[n as usize];
        }
    }

    match k2 {
        n if n < 0 => {
            *x2 = x0[(n + x0.len() as isize) as usize];
            *f2 += f0[(n + f0.len() as isize) as usize];
        }
        n => {
            *x2 = x0[n as usize];
            *f2 += f0[n as usize];
        }
    }
}

use std::ops::{Index, IndexMut};

struct Matrix<T> {
    data: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    fn new(data: Vec<Vec<T>>) -> Self {
        Matrix { data }
    }

    // Convert positive and negative indices to valid indices
    fn index_adjust(&self, index: isize, length: isize) -> Option<usize> {
        let idx = if index < 0 {
            length + index
        } else {
            index
        };
        if idx >= 0 && idx < length {
            Some(idx as usize)
        } else {
            None
        }
    }
}

impl<T> Index<(isize, isize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (isize, isize)) -> &Self::Output {
        let (row, col) = index;
        let nrows = self.data.len() as isize;
        let ncols = if !self.data.is_empty() {
            self.data[0].len() as isize
        } else {
            0
        };

        let row_idx = self.index_adjust(row, nrows).expect("Row index out of bounds");
        let col_idx = self.index_adjust(col, ncols).expect("Column index out of bounds");

        &self.data[row_idx][col_idx]
    }
}

impl<T> IndexMut<(isize, isize)> for Matrix<T> {
    fn index_mut(&mut self, index: (isize, isize)) -> &mut Self::Output {
        let (row, col) = index;
        let nrows = self.data.len() as isize;
        let ncols = if !self.data.is_empty() {
            self.data[0].len() as isize
        } else {
            0
        };

        let row_idx = self.index_adjust(row, nrows).expect("Row index out of bounds");
        let col_idx = self.index_adjust(col, ncols).expect("Column index out of bounds");

        &mut self.data[row_idx][col_idx]
    }
}


pub fn vertex(
    j: usize,
    n: usize,
    u: &[f64],
    v: &[f64],
    v1: &[f64],
    x0: &Vec<Vec<f64>>,
    f0: &Vec<Vec<f64>>,
    ipar: &mut [usize],
    isplit: &[i64],
    ichild: &[i64],
    z: &Vec<Vec<f64>>,
    f: &mut [Vec<f64>],
    l: &[usize],
    L: &[usize],
) -> (
    Vec<f64>, // n0
    Vec<f64>, // x
    Vec<f64>, // y
    Vec<f64>, // x1
    Vec<f64>, // x2
    Vec<f64>, // f1
    Vec<f64>, // f2
) {
    // Initialize vectors
    let mut x: Vec<f64> = vec![f64::INFINITY; n];
    let mut y: Vec<f64> = vec![f64::INFINITY; n];
    let mut x1: Vec<f64> = vec![f64::INFINITY; n];
    let mut x2: Vec<f64> = vec![f64::INFINITY; n];
    let mut f1: Vec<f64> = vec![0.0; n];
    let mut f2: Vec<f64> = vec![0.0; n];
    let mut n0: Vec<f64> = vec![0.0; n];

    let mut fold = f[0][j];
    let mut m = j;

    let x0_wrapped = Matrix::new(x0.clone());
    let f0_wrapped = Matrix::new(f0.clone());
    let z_wrapped = Matrix::new(z.clone());


    while m > 0 {
        let split_val = isplit[ipar[m]];
        let i = if split_val < 0 {
            (split_val.abs() as usize) - 1
        } else {
            split_val.abs() as usize
        };

        n0[i] += 1.0;

        if ichild[m] == 1 {
            if x[i] == f64::INFINITY || x[i] == z[0][ipar[m]] {
                let new_x = vert1(1, &z.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &f.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
                x[i] = new_x;
            } else {
                updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f[0][ipar[m]]);
                fold = f[0][ipar[m]];
                vert2(0, x[i], &z.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &f.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            }
        } else if ichild[m] >= 2 {
            updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f[0][ipar[m]]);
            fold = f[0][ipar[m]];
            if x[i] == f64::INFINITY || x[i] == z[1][ipar[m]] {
                x[i] = vert1(0, &z.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &f.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            } else {
                vert2(1, x[i], &z.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &f.iter().map(|row| row[ipar[m]]).collect::<Vec<f64>>(), &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
            }
        }

        if (ichild[m] >= 1 && ichild[m] <= 2) && y[i] == f64::INFINITY {
            y[i] = split1(
                z[0][ipar[m]],
                z[1][ipar[m]],
                f[0][ipar[m]],
                f[1][ipar[m]],
            );
        }

        if ichild[m] < 0 {
            let (mut j1, mut j2, mut j3) = if u[i] < x0_wrapped[(i as isize, 0 as isize)] {
                let j1 = (ichild[m].abs() as f64 / 2.0).ceil() as isize;
                let j2 = (ichild[m].abs() as f64 / 2.0).floor() as isize;
                let j3 = if ((ichild[m].abs() as f64 / 2.0) < j1 as f64 && j1 > 0)
                    || (j1 as usize == L[i] + 1)
                {
                    -1
                } else {
                    1
                };
                (j1, j2, j3)
            } else {
                let j1 = (ichild[m].abs() as f64 / 2.0).floor() as isize + 1;
                let j2 = (ichild[m].abs() as f64 / 2.0).ceil() as isize;
                let j3 = if (ichild[m].abs() as f64 / 2.0 + 1.0 > j1 as f64)
                    || ((j1 as usize) < (L[i] + 1))
                {
                    1
                } else {
                    -1
                };
                (j1, j2, j3)
            };

            j1 -= 1;
            j2 -= 1;

            let k = if isplit[ipar[m]] < 0 {
                i as isize
            } else {
                z_wrapped[(0, ipar[m] as isize)] as isize
            };

            if j1 != l[i] as isize || (x[i] != f64::INFINITY && x[i] != x0_wrapped[(i as isize, l[i] as isize)]) {
                updtf(
                    n,
                    i,
                    &x1,
                    &x2,
                    &mut f1,
                    &mut f2,
                    fold,
                    f0_wrapped[(l[i] as isize, k)],
                );
                fold = f0_wrapped[(l[i] as isize, k)]
            }

            if x[i] == f64::INFINITY || x[i] == x0_wrapped[(i as isize, j1)] {
                x[i] = x0_wrapped[(i as isize, j1)];
                let input = match k {
                    k if k < 0 => {
                        f0_wrapped.data.iter().map(|row| row[(k + (row.len() as isize)) as usize]).collect::<Vec<f64>>()
                    }
                    k => {
                        f0_wrapped.data.iter().map(|row| row[k as usize]).collect::<Vec<f64>>()
                    }
                };
                if x1[i] == f64::INFINITY {
                    vert3_2(
                        j1,
                        &x0_wrapped.data[i],
                        &input,
                        L[i],
                        &mut x1[i],
                        &mut x2[i],
                        &mut f1[i],
                        &mut f2[i],
                    );
                } else if x2[i] == f64::INFINITY && x1[i] != x0_wrapped[(i as isize, j1 + j3)] {
                    x2[i] = x0_wrapped[(i as isize, j1 + j3)];
                    f2[i] += f0_wrapped[(j1 + j3, k)];
                } else if x2[i] == f64::INFINITY {
                    if j1 != 1 && j1 != L[i] as isize {
                        x2[i] = x0_wrapped[(i as isize, j1 - j3)];
                        f2[i] += f0_wrapped[(j1 - j3, k)];
                    } else {
                        x2[i] = x0_wrapped[(i as isize, j1 + 2 * j3)];
                        f2[i] += f0_wrapped[(j1 + 2 * j3, k)];
                    }
                }
            } else {
                if x1[i] == f64::INFINITY {
                    x1[i] = x0_wrapped[(i as isize, j1)];
                    f1[i] += f0_wrapped[(j1, k)];
                    if x[i] != x0_wrapped[(i as isize, j1 + j3)] {
                        x2[i] = x0_wrapped[(i as isize, j1 + j3)];
                        f2[i] += f0_wrapped[(j1 + j3, k)];
                    }
                } else if x2[i] == f64::INFINITY {
                    if x1[i] != x0_wrapped[(i as isize, j1)] {
                        x2[i] = x0_wrapped[(i as isize, j1)];
                        f2[i] += f0_wrapped[(j1, k)];
                    } else if x[i] != x0_wrapped[(i as isize, j1 + j3)] {
                        x2[i] = x0_wrapped[(i as isize, j1 + j3)];
                        f2[i] += f0_wrapped[(j1 + j3, k)];
                    } else {
                        if j1 != 1 && j1 != L[i] as isize {
                            x2[i] = x0_wrapped[(i as isize, j1 - j3)];
                            f2[i] += f0_wrapped[(j1 - j3, k)];
                        } else {
                            x2[i] = x0_wrapped[(i as isize, j1 + 2 * j3)];
                            f2[i] += f0_wrapped[(j1 + 2 * j3, k)];
                        }
                    }
                }
            }

            if y[i] == f64::INFINITY {
                if j2 == -1 {
                    y[i] = u[i];
                } else if j2 == L[i] as isize {
                    y[i] = v[i];
                } else {
                    y[i] = split1(
                        x0_wrapped[(i as isize, j2)],
                        x0_wrapped[(i as isize, j2 + 1)],
                        f0_wrapped[(j2, k)],
                        f0_wrapped[(j2 + 1, k)],
                    );
                }
            }
        }
        m = ipar[m];
    }

    for i in 0..n {
        if x[i] == f64::INFINITY {
            x[i] = x0[i][l[i]];
            vert3(l[i], &x0[i], &f0.iter().map(|row| row[i]).collect::<Vec<f64>>(), L[i], &mut x1[i], &mut x2[i], &mut f1[i], &mut f2[i]);
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
        let n: usize = 1;
        let u: Vec<f64> = vec![0.0];
        let v: Vec<f64> = vec![1.0];
        let v1: Vec<f64> = vec![1.0];
        let x0: Vec<Vec<f64>> = vec![vec![0.0, 0.5, 1.0]];
        let f0: Vec<Vec<f64>> = vec![
            vec![-0.5, -0.8, -0.3],
            vec![-0.6, -0.9, -0.4],
            vec![-0.7, -1.0, -0.5],
        ];
        let mut ipar: Vec<usize> = vec![0];
        let isplit: Vec<i64> = vec![-1];
        let ichild: Vec<i64> = vec![1];
        let z: Vec<Vec<f64>> = vec![vec![0.0], vec![f64::INFINITY]];
        let mut f: Vec<Vec<f64>> = vec![vec![-0.5], vec![0.0]];
        let l: Vec<usize> = vec![1];
        let L: Vec<usize> = vec![2];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, n, &u, &v, &v1, &x0, &f0, &mut ipar, &isplit, &ichild, &z, &mut f, &l, &L);

        assert_eq!(n0, vec![0.0]);
        assert_eq!(x, vec![0.5]);
        assert_eq!(y, vec![1.0]);
        assert_eq!(x1, vec![0.0]);
        assert_eq!(x2, vec![1.0]);
        assert_eq!(f1, vec![-0.5]);
        assert_eq!(f2, vec![-0.7]);
    }

    #[test]
    fn test_1() {
        let j: usize = 0;
        let n: usize = 3;
        let u: Vec<f64> = vec![0.0, 0.0, 0.0];
        let v: Vec<f64> = vec![1.0, 1.0, 1.0];
        let v1: Vec<f64> = vec![1.0, 1.0, 1.0];
        let x0: Vec<Vec<f64>> = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
        ];
        let f0: Vec<Vec<f64>> = vec![vec![0.0; 3]; 3]; // 3x3 matrix of zeros
        let mut ipar: Vec<usize> = vec![1, 2, 3];
        let isplit: Vec<i64> = vec![3, 2, 1];
        let ichild: Vec<i64> = vec![1, 2, 1];
        let z: Vec<Vec<f64>> = vec![vec![f64::INFINITY; 3], vec![f64::INFINITY; 3]];
        let mut f: Vec<Vec<f64>> = vec![vec![0.0; 3], vec![0.0; 3]];
        let l: Vec<usize> = vec![1, 1, 1];
        let L: Vec<usize> = vec![2, 2, 2];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, n, &u, &v, &v1, &x0, &f0, &mut ipar, &isplit, &ichild, &z, &mut f, &l, &L);

        assert_eq!(n0, vec![0.0, 0.0, 0.0]);
        assert_eq!(x, vec![0.5, 0.5, 0.5]);
        assert_eq!(y, vec![1.0, 1.0, 1.0]);
        assert_eq!(x1, vec![0.0, 0.0, 0.0]);
        assert_eq!(x2, vec![1.0, 1.0, 1.0]);
        assert_eq!(f1, vec![0.0, 0.0, 0.0]);
        assert_eq!(f2, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_2() {
        let j: usize = 0;
        let n: usize = 2;
        let u: Vec<f64> = vec![0.0, 0.0, 0.0];
        let v: Vec<f64> = vec![1.0, 1.0, 1.0];
        let v1: Vec<f64> = vec![-1.0, -1.0, -1.0];
        let x0: Vec<Vec<f64>> = vec![
            vec![-0.0, -0.5, -1.0],
            vec![-0.0, -0.5, -1.0],
            vec![-0.0, -0.5, -1.0],
        ];
        let f0: Vec<Vec<f64>> = vec![vec![1.0; 3]; 3]; // 3x3 matrix of ones
        let mut ipar: Vec<usize> = vec![3, 2, 3];
        let isplit: Vec<i64> = vec![1, 1, 1];
        let ichild: Vec<i64> = vec![1, 2, 1];
        let z: Vec<Vec<f64>> = vec![vec![f64::INFINITY; 3], vec![f64::INFINITY; 3]];
        let mut f: Vec<Vec<f64>> = vec![vec![1.0; 3], vec![1.0; 3]];
        let l: Vec<usize> = vec![1, 1, 1];
        let L: Vec<usize> = vec![2, 2, 2];

        let (n0, x, y, x1, x2, f1, f2) = vertex(j, n, &u, &v, &v1, &x0, &f0, &mut ipar, &isplit, &ichild, &z, &mut f, &l, &L);

        assert_eq!(n0, vec![0.0, 0.0]);
        assert_eq!(x, vec![-0.5, -0.5]);
        assert_eq!(y, vec![-1.0, -1.0]);
        assert_eq!(x1, vec![-0.0, -0.0]);
        assert_eq!(x2, vec![-1.0, -1.0]);
        assert_eq!(f1, vec![1.0, 1.0]);
        assert_eq!(f2, vec![1.0, 1.0]);
    }
}