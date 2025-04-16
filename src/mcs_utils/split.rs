use crate::mcs_utils::{add_basket::add_basket, genbox::genbox, sign::sign};
use nalgebra::SVector;

const SQRT_5: f64 = 2.2360679774997896964091736687312;

pub(crate) fn split<const N: usize, const SMAX: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    i: usize, // -1 from Matlab
    s: usize, // as in Matlab
    par: usize,  // -1 from Matlab
    x: &SVector<f64, N>,
    y: &[f64; N],
    z0: f64,
    z1: f64,
    xmin: &mut Vec<SVector<f64, N>>,
    fmi: &mut Vec<f64>,
    ipar: &mut Vec<Option<usize>>,
    level: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    isplit: &mut Vec<isize>,
    nogain: &mut Vec<bool>,
    f: &mut [Vec<f64>; 2],
    z: &mut [Vec<f64>; 2],
    xbest: &mut SVector<f64, N>,
    fbest: &mut f64,
    record: &mut [Option<usize>; SMAX],
    nboxes: &mut usize,
    nbasket: &mut usize,
    nsweepbest: &mut usize,
    nsweep: &mut usize,
) {
    let mut x = x.clone();
    x[i] = z1;
    f[1][par] = func(&x);

    if f[1][par] < *fbest {
        *fbest = f[1][par];
        *xbest = x;
        *nsweepbest = *nsweep;
    }

    if s + 1 < SMAX {
        if f[0][par] <= f[1][par] {
            genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 1, 1, f[0][par], record);
            if s + 2 < SMAX {
                genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 2, 2, f[1][par], record);
            } else {
                x[i] = z1;
                add_basket(nbasket, xmin, fmi, &x, f[1][par]);
            }
        } else {
            if s + 2 < SMAX {
                genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 2, 1, f[0][par], record);
            } else {
                x[i] = z0;
                add_basket(nbasket, xmin, fmi, &x, f[0][par]);
            }

            genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 1, 2, f[1][par], record);
        }

        if z1 != y[i] {
            if (z1 - y[i]).abs() > (z1 - z0).abs() * (3.0 - SQRT_5) * 0.5 {
                genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 1, 3, f[1][par], record);
            } else {
                if s + 2 < SMAX {
                    genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 2, 3, f[1][par], record);
                } else {
                    x[i] = z1;
                    add_basket(nbasket, xmin, fmi, &x, f[1][par]);
                }
            }
        }
    } else {
        x[i] = z0;
        add_basket(nbasket, xmin, fmi, &x, f[0][par]);

        x[i] = z1;
        add_basket(nbasket, xmin, fmi, &x, f[1][par]);
    }
}


#[inline]
pub(crate) fn split1(x1: f64, x2: f64, f1: f64, f2: f64) -> f64 {
    if f1 <= f2 {
        x1 + 0.5 * (-1.0 + SQRT_5) * (x2 - x1)
    } else {
        x1 + 0.5 * (3.0 - SQRT_5) * (x2 - x1)
    }
}

#[inline]
pub(crate) fn split2(x: f64, y: f64) -> f64 {
    let mut x2 = y;
    if x == 0.0 && y.abs() > 1000.0 {
        x2 = sign(y);
    } else if x != 0.0 && y.abs() > (100.0 * x.abs()) {
        x2 = 10.0 * sign(y) * x.abs();
    }
    x + 2.0 * (x2 - x) / 3.0
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_split_1() {
        // Matlab test equivalent
        //
        // clearvars;
        // clear global;
        // n0 = []; % garbage value, do not touch
        // u = []; % garbage value, do not touch
        // v = []; % garbage value, do not touch
        // x1 = 0; % garbage value, do not touch
        // x2 = 0; % garbage value, do not touch
        // prt = 0; % garbage value, do not touch
        // nglob = 0;  % garbage value, do not touch
        // stop = [10000];  % garbage value, do not touch
        //
        // smax = 8; % length of record
        // fcn = "hm6";
        // data = "hm6";
        // i = 2; // +1 from Rust
        // s = 0;
        // par = 6; % Rust par is -1 from Matlab
        // x = [1., 2., 3., 4., 5., 6.];
        // y = [10., 20., 30., 40., 50., 60.];
        // z = [100., 1.]; % z0, z1
        // xmin = [[-10., -20., -30., -40., -50., -60.]; [-11., -21., -31., -41., -51., -61.]];
        // fmi = [0., 1., 2., 3., 4., 5.];
        // ipar = zeros(1, 10);
        // level = zeros(1, 10);
        // ichild = -ones(1, 10);
        // f = ones(2, 10);
        // xbest = zeros(1, 6);
        // fbest = 10.0;
        // global nbasket nboxes nglob nsweep nsweepbest record xglob;
        // record = [1, 2, 3, 4, 5, 6, 7, 0]; % +1 from Rust; None->0
        // nboxes = 1;
        // nbasket = 0;
        // nsweepbest = 0;
        // nsweep = 1;
        //
        // [xbest,fbest,xmin,fmi,ipar,level,ichild,f,flag,ncall] = split_(fcn,data,i,s,smax,par,n0,u,v,x,y,x1,x2,z,xmin,fmi,ipar,level,ichild,f,xbest,fbest,stop,prt);
        // format long g;
        // disp(xmin);
        // disp(fmi);
        // disp(ipar);
        // disp(level);
        // disp(ichild);
        // disp(f);
        // disp(xbest);
        // disp(fbest);
        // disp(record);
        // disp(nboxes);
        // disp(nbasket);
        // disp(nsweepbest);
        // disp(nsweep);

        let i = 1_usize;
        let s = 0_usize;
        let par = 5_usize;
        let x = SVector::<f64, 6>::from_row_slice(&[1., 2., 3., 4., 5., 6.]);
        let y = [10., 20., 30., 40., 50., 60.];
        let (z0, z1) = (100., 1.);
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[-10., -20., -30., -40., -50., -60.]), SVector::<f64, 6>::from_row_slice(&[-11., -21., -31., -41., -51., -61.])];
        let mut fmi = vec![0., 1., 2., 3., 4., 5.];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![-1; 10];
        let mut f = [vec![1.0; 10], vec![1.0; 10]];
        let mut z = [vec![1.0; 10], vec![1.0; 10]];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let mut fbest = 10.0;
        let mut record = [Some(0), Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), None];
        let mut nboxes = 1_usize;
        let mut nbasket = 0;
        let mut nsweepbest = 0_usize;
        let mut nsweep = 1_usize;

        split(hm6, i, s, par, &x, &y, z0, z1, &mut xmin, &mut fmi, &mut ipar, &mut level,
              &mut ichild, &mut vec![], &mut vec![], &mut f, &mut z, &mut xbest, &mut fbest, &mut record, &mut nboxes,
              &mut nbasket, &mut nsweepbest, &mut nsweep);

        let expected_f = [
            vec![1.0, 1.0, -9.93492055883314e-188, -9.93492055883314e-188, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, -9.93492055883314e-188, 1.0, 1.0, 1.0, 1.0]
        ];

        assert_eq!(xmin, [SVector::<f64, 6>::from_row_slice(&[-10., -20., -30., -40., -50., -60.]), SVector::<f64, 6>::from_row_slice(&[-11., -21., -31., -41., -51., -61.])]);
        assert_eq!(fmi, [0., 1., 2., 3., 4., 5.]);
        assert_eq!(ipar, [Some(0), Some(6), Some(6), Some(6), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(level, [0, 2, 1, 2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(ichild, [-1, 1, 2, 3, -1, -1, -1, -1, -1, -1]);
        assert_eq!(f, expected_f);
        assert_eq!(xbest.as_slice(), [1., 1., 3., 4., 5., 6.]);
        assert_eq!(fbest, -9.93492055883314e-188);
        assert_eq!(record, [Some(2), Some(3), Some(2), Some(3), Some(4), Some(5), Some(6), None]);
        assert_eq!(nboxes, 4);
        assert_eq!(nbasket, 0);
        assert_eq!(nsweepbest, 1);
        assert_eq!(nsweep, 1);
    }

    #[test]
    fn test_split_2() {
        // Matlab test equivalent
        //
        // clearvars;
        // clear global;
        // n0 = []; % garbage value, do not touch
        // u = []; % garbage value, do not touch
        // v = []; % garbage value, do not touch
        // x1 = 0; % garbage value, do not touch
        // x2 = 0; % garbage value, do not touch
        // prt = 0; % garbage value, do not touch
        // nglob = 0;  % garbage value, do not touch
        // stop = [10000];  % garbage value, do not touch
        //
        // smax = 7; % length of record
        // fcn = "hm6";
        // data = "hm6";
        // i = 2; // +1 from Rust
        // s = 3;
        // par = 4; % Rust par is -1 from Matlab
        // x = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0];
        // y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // z = [5.1, -5.2]; % z0, z1
        // xmin = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        // fmi = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        // ipar = ones(1, 6);
        // level = ones(1, 6);
        // ichild = ones(1, 6);
        // f = ones(2, 10);
        // xbest = ones(1, 6);
        // fbest = 0.0;
        // global nbasket nboxes nglob nsweep nsweepbest record xglob;
        // record = [2, 3, 4, 5, 6, 7, 8]; % +1 from Rust; None->0
        // nboxes = 2;
        // nbasket = 1;
        // nsweepbest = 3;
        // nsweep = 1;
        //
        // [xbest,fbest,xmin,fmi,ipar,level,ichild,f,flag,ncall] = split_(fcn,data,i,s,smax,par,n0,u,v,x,y,x1,x2,z,xmin,fmi,ipar,level,ichild,f,xbest,fbest,stop,prt);
        // format long g;
        // disp(xmin);
        // disp(fmi);
        // disp(ipar);
        // disp(level);
        // disp(ichild);
        // disp(f);
        // disp(xbest);
        // disp(fbest);
        // disp(record);
        // disp(nboxes);
        // disp(nbasket);
        // disp(nsweepbest);
        // disp(nsweep);

        let i = 1_usize;
        let s = 3_usize;
        let par = 3_usize;
        let x = SVector::<f64, 6>::from_row_slice(&[10.0, 9.0, 8.0, 7.0, 6.0, 5.0]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (z0, z1) = (5.1, -5.2);
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])];
        let mut fmi = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let mut ipar = vec![Some(1); 6];
        let mut level = vec![1; 6];
        let mut ichild = vec![1; 6];
        let mut f = [vec![1.0; 10], vec![1.0; 10]];
        let mut z = [vec![1.0; 10], vec![1.0; 10]];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut fbest = 0.0;
        let mut record = [Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)];
        let mut nboxes = 2_usize;
        let mut nbasket = 1;
        let mut nsweepbest = 3_usize;
        let mut nsweep = 1_usize;

        split(hm6, i, s, par, &x, &y, z0, z1, &mut xmin, &mut fmi, &mut ipar, &mut level,
              &mut ichild, &mut vec![], &mut vec![], &mut f, &mut z, &mut xbest, &mut fbest, &mut record, &mut nboxes,
              &mut nbasket, &mut nsweepbest, &mut nsweep);

        let expected_f = [
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ];

        assert_eq!(xmin, [SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
        assert_eq!(fmi, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        assert_eq!(ipar, [Some(1), Some(1), Some(4), Some(4), Some(4), Some(1)]);
        assert_eq!(level, [1, 1, 5, 4, 4, 1]);
        assert_eq!(ichild, [1, 1, 1, 2, 3, 1]);
        assert_eq!(f, expected_f);
        assert_eq!(xbest.as_slice(), [1.; 6]);
        assert_eq!(fbest, 0.0);
        assert_eq!(record, [Some(1), Some(2), Some(3), Some(3), Some(5), Some(6), Some(7)]);
        assert_eq!(nboxes, 5);
        assert_eq!(nbasket, 1);
        assert_eq!(nsweepbest, 3);
        assert_eq!(nsweep, 1);
    }


    #[test]
    fn split1_test_0() {
        let x1 = 0.0_f64;
        let x2 = 1.0_f64;
        let f1 = -1.0_f64;
        let f2 = -0.5_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 0.6180339887498949_f64;
        assert_eq!(result, expected);
    }

    #[test]
    fn split1_test_1() {
        let x1 = 0.0_f64;
        let x2 = 1.0_f64;
        let f1 = -0.5_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 0.3819660112501051_f64;
        assert_eq!(result, expected);
    }

    #[test]
    fn split1_test_2() {
        let x1 = 0.0_f64;
        let x2 = 0.0_f64;
        let f1 = -1.0_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 0.0_f64;
        assert_eq!(result, expected);
    }

    #[test]
    fn split1_test_3() {
        let x1 = 0.0_f64;
        let x2 = 1.0_f64;
        let f1 = -1.0_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 0.6180339887498949_f64;
        assert_eq!(result, expected);
    }

    #[test]
    fn split1_test_4() {
        let x1 = 1e-10_f64;
        let x2 = 2e-10_f64;
        let f1 = 1.0_f64;
        let f2 = 2.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 1.618033988749895e-10_f64;
        assert_eq!(result, expected);
    }

    #[test]
    fn split1_test_5() {
        let x1 = 10.0_f64;
        let x2 = 20.0_f64;
        let f1 = -1.0000001_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 16.18033988749895_f64;
        assert_eq!(result, expected);
    }

    // split2 ----------------------
    #[test]
    fn split2_test_0() {
        let x = 0.5;
        let y = 0.19;
        let result = split2(x, y);
        assert_eq!(result, 0.29333333333333333);
    }

    #[test]
    fn split2_test_1() {
        let x = 0.0;
        let y = 2000.0;
        let result = split2(x, y);
        assert_eq!(result, 0.6666666666666666);
    }

    #[test]
    fn split2_test_2() {
        let x = 0.0;
        let y = -2000.0;
        let result = split2(x, y);
        assert_eq!(result, -0.6666666666666666);
    }

    #[test]
    fn split2_test_3() {
        let x = 0.5;
        let y = f64::INFINITY;
        let result = split2(x, y);
        assert_eq!(result, 3.5);
    }

    #[test]
    fn split2_test_4() {
        let x = 0.5;
        let y = f64::NEG_INFINITY;
        let result = split2(x, y);
        assert_eq!(result, -3.1666666666666665);
    }
}

