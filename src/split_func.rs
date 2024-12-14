use crate::add_basket::add_basket;
use crate::chk_flag::update_flag;
use crate::feval::feval;
use crate::sign::sign;
use crate::updtrec::updtrec;
use nalgebra::SMatrix;


const SQRT_5: f64 = 2.2360679774997896964091736687312;


fn genbox<const SMAX: usize>(
    nboxes: &mut usize,
    ipar: &mut Vec<Option<usize>>,
    level: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    f: &mut [Vec<f64>; 2],
    ipar_upd: usize,
    level_upd: usize,
    ichild_upd: isize,
    f_upd: f64,
    record: &mut [usize; SMAX],
) {
    *nboxes += 1;

    ipar[*nboxes] = Some(ipar_upd);
    level[*nboxes] = level_upd;
    ichild[*nboxes] = ichild_upd;
    f[0][*nboxes] = f_upd;

    updtrec(*nboxes, level[*nboxes], &f[0], record);
}


pub fn splinit<const N: usize, const SMAX: usize>(
    i: usize,
    s: usize,
    smax: usize,
    par: usize,
    x0: &SMatrix::<f64, N, 3>,
    u: &[f64; N],
    v: &[f64; N],
    x: &mut [f64; N],
    xmin: &mut Vec<[f64; N]>,
    fmi: &mut Vec<f64>,
    ipar: &mut Vec<Option<usize>>,
    level: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    f: &mut [Vec<f64>; 2],
    xbest: &mut [f64; N],
    fbest: &mut f64,
    stop: &[f64],
    record: &mut [usize; SMAX],
    nboxes: &mut usize,
    nbasket_option: &mut Option<usize>,
    nsweepbest: &mut usize,
    nsweep: &mut usize,
) -> (
    [f64; 3], // f0
    bool,     // flag
    usize     // ncall
) {
    let mut ncall: usize = 0;
    let mut f0 = [0.0; 3]; // L.iter().max().unwrap() + 1 always 3
    let mut flag = true;

    for j in 0..3 {
        if j != 1 {
            x[i] = x0[(i, j)];
            f0[j] = feval(x);
            ncall += 1;
            if f0[j] < *fbest {
                *fbest = f0[j];
                *xbest = x.clone();
                *nsweepbest = *nsweep;
                update_flag(&mut flag, stop, *fbest, 2);
                if !flag {
                    return (f0, flag, ncall);
                }
            }
        } else { f0[j] = f[0][par] }
    }

    if s + 1 < smax {
        let mut nchild: usize = 0;

        if u[i] < x0[(i, 0)] {
            nchild += 1;
            genbox(
                nboxes, ipar, level, ichild, f,
                par, s + 1, -(nchild as isize), f0[0], record,
            );
        };
        for j in 0..2 {
            nchild += 1;
            if (f0[j] <= f0[j + 1]) || (s + 2 < smax) {
                let level0 = if f0[j] <= f0[j + 1] { s + 1 } else { s + 2 };
                genbox(
                    nboxes, ipar, level, ichild, f,
                    par, level0, -(nchild as isize), f0[j], record,
                );
            } else {
                x[i] = x0[(i, j)];
                add_basket(nbasket_option, xmin, fmi, x.clone(), f0[j]);
            }
            nchild += 1;
            if (f0[j + 1] < f0[j]) || (s + 2 < smax) {
                let level0 = if f0[j + 1] < f0[j] { s + 1 } else { s + 2 };
                genbox(
                    nboxes, ipar, level, ichild, f,
                    par, level0, -(nchild as isize), f0[j + 1], record,
                );
            } else {
                x[i] = x0[(i, j + 1)];
                add_basket(nbasket_option, xmin, fmi, x.clone(), f0[j + 1]);
            }
        }

        if x0[(i, 2)] < v[i] {
            nchild += 1;
            genbox(
                nboxes, ipar, level, ichild, f,
                par, s + 1, -(nchild as isize), f0[2], record,
            );
        }
    } else {
        for j in 0..3 {
            x[i] = x0[(i, j)];
            add_basket(nbasket_option, xmin, fmi, x.clone(), f0[j]);
        }
    }
    (f0, flag, ncall)
}


pub fn split<const N: usize, const SMAX: usize>(
    i: usize,
    s: usize,
    smax: usize,
    par: usize,
    x: &mut [f64; N],
    y: &mut [f64; N],
    z: &Vec<f64>,
    xmin: &mut Vec<[f64; N]>,
    fmi: &mut Vec<f64>,
    ipar: &mut Vec<Option<usize>>,
    level: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    f: &mut [Vec<f64>; 2],
    xbest: &mut [f64; N],
    fbest: &mut f64,
    stop: &[f64],
    record: &mut [usize; SMAX],
    nboxes: &mut usize,
    nbasket_option: &mut Option<usize>,
    nsweepbest: &mut usize,
    nsweep: &mut usize,
) -> (
    bool,  // flag
    usize  // ncall
) {
    let mut ncall: usize = 0;
    let mut flag = true;
    x[i] = z[1];
    f[1][par] = feval(x);
    ncall += 1;

    if f[1][par] < *fbest {
        *fbest = f[1][par];
        *xbest = x.clone();
        *nsweepbest = *nsweep;
        update_flag(&mut flag, stop, *fbest, 2);
        if !flag {
            return (flag, ncall);
        }
    }

    if s + 1 < smax {
        if f[0][par] <= f[1][par] {
            genbox(
                nboxes, ipar, level, ichild, f,
                par, s + 1, 1, f[0][par], record,
            );
            if s + 2 < smax {
                genbox(
                    nboxes, ipar, level, ichild, f,
                    par, s + 2, 2, f[1][par], record,
                );
            } else {
                x[i] = z[1];
                add_basket(nbasket_option, xmin, fmi, x.clone(), f[1][par]);
            }
        } else {
            if s + 2 < smax {
                genbox(
                    nboxes, ipar, level, ichild, f,
                    par, s + 2, 1, f[0][par], record,
                );
            } else {
                x[i] = z[0];
                add_basket(nbasket_option, xmin, fmi, x.clone(), f[0][par]);
            }

            genbox(
                nboxes, ipar, level, ichild, f,
                par, s + 1, 2, f[1][par], record,
            );
        }

        if z[1] != y[i] {
            if (z[1] - y[i]).abs() > (z[1] - z[0]).abs() * (3.0 - SQRT_5) * 0.5 {
                genbox(
                    nboxes, ipar, level, ichild, f,
                    par, s + 1, 3, f[1][par], record,
                );
            } else {
                if s + 2 < smax {
                    genbox(
                        nboxes, ipar, level, ichild, f,
                        par, s + 2, 3, f[1][par], record,
                    );
                } else {
                    x[i] = z[1];
                    add_basket(nbasket_option, xmin, fmi, x.clone(), f[1][par]);
                }
            }
        }
    } else {
        let mut xi1 = x.clone();
        let mut xi2 = x.clone();

        xi1[i] = z[0];
        add_basket(nbasket_option, xmin, fmi, xi1, f[0][par]);

        xi2[i] = z[1];
        add_basket(nbasket_option, xmin, fmi, xi2, f[1][par]);
    }
    (flag, ncall)
}


pub fn split1(x1: f64, x2: f64, f1: f64, f2: f64) -> f64 {
    if f1 <= f2 {
        x1 + 0.5 * (-1.0 + SQRT_5) * (x2 - x1)
    } else {
        x1 + 0.5 * (3.0 - SQRT_5) * (x2 - x1)
    }
}

pub fn split2(x: f64, y: f64) -> f64 {
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

    #[test]
    fn test_splinit_0() {
        let i = 1_usize;
        let s = 2_usize;
        let smax = 3_usize;
        let par = 4_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[0.0; 18]);
        let u = [0.0; 6];
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x = [0.0; 6];
        let mut xmin = vec![[0.0; 6]];
        let mut fmi = vec![0.0];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![0; 10];
        let mut f: [Vec<f64>; 2] = [vec![0.0; 10], vec![0.0; 10]];
        let mut xbest = [0.0; 6];
        let mut fbest = 10.0;
        let stop = [20., f64::NEG_INFINITY];
        let mut record = [0; 10];
        let mut nboxes = 1_usize;
        let mut nbasket = Some(0);
        let mut nsweepbest = 0_usize;
        let mut nsweep = 1_usize;

        // Call splinit
        let (f0, flag, ncall) =
            splinit(i, s, smax, par, &x0, &u, &v, &mut x, &mut xmin,
                    &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest,
                    &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket,
                    &mut nsweepbest, &mut nsweep,
            );

        assert_eq!(xbest, [0.0; 6]);
        assert_eq!(fbest, -0.00508911288366444);
        assert_eq!(f0, [-0.00508911288366444, 0.0, -0.00508911288366444]);
        assert_eq!(xmin, [[0.0; 6]; 4]);
        assert_eq!(fmi, [0.0, -0.00508911288366444, 0.0, -0.00508911288366444]);
        assert_eq!(ipar, [Some(0); 10]);
        assert_eq!(level, [0; 10]);
        assert_eq!(ichild, [0; 10]);
        assert_eq!(f, [[0.0; 10]; 2]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 2);
        assert_eq!(record, [0; 10]);
        assert_eq!(nboxes, 1);
        assert_eq!(nbasket, Some(3));
        assert_eq!(nsweepbest, 1);
        assert_eq!(nsweep, 1);
    }

    #[test]
    fn test_splinit_1() {
        let i = 1_usize;
        let s = 1_usize;
        let smax = 2_usize;
        let par = 3_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[0.0; 18]);
        let u = [0.0; 6];
        let v = [1.0; 6];
        let mut x = [0.0; 6];
        let mut xmin = vec![[1.0; 6], [1.0; 6], [1.0; 6]];
        let mut fmi = vec![10.0, 10.0, 10.0];
        let mut ipar = vec![Some(1); 10];
        let mut level = vec![1; 10];
        let mut ichild = vec![1; 10];
        let mut f: [Vec<f64>; 2] = [vec![1.0; 10], vec![1.0; 10]];
        let mut xbest = [1.0; 6];
        let mut fbest = 0.0;
        let stop = [10., f64::NEG_INFINITY];
        let mut record = [1; 10];
        let mut nboxes = 2_usize;
        let mut nbasket = Some(2_usize);
        let mut nsweepbest = 4_usize;
        let mut nsweep = 5_usize;

        let (f0, flag, ncall) = splinit(
            i, s, smax, par, &x0, &u, &v, &mut x, &mut xmin,
            &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest,
            &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket,
            &mut nsweepbest, &mut nsweep,
        );

        assert_eq!(xbest, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(fbest, -0.00508911288366444);
        assert_eq!(f0, [-0.00508911288366444, 1.0, -0.00508911288366444]);
        assert_eq!(xmin, [[1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]);
        assert_eq!(fmi, [10.0, 10., 10., -0.00508911288366444, 1.0, -0.00508911288366444]);
        assert_eq!(ipar, [Some(1); 10]);
        assert_eq!(level, [1; 10]);
        assert_eq!(ichild, [1; 10]);
        assert_eq!(f, [[1.0; 10]; 2]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 2);
        assert_eq!(record, [1; 10]);
        assert_eq!(nboxes, 2);
        assert_eq!(nbasket, Some(5));
        assert_eq!(nsweepbest, 5);
        assert_eq!(nsweep, 5);
    }

    #[test]
    fn test_splinit_2() {
        let i = 1_usize;
        let s = 2_usize;
        let smax = 10_usize;
        let par = 4_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[1.0; 18]);
        let u = [0.0; 6];
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x = [5.0; 6];
        let mut xmin = vec![[1.0; 6]];
        let mut fmi = vec![0.0];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![1; 10];
        let mut f: [Vec<f64>; 2] = [vec![0.0; 10], vec![0.0; 10]];
        let mut xbest = [0.0; 6];
        let mut fbest = 0.0;
        let stop = [1.0, 2.0, f64::NEG_INFINITY];
        let mut record = [1; 10];
        let mut nboxes = 1_usize;
        let mut nbasket = None;
        let mut nsweepbest = 1_usize;
        let mut nsweep = 2_usize;

        let (f0, flag, ncall) =
            splinit(i, s, smax, par, &x0, &u, &v, &mut x, &mut xmin,
                    &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest,
                    &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket,
                    &mut nsweepbest, &mut nsweep,
            );

        assert_eq!(xbest, [5., 1.0, 5., 5., 5., 5.]);
        assert_eq!(fbest, -1.234805298277e-312);
        assert_eq!(f0, [-1.234805298277e-312, 0.0, -1.234805298277e-312]);
        assert_eq!(xmin, [[1., 1., 1., 1., 1., 1.]]);
        assert_eq!(fmi, [0.0]);
        assert_eq!(ipar, [Some(0), Some(0), Some(4), Some(4), Some(4), Some(4), Some(4), Some(4), Some(0), Some(0)]);
        assert_eq!(level, [0, 0, 3, 3, 4, 4, 3, 3, 0, 0]);
        assert_eq!(ichild, [1, 1, -1, -2, -3, -4, -5, -6, 1, 1]);
        assert_eq!(f, [[0.0, 0.0, -1.234805298277e-312, -1.234805298277e-312, 0.0, 0.0, -1.234805298277e-312, -1.234805298277e-312, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 2);
        assert_eq!(record, [1, 1, 1, 2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(nboxes, 7);
        assert_eq!(nbasket, None);
        assert_eq!(nsweepbest, 2);
        assert_eq!(nsweep, 2);
    }

    #[test]
    fn test_splinit_3() {
        let i = 2_usize;
        let s = 3_usize;
        let smax = 5_usize;
        let par = 4_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[1.0; 18]);
        let u = [0.0; 6];
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x = [5.0; 6];
        let mut xmin = vec![[1.0; 6]];
        let mut fmi = vec![0.0];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![1; 10];
        let mut f: [Vec<f64>; 2] = [vec![0.0; 10], vec![0.0; 10]];
        let mut xbest = [0.0; 6];
        let mut fbest = 10.0;
        let stop = [30., f64::NEG_INFINITY];
        let mut record = [1; 10];
        let mut nboxes = 1_usize;
        let mut nbasket = Some(0);
        let mut nsweepbest = 1_usize;
        let mut nsweep = 2_usize;

        let (f0, flag, ncall) =
            splinit(i, s, smax, par, &x0, &u, &v, &mut x, &mut xmin,
                    &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest,
                    &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket,
                    &mut nsweepbest, &mut nsweep,
            );

        assert_eq!(xbest, [5., 5., 1., 5., 5., 5.]);
        assert_eq!(fbest, -8.440757176906739e-254);
        assert_eq!(f0, [-8.440757176906739e-254, 0.0, -8.440757176906739e-254]);
        assert_eq!(xmin, [[1., 1., 1., 1., 1., 1.], [5., 5., 1.0, 5., 5., 5.], [5., 5., 1.0, 5., 5., 5.]]);
        assert_eq!(fmi, [0.0, 0.0, 0.0]);
        assert_eq!(ipar, [Some(0), Some(0), Some(4), Some(4), Some(4), Some(4), Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(level, [0, 0, 4, 4, 4, 4, 0, 0, 0, 0]);
        assert_eq!(ichild, [1, 1, -1, -2, -5, -6, 1, 1, 1, 1]);
        assert_eq!(f, [[0.0, 0.0, -8.440757176906739e-254, -8.440757176906739e-254, -8.440757176906739e-254, -8.440757176906739e-254, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 2);
        assert_eq!(record, [1, 1, 1, 1, 2, 1, 1, 1, 1, 1]);
        assert_eq!(nboxes, 5);
        assert_eq!(nbasket, Some(2));
        assert_eq!(nsweepbest, 2);
        assert_eq!(nsweep, 2);
    }

    #[test]
    fn test_splinit_4() {
        let i = 2_usize;
        let s = 3_usize;
        let smax = 5_usize;
        let par = 4_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[1.0; 18]);
        let u = [0.0; 6];
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x = [5.0; 6];
        let mut xmin = vec![[1.0; 6], [1.0; 6], [1.0; 6]];
        let mut fmi = vec![0.0, 0.0, 0.0];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![1; 10];
        let mut f: [Vec<f64>; 2] = [vec![0.0; 10], vec![0.0; 10]];
        let mut xbest = [0.0; 6];
        let mut fbest = 10.0;
        let stop = [18., f64::NEG_INFINITY];
        let mut record = [1; 10];
        let mut nboxes = 1_usize;
        let mut nbasket = None;
        let mut nsweepbest = 3_usize;
        let mut nsweep = 2_usize;

        let (f0, flag, ncall) =
            splinit(i, s, smax, par, &x0, &u, &v, &mut x, &mut xmin,
                    &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest,
                    &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket,
                    &mut nsweepbest, &mut nsweep,
            );

        assert_eq!(xbest, [5., 5., 1., 5., 5., 5.]);
        assert_eq!(fbest, -8.440757176906739e-254);
        assert_eq!(f0, [-8.440757176906739e-254, 0.0, -8.440757176906739e-254]);
        assert_eq!(xmin, [[5., 5., 1.0, 5., 5., 5.], [5., 5., 1.0, 5., 5., 5.], [1., 1., 1., 1., 1., 1.]]);
        assert_eq!(fmi, [0.0, 0.0, 0.0]);
        assert_eq!(ipar, [Some(0), Some(0), Some(4), Some(4), Some(4), Some(4), Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(level, [0, 0, 4, 4, 4, 4, 0, 0, 0, 0]);
        assert_eq!(ichild, [1, 1, -1, -2, -5, -6, 1, 1, 1, 1]);
        assert_eq!(f, [[0.0, 0.0, -8.440757176906739e-254, -8.440757176906739e-254, -8.440757176906739e-254, -8.440757176906739e-254, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 2);
        assert_eq!(record, [1, 1, 1, 1, 2, 1, 1, 1, 1, 1]);
        assert_eq!(nboxes, 5);
        assert_eq!(nbasket, Some(1));
        assert_eq!(nsweepbest, 2);
        assert_eq!(nsweep, 2);
    }

    #[test]
    fn test_split_0() {
        let i = 0_usize;
        let s = 1_usize;
        let smax = 1_usize;
        let par = 4_usize;
        let mut x = [1., 2., 3., 4., 5., 6.];
        let mut y = [10., 20., 30., 40., 50., 60.];
        let z = vec![100., 1., 0., 0., 0., 0.];
        let mut xmin = vec![[-10., -20., -30., -40., -50., -60.], [-11., -21., -31., -41., -51., -61.]];
        let mut fmi = vec![0., 1., 2., 3., 4., 5.];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![-1; 10];
        let mut f: [Vec<f64>; 2] = [vec![1.0; 10], vec![1.0; 10]];
        let mut xbest = [0.0; 6];
        let mut fbest = 10.0;
        let stop = [0.0, 0.0, f64::NEG_INFINITY];
        let mut record = [1, 2, 3, 4, 5, 6, 7];
        let mut nboxes = 0_usize;
        let mut nbasket = Some(1);
        let mut nsweepbest = 0_usize;
        let mut nsweep = 1_usize;

        let (flag, ncall) = split(
            i, s, smax, par, &mut x, &mut y, &z, &mut xmin, &mut fmi, &mut ipar, &mut level,
            &mut ichild, &mut f, &mut xbest, &mut fbest, &stop, &mut record, &mut nboxes,
            &mut nbasket, &mut nsweepbest, &mut nsweep);

        assert_eq!(x, [1., 2., 3., 4., 5., 6.]);
        assert_eq!(y, [10., 20., 30., 40., 50., 60.]);
        assert_eq!(xmin, [[-10., -20., -30., -40., -50., -60.], [-11., -21., -31., -41., -51., -61.], [100., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.]]);
        assert_eq!(fmi, [0., 1., 2., 3., 4., 5., 1.0, -3.391970967769076e-191]);
        assert_eq!(ipar, [Some(0); 10]);
        assert_eq!(level, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(ichild, [-1; 10]);
        assert_eq!(f, [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, -3.391970967769076e-191, 1.0, 1.0, 1.0, 1.0, 1.0]]);
        assert_eq!(xbest, [1., 2., 3., 4., 5., 6.]);
        assert_eq!(fbest, -3.391970967769076e-191);
        assert_eq!(record, [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(nboxes, 0);
        assert_eq!(nbasket, Some(3));
        assert_eq!(nsweepbest, 1);
        assert_eq!(nsweep, 1);
        assert_eq!(flag, true);
        assert_eq!(ncall, 1);
    }

    #[test]
    fn test_split_1() {
        let i = 2_usize;
        let s = 0_usize;
        let smax = 3_usize;
        let par = 4_usize;
        let mut x = [1., 2., 3., 4., 5., 6.];
        let mut y = [10., 20., 30., 40., 50., 60.];
        let z = vec![100., 1., 0., 0., 0., 0.];
        let mut xmin = vec![[-10., -20., -30., -40., -50., -60.], [-11., -21., -31., -41., -51., -61.]];
        let mut fmi = vec![0., 1., 2., 3., 4., 5.];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![-1; 10];
        let mut f: [Vec<f64>; 2] = [vec![1.0; 10], vec![1.0; 10]];
        let mut xbest = [0.0; 6];
        let mut fbest = 10.0;
        let stop = [0.0, 0.0, f64::NEG_INFINITY];
        let mut record = [1, 2, 3, 4, 5, 6, 7];
        let mut nboxes = 0_usize;
        let mut nbasket = None;
        let mut nsweepbest = 0_usize;
        let mut nsweep = 1_usize;

        let (flag, ncall) = split(
            i, s, smax, par, &mut x, &mut y, &z, &mut xmin, &mut fmi, &mut ipar, &mut level,
            &mut ichild, &mut f, &mut xbest, &mut fbest, &stop, &mut record, &mut nboxes,
            &mut nbasket, &mut nsweepbest, &mut nsweep);

        assert_eq!(x, [1., 2., 1., 4., 5., 6.]);
        assert_eq!(y, [10., 20., 30., 40., 50., 60.]);
        assert_eq!(xmin, [[-10., -20., -30., -40., -50., -60.], [-11., -21., -31., -41., -51., -61.]]);
        assert_eq!(fmi, [0., 1., 2., 3., 4., 5.]);
        assert_eq!(ipar, [Some(0), Some(4), Some(4), Some(4), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(level, [0, 2, 1, 2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(ichild, [-1, 1, 2, 3, -1, -1, -1, -1, -1, -1]);
        assert_eq!(f, [[1.0, 1.0, -1.4064265983273568e-148, -1.4064265983273568e-148, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, -1.4064265983273568e-148, 1.0, 1.0, 1.0, 1.0, 1.0]]);
        assert_eq!(xbest, [1., 2., 1., 4., 5., 6.]);
        assert_eq!(fbest, -1.4064265983273568e-148);
        assert_eq!(record, [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(nboxes, 3);
        assert_eq!(nbasket, None);
        assert_eq!(nsweepbest, 1);
        assert_eq!(nsweep, 1);
        assert_eq!(flag, true);
        assert_eq!(ncall, 1);
    }

    #[test]
    fn test_split_2() {
        let i = 2_usize;
        let s = 3_usize;
        let smax = 8_usize;
        let par = 3_usize;
        let mut x = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0];
        let mut y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let z = vec![5.1, -5.2, 5.3, -5.4, 5.5, -5.6];
        let mut xmin = vec![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let mut fmi = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let mut ipar = vec![Some(1); 6];
        let mut level = vec![1; 6];
        let mut ichild = vec![1; 6];
        let mut f: [Vec<f64>; 2] = [vec![1.0; 10], vec![1.0; 10]];
        let mut xbest = [1.0; 6];
        let mut fbest = 0.0;
        let stop = [0.0, 0.0, f64::NEG_INFINITY];
        let mut record = [1, 2, 3, 4, 5, 6, 7];
        let mut nboxes = 1_usize;
        let mut nbasket = Some(0_usize);
        let mut nsweepbest = 3_usize;
        let mut nsweep = 1_usize;

        let (flag, ncall) = split(
            i, s, smax, par, &mut x, &mut y, &z, &mut xmin, &mut fmi, &mut ipar, &mut level,
            &mut ichild, &mut f, &mut xbest, &mut fbest, &stop, &mut record, &mut nboxes,
            &mut nbasket, &mut nsweepbest, &mut nsweep);

        assert_eq!(x, [10.0, 9.0, -5.2, 7.0, 6.0, 5.0]);
        assert_eq!(y, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(xmin, [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
        assert_eq!(fmi, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        assert_eq!(ipar, [Some(1), Some(1), Some(3), Some(3), Some(3), Some(1)]);
        assert_eq!(level, [1, 1, 5, 4, 4, 1]);
        assert_eq!(ichild, [1, 1, 1, 2, 3, 1]);
        assert_eq!(f, [[1.0, 1.0, 1.0, -0.0, -0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, -0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]);
        assert_eq!(xbest, [1.; 6]);
        assert_eq!(fbest, 0.0);
        assert_eq!(record, [1, 2, 3, 4, 3, 6, 7]);
        assert_eq!(nboxes, 4);
        assert_eq!(nbasket, Some(0));
        assert_eq!(nsweepbest, 3);
        assert_eq!(nsweep, 1);
        assert_eq!(flag, true);
        assert_eq!(ncall, 1);
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

