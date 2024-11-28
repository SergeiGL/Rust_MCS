use crate::chk_flag::{chrelerr, chvtr};
use crate::feval::feval;
use crate::sign::sign;
use crate::updtrec::updtrec;

const SQRT_5: f64 = 2.23606797749978964;

pub fn splinit(i: usize, s: usize, smax: usize, par: usize,
               x0: &[&[f64]], u: &[f64], v: &[f64], x: &mut Vec<f64>, L: &[usize],
               l: &[usize], xmin: &mut Vec<Vec<f64>>, fmi: &mut Vec<f64>,
               ipar: &mut Vec<usize>, level: &mut Vec<usize>, ichild: &mut Vec<i64>,
               f: &mut Vec<Vec<f64>>, xbest: &mut Vec<f64>, fbest: &mut f64,
               stop: &[f64], record: &mut Vec<usize>, nboxes: &mut usize,
               nbasket: &mut usize, nsweepbest: &mut usize, nsweep: &mut usize,
) -> (
    Vec<f64>, //f0
    bool, //flag
    usize //ncall
) {
    let mut ncall: usize = 0;
    let mut f0 = vec![0.0; L.iter().max().unwrap() + 1];
    let mut flag = true;

    for j in 0..L[i] + 1 {
        if j != l[i] {
            x[i] = x0[i][j];
            f0[j] = feval(x);
            ncall += 1;
            if f0[j] < *fbest {
                *fbest = f0[j];
                *xbest = x.clone();
                *nsweepbest = *nsweep;
                if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                    flag = chrelerr(*fbest, stop);
                } else if stop.len() > 1 && stop[0] == 0.0 {
                    flag = chvtr(*fbest, stop[2]);
                }
                if !flag {
                    return (f0, flag, ncall);
                }
            }
        } else { f0[j] = f[0][par] }
    }

    if s + 1 < smax {
        let mut nchild: usize = 0;
        if u[i] < x0[i][0] {
            nchild += 1;
            *nboxes += 1;
            ipar[*nboxes] = par;
            level[*nboxes] = s + 1;
            ichild[*nboxes] = -(nchild as i64);
            f[0][*nboxes] = f0[0];
            updtrec(*nboxes, level[*nboxes], &f[0], record);
        };
        for j in 0..L[i] {
            nchild += 1;
            if f0[j] <= f0[j + 1] || s + 2 < smax {
                *nboxes += 1;
                let level0 = if f0[j] <= f0[j + 1] { s + 1 } else { s + 2 };
                ipar[*nboxes] = par;
                level[*nboxes] = level0;
                ichild[*nboxes] = -(nchild as i64);
                f[0][*nboxes] = f0[j];
                updtrec(*nboxes, level[*nboxes], &f[0], record);
            } else {
                x[i] = x0[i][j];
                *nbasket += 1;
                if xmin.len() == *nbasket {
                    xmin.push(x.clone());
                    fmi.push(f0[j]);
                } else {
                    xmin[*nbasket] = x.clone();
                    fmi[*nbasket] = f0[j];
                }
            }
            nchild += 1;

            if f0[j + 1] < f0[j] || s + 2 < smax {
                *nboxes += 1;
                let level0 = if f0[j + 1] < f0[j] { s + 1 } else { s + 2 };
                ipar[*nboxes] = par;
                level[*nboxes] = level0;
                ichild[*nboxes] = -(nchild as i64);
                f[0][*nboxes] = f0[j + 1];
                updtrec(*nboxes, level[*nboxes], &f[0], record);
            } else {
                x[i] = x0[i][j + 1];
                *nbasket += 1;
                if xmin.len() == *nbasket {
                    xmin.push(x.clone());
                    fmi.push(f0[j + 1]);
                } else {
                    xmin[*nbasket] = x.clone();
                    fmi[*nbasket] = f0[j + 1];
                }
            }
        }

        if x0[i][L[i]] < v[i] {
            nchild += 1;
            *nboxes += 1;

            ipar[*nboxes] = par;
            level[*nboxes] = s + 1;
            ichild[*nboxes] = -(nchild as i64);
            f[0][*nboxes] = f0[L[i]];
            updtrec(*nboxes, level[*nboxes], &f[0], record);
        }
    } else {
        for j in 0..L[i] + 1 {
            x[i] = x0[i][j];
            *nbasket += 1;
            if xmin.len() == *nbasket {
                xmin.push(x.clone());
                fmi.push(f0[j]);
            } else {
                xmin[*nbasket] = x.clone();
                fmi[*nbasket] = f0[j];
            }
        }
    }
    (f0, flag, ncall)
}

pub fn split(i: usize, s: usize, smax: usize, par: usize,
             x: &mut Vec<f64>, y: &mut Vec<f64>, z: &mut Vec<f64>,
             xmin: &mut Vec<Vec<f64>>, fmi: &mut Vec<f64>, ipar: &mut Vec<usize>, level: &mut Vec<usize>,
             ichild: &mut Vec<i64>, f: &mut Vec<Vec<f64>>, xbest: &mut Vec<f64>,
             fbest: &mut f64, stop: &[f64], record: &mut Vec<usize>, nboxes: &mut usize, nbasket: &mut usize, nsweepbest: &mut usize, nsweep: &mut usize,
) -> (
    bool, //flag
    usize //ncall
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

        if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
            flag = chrelerr(*fbest, stop);
        } else if stop.len() > 1 && stop[0] == 0.0 {
            flag = chvtr(*fbest, stop[2]);
        }
        if !flag {
            return (flag, ncall);
        }
    }

    if s + 1 < smax {
        if f[0][par] <= f[1][par] {
            *nboxes += 1;
            ipar[*nboxes] = par;
            level[*nboxes] = s + 1;
            ichild[*nboxes] = 1;
            f[0][*nboxes] = f[0][par];
            updtrec(*nboxes, level[*nboxes], &f[0], record);
            if s + 2 < smax {
                *nboxes += 1;
                ipar[*nboxes] = par;
                level[*nboxes] = s + 2;
                ichild[*nboxes] = 2;
                f[0][*nboxes] = f[1][par];
                updtrec(*nboxes, level[*nboxes], &f[0], record);
            } else {
                x[i] = z[1];
                *nbasket += 1;
                if xmin.len() == *nbasket {
                    xmin.push(x.clone());
                    fmi.push(f[1][par]);
                } else {
                    xmin[*nbasket] = x.clone();
                    fmi[*nbasket] = f[1][par];
                }
            }
        } else {
            if s + 2 < smax {
                *nboxes += 1;
                ipar[*nboxes] = par;
                level[*nboxes] = s + 2;
                ichild[*nboxes] = 1;
                f[0][*nboxes] = f[0][par];
                updtrec(*nboxes, level[*nboxes], &f[0], record);
            } else {
                x[i] = z[0];
                *nbasket += 1;
                if xmin.len() == *nbasket {
                    xmin.push(x.clone());
                    fmi.push(f[0][par]);
                } else {
                    xmin[*nbasket] = x.clone();
                    fmi[*nbasket] = f[0][par];
                }
            }
            *nboxes += 1;
            ipar[*nboxes] = par;
            level[*nboxes] = s + 1;
            ichild[*nboxes] = 2;
            f[0][*nboxes] = f[1][par];
            updtrec(*nboxes, level[*nboxes], &f[0], record);
        }

        if z[1] != y[i] {
            if (z[1] - y[i]).abs() > (z[1] - z[0]).abs() * (3.0 - SQRT_5) * 0.5 {
                *nboxes += 1;
                ipar[*nboxes] = par;
                level[*nboxes] = s + 1;
                ichild[*nboxes] = 3;
                f[0][*nboxes] = f[1][par];
                updtrec(*nboxes, level[*nboxes], &f[0], record);
            } else {
                if s + 2 < smax {
                    *nboxes += 1;
                    ipar[*nboxes] = par;
                    level[*nboxes] = s + 2;
                    ichild[*nboxes] = 3;
                    f[0][*nboxes] = f[1][par];
                    updtrec(*nboxes, level[*nboxes], &f[0], record);
                } else {
                    x[i] = z[1];
                    *nbasket += 1;
                    if xmin.len() == *nbasket {
                        xmin.push(x.clone());
                        fmi.push(f[1][par]);
                    } else {
                        xmin[*nbasket] = x.clone();
                        fmi[*nbasket] = f[1][par];
                    }
                }
            }
        }
    } else {
        let mut xi1 = x.clone();
        let mut xi2 = x.clone();

        xi1[i] = z[0];
        *nbasket += 1;
        if xmin.len() == *nbasket {
            xmin.push(xi1.clone());
            fmi.push(f[0][par]);
        } else {
            xmin[*nbasket] = xi1;
            fmi[*nbasket] = f[0][par];
        }
        xi2[i] = z[1];
        *nbasket += 1;
        if xmin.len() == *nbasket {
            xmin.push(xi2.clone());
            fmi.push(f[1][par]);
        } else {
            xmin[*nbasket] = xi2;
            fmi[*nbasket] = f[1][par];
        }
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
    use approx::assert_relative_eq;

    #[test]
    fn test_splinit_0() {
        let i = 1_usize;
        let s = 2_usize;
        let smax = 3_usize;
        let par = 4_usize;
        let x0: Vec<&[f64]> = vec![&[0.0; 3]; 6];
        let u = vec![0.0; 6];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x = vec![0.0; 6];
        let l = vec![1, 2, 3, 4, 5, 6];
        let l_lower = vec![0; 6];
        let mut xmin = vec![vec![0.0; 6]];
        let mut fmi = vec![0.0];
        let mut ipar = vec![0; 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![0; 10];
        let mut f = vec![vec![0.0; 10]; 2];
        let mut xbest = vec![0.0; 6];
        let mut fbest = 0.0;
        let stop = vec![1.0, 2.0, f64::NEG_INFINITY];
        let mut record = vec![0; 10];
        let mut nboxes = 1_usize;
        let mut nbasket = 0_usize;
        let mut nsweepbest = 0_usize;
        let mut nsweep = 1_usize;

        // Call splinit
        let (f0, flag, ncall) = splinit(
            i, s, smax, par, &x0, &u, &v, &mut x, &l, &l_lower, &mut xmin,
            &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest,
            &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket,
            &mut nsweepbest, &mut nsweep,
        );

        assert_eq!(xbest, vec![0.0; 6]);
        assert_relative_eq!(fbest, -0.00508911288366444);
        assert_eq!(f0,
                   vec![0.0, -0.00508911288366444, -0.00508911288366444, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(xmin, vec![vec![0.0; 6]; 4]);
        assert_eq!(
            fmi,
            vec![0.0, 0.0, -0.00508911288366444, -0.00508911288366444]
        );
        assert_eq!(ipar, vec![0; 10]);
        assert_eq!(level, vec![0; 10]);
        assert_eq!(ichild, vec![0; 10]);
        assert_eq!(f, vec![vec![0.0; 10]; 2]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 2);
        assert_eq!(record, vec![0; 10]);
        assert_eq!(nboxes, 1);
        assert_eq!(nbasket, 3);
        assert_eq!(nsweepbest, 1);
        assert_eq!(nsweep, 1);
    }

    #[test]
    fn test_splinit_1() {
        // Setup inputs
        let i = 1_usize;
        let s = 1_usize;
        let smax = 2_usize;
        let par = 3_usize;
        let x0: Vec<&[f64]> = vec![&[0.0; 3]; 6];
        let u = vec![0.0; 6];
        let v = vec![1.0; 6];
        let mut x = vec![0.0; 6];
        let l = vec![2, 2, 2, 2, 2, 2];
        let l_lower = vec![0; 6];
        let mut xmin = vec![vec![1.0; 6], vec![1.0; 6], vec![1.0; 6]];
        let mut fmi = vec![10.0, 10.0, 10.0];
        let mut ipar = vec![1; 10];
        let mut level = vec![1; 10];
        let mut ichild = vec![1; 10];
        let mut f = vec![vec![1.0; 10]; 2];
        let mut xbest = vec![1.0; 6];
        let mut fbest = 0.0;
        let stop = vec![1.0, 2.0, f64::NEG_INFINITY];
        let mut record = vec![1; 10];
        let mut nboxes = 2_usize;
        let mut nbasket = 2_usize;
        let mut nsweepbest = 4_usize;
        let mut nsweep = 5_usize;

        // Call splinit
        let (f0, flag, ncall) = splinit(
            i, s, smax, par, &x0, &u, &v, &mut x, &l, &l_lower, &mut xmin,
            &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest,
            &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket,
            &mut nsweepbest, &mut nsweep,
        );

        assert_eq!(xbest, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(fbest, -0.00508911288366444);
        assert_eq!(
            f0,
            vec![1.0, -0.00508911288366444, -0.00508911288366444]
        );
        assert_eq!(xmin.len(), 6);
        assert_eq!(
            fmi,
            vec![
                10.0, 10.0, 10.0, 1.0, -0.00508911288366444,
                -0.00508911288366444
            ]
        );
        assert_eq!(ipar, vec![1; 10]);
        assert_eq!(level, vec![1; 10]);
        assert_eq!(ichild, vec![1; 10]);
        assert_eq!(f, vec![vec![1.0; 10]; 2]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 2);
        assert_eq!(record, vec![1; 10]);
        assert_eq!(nboxes, 2);
        assert_eq!(nbasket, 5);
        assert_eq!(nsweepbest, 5);
        assert_eq!(nsweep, 5);
    }

    #[test]
    fn test_split_0() {
        // Setup inputs
        let i = 0_usize;
        let s = 0_usize;
        let smax = 1_usize;
        let par = 0_usize;
        let mut x = vec![0.0; 6];
        let mut y = vec![0.0; 6];
        let mut z = vec![0.0; 2];
        let mut xmin = vec![vec![0.0; 6]];
        let mut fmi = vec![0.0];
        let mut ipar = vec![0; 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![0; 10];
        let mut f = vec![vec![0.0; 10]; 2];
        let mut xbest = vec![0.0; 6];
        let mut fbest = 0.0;
        let stop = vec![0.0, 0.0, f64::NEG_INFINITY];
        let mut record = vec![0; 10];
        let mut nboxes = 1_usize;
        let mut nbasket = 0_usize;
        let mut nsweepbest = 0_usize;
        let mut nsweep = 1_usize;

        // Call split
        let (flag, ncall) = split(
            i, s, smax, par, &mut x, &mut y, &mut z, &mut xmin, &mut fmi, &mut ipar, &mut level,
            &mut ichild, &mut f, &mut xbest, &mut fbest, &stop, &mut record, &mut nboxes,
            &mut nbasket, &mut nsweepbest, &mut nsweep);

        assert_eq!(xbest, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(fbest, -0.00508911288366444);
        assert_eq!(fmi, vec![0.0, 0.0, -0.00508911288366444]);
        assert_eq!(flag, true);
        assert_eq!(ncall, 1);
        assert_eq!(stop, vec![0.0, 0.0, f64::NEG_INFINITY]);
        assert_eq!(record, vec![0; 10]);
        assert_relative_eq!(fbest, -0.00508911288366444);
        assert_eq!(xmin, vec![vec![0.0; 6]; 3]);
        assert_eq!(nboxes, 1);
        assert_eq!(nbasket, 2);
        assert_eq!(nsweepbest, 1);
        assert_eq!(nsweep, 1);
    }

    #[test]
    fn test_split_1() {
        // Setup inputs
        let i = 2_usize;
        let s = 3_usize;
        let smax = 8_usize;
        let par = 3_usize;
        let mut x = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0];
        let mut y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut z = vec![5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, 6.0, -6.0, 6.0];
        let mut xmin = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let mut fmi = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let mut ipar = vec![1; 6];
        let mut level = vec![1; 6];
        let mut ichild = vec![1; 6];
        let mut f = vec![vec![1.0; 6]; 2];
        let mut xbest = vec![1.0; 6];
        let mut fbest = 0.0;
        let stop = vec![0.0, 0.0, f64::NEG_INFINITY];
        let mut record = vec![1; 6];
        let mut nboxes = 1_usize;
        let mut nbasket = 0_usize;
        let mut nsweepbest = 3_usize;
        let mut nsweep = 1_usize;

        let (flag, ncall) = split(
            i, s, smax, par, &mut x, &mut y, &mut z, &mut xmin, &mut fmi, &mut ipar, &mut level,
            &mut ichild, &mut f, &mut xbest, &mut fbest, &stop, &mut record, &mut nboxes,
            &mut nbasket, &mut nsweepbest, &mut nsweep);

        assert_eq!(fmi, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        assert_eq!(ipar, vec![1, 1, 3, 3, 3, 1]);
        assert_eq!(level, vec![1, 1, 5, 4, 4, 1]);
        assert_eq!(ichild, vec![1, 1, 1, 2, 3, 1]);
        assert_eq!(record, vec![1, 1, 1, 1, 3, 1]);
        assert_eq!(xbest, vec![1.0; 6]);
        assert_eq!(fbest, 0.0);
        assert_eq!(flag, true);
        assert_eq!(ncall, 1);
        assert_eq!(nboxes, 4);
        assert_eq!(nbasket, 0);
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
        assert_relative_eq!(result, expected);
    }

    #[test]
    fn split1_test_1() {
        let x1 = 0.0_f64;
        let x2 = 1.0_f64;
        let f1 = -0.5_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 0.3819660112501051_f64;
        assert_relative_eq!(result, expected);
    }

    #[test]
    fn split1_test_2() {
        let x1 = 0.0_f64;
        let x2 = 0.0_f64;
        let f1 = -1.0_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 0.0_f64;
        assert_relative_eq!(result, expected);
    }

    #[test]
    fn split1_test_3() {
        let x1 = 0.0_f64;
        let x2 = 1.0_f64;
        let f1 = -1.0_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 0.6180339887498949_f64;
        assert_relative_eq!(result, expected);
    }

    #[test]
    fn split1_test_4() {
        let x1 = 1e-10_f64;
        let x2 = 2e-10_f64;
        let f1 = 1.0_f64;
        let f2 = 2.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 1.618033988749895e-10_f64;
        assert_relative_eq!(result, expected);
    }

    #[test]
    fn split1_test_5() {
        let x1 = 10.0_f64;
        let x2 = 20.0_f64;
        let f1 = -1.0000001_f64;
        let f2 = -1.0_f64;
        let result = split1(x1, x2, f1, f2);
        let expected = 16.18033988749895_f64;
        assert_relative_eq!(result, expected);
    }

    // split2 ----------------------
    #[test]
    fn split2_test_0() {
        let x = 0.5;
        let y = 0.19;
        let result = split2(x, y);
        assert_relative_eq!(result, 0.29333333333333333, max_relative = 1e-8);
    }

    #[test]
    fn split2_test_1() {
        let x = 0.0;
        let y = 2000.0;
        let result = split2(x, y);
        assert_relative_eq!(result, 0.6666666666666666, max_relative = 1e-8);
    }

    #[test]
    fn split2_test_2() {
        let x = 0.0;
        let y = -2000.0;
        let result = split2(x, y);
        assert_relative_eq!(result, -0.6666666666666666, max_relative = 1e-8);
    }

    #[test]
    fn split2_test_3() {
        let x = 0.5;
        let y = f64::INFINITY;
        let result = split2(x, y);
        assert_relative_eq!(result, 3.5, max_relative = 1e-8);
    }

    #[test]
    fn split2_test_4() {
        let x = 0.5;
        let y = f64::NEG_INFINITY;
        let result = split2(x, y);
        assert_relative_eq!(result, -3.1666666666666665, max_relative = 1e-8);
    }
}

