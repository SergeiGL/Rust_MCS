use crate::gls::lsguard::lsguard;
use crate::gls::lsnew::lsnew;
use crate::gls::lssort::lssort;
use nalgebra::SVector;

pub fn lspar<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    nloc: usize,
    small: f64,
    sinit: usize,
    short: f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    alp: &mut f64,
    abest: &mut f64,
    fmed: &mut f64,
    unitlen: &mut f64,
    s: &mut usize,
) -> (
    f64,         // fbest
    Vec<bool>,   // up
    Vec<bool>,   // down
    bool,        // monotone
    Vec<bool>,   // minima
    usize,       // nmin
) {
    if *s < 3 {
        *alp = lsnew(func, nloc, small, sinit, short, x, p, *s, alist, flist, amin, amax, *abest, *fmed, *unitlen);
    } else {
        // Select three points for parabolic interpolation
        let i = flist
            .iter()
            .enumerate()
            .min_by(|(_, a_i), (_, b_i)| a_i.total_cmp(b_i))
            .unwrap()
            .0;

        let ((aa0, aa1, aa2), (ff0, ff1, ff2), ii) = if i <= 1 {
            (
                (alist[0], alist[1], alist[2]),
                (flist[0], flist[1], flist[2]),
                i
            )
        } else if i >= *s - 2 {
            (
                (alist[*s - 3], alist[*s - 2], alist[*s - 1]),
                (flist[*s - 3], flist[*s - 2], flist[*s - 1]),
                i + 3 - *s)
        } else {
            panic!();
        };

        // Divided differences
        let f12 = (ff1 - ff0) / (aa1 - aa0);
        let f23 = (ff2 - ff1) / (aa2 - aa1);
        let f123 = (f23 - f12) / (aa2 - aa0);

        if !(f123 > 0.0) {
            *alp = lsnew(func, nloc, small, sinit, short, x, p, *s, alist, flist, amin, amax, *abest, *fmed, *unitlen);
        } else {
            // Parabolic minimizer
            let alp0 = 0.5 * (aa1 + aa2 - f23 / f123);
            *alp = lsguard(alp0, alist, amax, amin, small);

            let alptol = small * (aa2 - aa0);

            // Handle infinities and close predictor
            if f123 == f64::INFINITY || alist.iter().any(|&a| (a - *alp).abs() <= alptol) {
                if ii == 0 || (ii == 1 && (aa1 >= 0.5 * (aa0 + aa2))) {
                    *alp = 0.5 * (aa0 + aa1);
                } else {
                    *alp = 0.5 * (aa1 + aa2);
                }
            }

            // Evaluate the function at the new alpha
            let falp = func(&(x + p.scale(*alp)));

            // Add the new point to alist and flist
            alist.push(*alp);
            flist.push(falp);
        }
    }

    let (fbest, up, down, monotone, minima, nmin);
    (*abest, fbest, *fmed, up, down, monotone, minima, nmin, *unitlen, *s) = lssort(alist, flist);

    (fbest, up, down, monotone, minima, nmin)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_0() {
        let nloc = 1;
        let small = 1e-8;
        let sinit = 1;
        let short = 2.0;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.1, 3.2, 4.1, 5.4, 0.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 0.1, 0.2, 0.4, 0.4, 0.5];
        let mut flist = vec![0.0, 0.1, 0.4, 0.3, 0.42, 0.21];
        let amin = -10.0;
        let amax = 10.0;
        let mut alp = 0.0;
        let mut abest = 0.0;
        let mut fmed = 0.0;
        let mut unitlen = 1.0;
        let mut s = 3;

        let (fbest, up, down, monotone, minima, nmin) = lspar(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax,
                                                              &mut alp, &mut abest, &mut fmed, &mut unitlen, &mut s);

        assert_eq!(alist, vec![0.0, 0.03333333333333333, 0.1, 0.2, 0.4, 0.4, 0.5]);
        assert_eq!(flist, vec![0.0, -5.446883046391155e-65, 0.1, 0.4, 0.3, 0.42, 0.21]);
        assert_eq!(abest, 0.03333333333333333);
        assert_eq!(fbest, -5.446883046391155e-65);
        assert_eq!(fmed, 0.21);
        assert_eq!(up, vec![false, true, true, false, true, false]);
        assert_eq!(down, vec![true, false, false, true, false, true]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, true, false, false, true, false, true]);
        assert_eq!(nmin, 3);
        assert_eq!(unitlen, 0.3666666666666667);
        assert_eq!(s, 7);
        assert_eq!(alp, 0.03333333333333333);
    }

    #[test]
    fn test_1() {
        let nloc = 1;
        let small = 1e-8;
        let sinit = 1;
        let short = 2.0;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.1, 3.2, 4.1, 5.4, 0.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 0.1, 0.2, 0.11, 0.4, 0.13];
        let mut flist = vec![0.0, 0.31, 0.4, 0.1, 0.4, 0.91];
        let amin = -10.0;
        let amax = 10.0;
        let mut alp = 0.0;
        let mut abest = 0.0;
        let mut fmed = 0.0;
        let mut unitlen = 1.0;
        let mut s = 2;

        let (fbest, up, down, monotone, minima, nmin) = lspar(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax,
                                                              &mut alp, &mut abest, &mut fmed, &mut unitlen, &mut s);

        assert_eq!(alist, vec![-10.0, 0.0, 0.1, 0.11, 0.13, 0.2, 0.4]);
        assert_eq!(flist, vec![0.0, 0.0, 0.31, 0.1, 0.91, 0.4, 0.4]);
        assert_eq!(abest, -10.0);
        assert_eq!(fbest, 0.0);
        assert_eq!(fmed, 0.31);
        assert_eq!(up, vec![false, true, false, true, false, false]);
        assert_eq!(down, vec![true, false, true, false, true, false, ]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, true, false, true, false, false, false]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 10.0);
        assert_eq!(s, 7);
        assert_eq!(alp, -10.0);
    }

    #[test]
    fn test_2() {
        let nloc = 2;
        let small = 1e-5;
        let sinit = 3;
        let short = 5.0;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.1, 3.2, 4.1, 5.4, 0.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 0.11, 0.2, 0.13, 0.4, 0.41];
        let mut flist = vec![0.0, 0.1, 0.4, 0.132, 0.4, 0.231];
        let amin = 1.0;
        let amax = 10.0;
        let mut alp = 0.0;
        let mut abest = 0.0;
        let mut fmed = 0.0;
        let mut unitlen = 5.0;
        let mut s = 3;

        let (fbest, up, down, monotone, minima, nmin) = lspar(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax,
                                                              &mut alp, &mut abest, &mut fmed, &mut unitlen, &mut s);


        assert_eq!(alist, vec![0.0, 0.11, 0.13, 0.2, 0.4, 0.41, 1.0]);
        assert_eq!(flist, [0.0, 0.1, 0.4, 0.132, 0.4, 0.231, -7.386702232051913e-134]);
        assert_eq!(abest, 1.0);
        assert_eq!(fbest, -7.386702232051913e-134);
        assert_eq!(fmed, 0.132);
        assert_eq!(up, vec![true, true, false, true, false, false]);
        assert_eq!(down, vec![false, false, true, false, true, true]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![true, false, false, true, false, false, true]);
        assert_eq!(nmin, 3);
        assert_eq!(unitlen, 0.8);
        assert_eq!(s, 7);
        assert_eq!(alp, 1.0);
    }

    #[test]
    fn test_3() {
        let nloc = 0;
        let small = 1e-10;
        let sinit = 0;
        let short = 0.01;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.1, 3.2, 4.1, 5.4, 0.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 2.0, 1.0, 3.0];
        let mut flist = vec![0.0, 2.0, 1.0, 3.0];
        let amin = -10.0;
        let amax = 10.0;
        let mut alp = 0.0;
        let mut abest = 0.0;
        let mut fmed = 0.0;
        let mut unitlen = 1.0;
        let mut s = 3;

        let (fbest, up, down, monotone, minima, nmin) = lspar(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax,
                                                              &mut alp, &mut abest, &mut fmed, &mut unitlen, &mut s);

        assert_eq!(alist, vec![-10.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(flist, [0.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(abest, -10.0);
        assert_eq!(fbest, 0.0);
        assert_eq!(fmed, 1.);
        assert_eq!(up, vec![false, true, true, true]);
        assert_eq!(down, vec![true, false, false, false]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, true, false, false, false]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 13.0);
        assert_eq!(s, 5);
        assert_eq!(alp, -10.0);
    }
}