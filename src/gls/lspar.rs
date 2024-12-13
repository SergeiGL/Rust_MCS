use crate::feval::feval;
use crate::gls::lsguard::lsguard;
use crate::gls::lsnew::lsnew;
use crate::gls::lssort::lssort;
use nalgebra::SVector;

pub fn lspar<const N: usize>(
    nloc: usize,
    small: f64,
    sinit: usize,
    short: f64,
    x: &[f64; N],
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    mut alp: f64,
    abest: f64,
    fmed: f64,
    unitlen: f64,
    s: usize,
) -> (
    f64,         // abest
    f64,         // fbest
    f64,         // fmed
    Vec<bool>,   // up
    Vec<bool>,   // down
    bool,        // monotone
    Vec<bool>,   // minima
    usize,       // nmin
    f64,         // unitlen
    usize,       // s
    f64,         // alp
    f64,         // fac
) {
    let mut cont = true;
    let mut fac = short;

    if s < 3 {
        (alp, fac) = lsnew(nloc, small, sinit, short, x, p, s, alist, flist, amin, amax, abest, fmed, unitlen);
        cont = false;
    }

    if cont {
        // Select three points for parabolic interpolation
        let i = flist
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let (ind, ii) = if i <= 1 {
            ([0, 1, 2], i)
        } else if i >= s - 2 {
            ([s - 3, s - 2, s - 1], i + 3 - s)
        } else {
            panic!();
            // ([ii - 1, i, i + 1], 1) // TODO: strange
        };

        // Retrieve values from alist and flist based on ind
        let aa: [f64; 3] = std::array::from_fn(|i| alist[ind[i]]);
        let ff: [f64; 3] = std::array::from_fn(|i| flist[ind[i]]);

        // Divided differences
        let f12 = (ff[1] - ff[0]) / (aa[1] - aa[0]);
        let f23 = (ff[2] - ff[1]) / (aa[2] - aa[1]);
        let f123 = (f23 - f12) / (aa[2] - aa[0]);

        if !(f123 > 0.0) {
            (alp, fac) =
                lsnew(nloc, small, sinit, short, x, p, s, alist, flist, amin, amax, abest, fmed, unitlen);
            cont = false;
        }

        if cont {
            // Parabolic minimizer
            let alp0 = 0.5 * (aa[1] + aa[2] - f23 / f123);
            alp = lsguard(alp0, alist, amax, amin, small);

            let alptol = small * (aa[2] - aa[0]);

            // Handle infinities and close predictor
            if f123 == f64::INFINITY || alist.iter().any(|&a| (a - alp).abs() <= alptol) {
                if ii == 0 || (ii == 1 && (aa[1] >= 0.5 * (aa[0] + aa[2]))) {
                    alp = 0.5 * (aa[0] + aa[1]);
                } else {
                    alp = 0.5 * (aa[1] + aa[2]);
                }
            }

            // Evaluate the function at the new alpha
            let falp = feval(&std::array::from_fn::<f64, N, _>(|i| x[i] + alp * p[i]));

            // Add the new point to alist and flist
            alist.push(alp);
            flist.push(falp);
        }
    }

    let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(alist, flist);

    (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s, alp, fac)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let nloc = 1;
        let small = 1e-8;
        let sinit = 1;
        let short = 2.0;
        let x = [1.0, 2.1, 3.2, 4.1, 5.4, 0.1];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 0.1, 0.2, 0.4, 0.4, 0.5];
        let mut flist = vec![0.0, 0.1, 0.4, 0.3, 0.42, 0.21];
        let amin = -10.0;
        let amax = 10.0;
        let alp = 0.0;
        let abest = 0.0;
        let fmed = 0.0;
        let unitlen = 1.0;
        let s = 3;

        let (abest, fbest, fmed, up, down,
            monotone, minima, nmin, unitlen, s, alp, fac) =
            lspar(nloc, small, sinit, short, &x, &p, &mut alist, &mut flist,
                  amin, amax, alp, abest, fmed, unitlen, s);

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
        assert_eq!(fac, 2.0);
    }

    #[test]
    fn test_1() {
        let nloc = 1;
        let small = 1e-8;
        let sinit = 1;
        let short = 2.0;
        let x = [1.0, 2.1, 3.2, 4.1, 5.4, 0.1];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 0.1, 0.2, 0.11, 0.4, 0.13];
        let mut flist = vec![0.0, 0.31, 0.4, 0.1, 0.4, 0.91];
        let amin = -10.0;
        let amax = 10.0;
        let alp = 0.0;
        let abest = 0.0;
        let fmed = 0.0;
        let unitlen = 1.0;
        let s = 2;

        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s, alp, fac) = lspar(
            nloc, small, sinit, short, &x, &p, &mut alist, &mut flist,
            amin, amax, alp, abest, fmed, unitlen, s);

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
        assert_eq!(fac, 2.0);
    }

    #[test]
    fn test_2() {
        let nloc = 2;
        let small = 1e-5;
        let sinit = 3;
        let short = 5.0;
        let x = [1.0, 2.1, 3.2, 4.1, 5.4, 0.1];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 0.11, 0.2, 0.13, 0.4, 0.41];
        let mut flist = vec![0.0, 0.1, 0.4, 0.132, 0.4, 0.231];
        let amin = 1.0;
        let amax = 10.0;
        let alp = 0.0;
        let abest = 0.0;
        let fmed = 0.0;
        let unitlen = 5.0;
        let s = 3;

        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s, alp, fac) = lspar(
            nloc, small, sinit, short, &x, &p, &mut alist, &mut flist,
            amin, amax, alp, abest, fmed, unitlen, s);


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
        assert_eq!(fac, 5.0);
    }

    #[test]
    fn test_3() {
        let nloc = 0;
        let small = 1e-10;
        let sinit = 0;
        let short = 0.01;
        let x = [1.0, 2.1, 3.2, 4.1, 5.4, 0.1];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut alist = vec![0.0, 2.0, 1.0, 3.0];
        let mut flist = vec![0.0, 2.0, 1.0, 3.0];
        let amin = -10.0;
        let amax = 10.0;
        let alp = 0.0;
        let abest = 0.0;
        let fmed = 0.0;
        let unitlen = 1.0;
        let s = 3;

        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s, alp, fac) = lspar(
            nloc, small, sinit, short, &x, &p, &mut alist, &mut flist,
            amin, amax, alp, abest, fmed, unitlen, s);

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
        assert_eq!(fac, 0.01);
    }
}