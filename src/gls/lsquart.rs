use crate::gls::lsguard::lsguard;
use crate::gls::lslocal::lslocal;
use crate::gls::lssort::lssort;
use crate::gls::quartic::quartic;
use nalgebra::{Matrix3, SVector};

pub fn lsquart<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    nloc: usize,
    small: f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: &mut f64,
    amax: &mut f64,
    alp: &mut f64,
    abest: &mut f64,
    fbest: &mut f64,
    fmed: &mut f64,
    up: &mut Vec<bool>,
    down: &mut Vec<bool>,
    monotone: &mut bool,
    minima: &mut Vec<bool>,
    nmin: &mut usize,
    unitlen: &mut f64,
    s: &mut usize,
    saturated: &mut bool,
) {
    let f12 = if alist[0] == alist[1] { 0.0 } else { (flist[1] - flist[0]) / (alist[1] - alist[0]) };
    let f23 = if alist[1] == alist[2] { 0.0 } else { (flist[2] - flist[1]) / (alist[2] - alist[1]) };
    let f34 = if alist[2] == alist[3] { 0.0 } else { (flist[3] - flist[2]) / (alist[3] - alist[2]) };
    let f45 = if alist[3] == alist[4] { 0.0 } else { (flist[4] - flist[3]) / (alist[4] - alist[3]) };

    let f123 = (f23 - f12) / (alist[2] - alist[0]);
    let f234 = (f34 - f23) / (alist[3] - alist[1]);
    let f345 = (f45 - f34) / (alist[4] - alist[2]);
    let f1234 = (f234 - f123) / (alist[3] - alist[0]);
    let f2345 = (f345 - f234) / (alist[4] - alist[1]);
    let f12345 = (f2345 - f1234) / (alist[4] - alist[0]);

    if f12345 <= 0.0 { // quart false
        // quartic not bounded below
        lslocal(func, nloc, small, x, p, alist, flist, *amin, *amax, alp, abest, fbest, fmed,
                up, down, monotone, minima, nmin, unitlen, s, saturated);
    } else { // quart true
        // Expanding around alist[2]
        let mut c: SVector<f64, 5> = SVector::zeros();
        c[0] = f12345;
        c[1] = f1234 + c[0] * (alist[2] - alist[0]);
        c[2] = f234 + c[1] * (alist[2] - alist[3]);
        c[1] += c[0] * (alist[2] - alist[3]);
        c[3] = f23 + c[2] * (alist[2] - alist[1]);
        c[2] += c[1] * (alist[2] - alist[1]);
        c[1] += c[0] * (alist[2] - alist[1]);
        c[4] = flist[2];

        let cmax = c.normalize_mut();

        let hk = 4.0 * c[0];
        let compmat = Matrix3::<f64>::new(
            0.0, 0.0, -c[3],
            hk, 0.0, -2.0 * c[2],
            0.0, hk, -3.0 * c[1],
        );

        // Calculate eigenvalues (complex)
        let ev = compmat.complex_eigenvalues();

        let n_real_roots = ev.iter().filter(|ev_i| ev_i.im == 0.0).count();

        if n_real_roots == 1 {
            *alp = alist[2] + (ev[0].re / hk); // Img part is 0
        } else {
            let mut alp1 = alist[2] + ev[0].re.min(ev[1].re.min(ev[2].re)) / hk;
            lsguard(&mut alp1, alist, *amax, *amin, small);

            let mut alp2 = alist[2] + ev[0].re.max(ev[1].re.max(ev[2].re)) / hk;
            lsguard(&mut alp2, alist, *amax, *amin, small);

            let f1 = cmax * quartic(&c, alp1 - alist[2]);
            let f2 = cmax * quartic(&c, alp2 - alist[2]);

            *alp = if alp2 > alist[4] && f2 < *flist.iter().max_by(|a, b| a.total_cmp(b)).unwrap() {
                alp2
            } else if alp1 < alist[0] && f1 < *flist.iter().max_by(|a, b| a.total_cmp(b)).unwrap() {
                alp1
            } else if f2 <= f1 {
                alp2
            } else {
                alp1
            };
        }

        if !alist.iter().any(|&v| v == *alp) { // opposite to if max(alist==alp);
            lsguard(alp, alist, *amax, *amin, small);
            let falp = func(&(x + p.scale(*alp)));
            alist.push(*alp);
            flist.push(falp);
            (*abest, *fbest, *fmed, *up, *down, *monotone, *minima, *nmin, *unitlen, *s) = lssort(alist, flist);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-14;

    #[test]
    fn test_real_mistake_0() {
        let nloc = 1;
        let small = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[0.190983, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-1.6, -1.1, -0.6, -0.40767007775845987, 0.0];
        let mut flist = vec![-0.00033007085742366247, -0.005222762849281021, -0.019362461793975085, -0.02327501776110989, -0.01523227097945881];
        let (mut amin, mut amax, mut alp, mut abest, mut fbest, mut fmed) = (-1.6, 0.4, -0.40767007775845987, -0.40767007775845987, -0.02327501776110989, -0.01523227097945881);
        let mut up = vec![false, false, false, true];
        let mut down = vec![true, true, true, false];
        let mut monotone = false;
        let mut minima = vec![false, false, false, true, false];
        let (mut nmin, mut unitlen, mut s) = (1, 1.1923299222415402, 5);
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        let expected_alist = vec![-1.6, -1.1, -0.6, -0.40767007775845987, -0.30750741391479824, 0.0];
        let expected_flist = vec![-0.00033007085742366247, -0.005222762849281021, -0.019362461793975085, -0.02327501776110989, -0.023740401281451082, -0.01523227097945881];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(amin, -1.6);
        assert_eq!(amax, 0.4);
        assert_relative_eq!(alp, -0.30750741391479824, epsilon = TOLERANCE);
        assert_relative_eq!(abest, -0.30750741391479824, epsilon = TOLERANCE);
        assert_relative_eq!(fbest, -0.023740401281451082, epsilon = TOLERANCE);
        assert_relative_eq!(fmed, -0.017297366386716948, epsilon = TOLERANCE);
        assert_eq!(monotone, false);
    }


    #[test]
    fn test_0() {
        let nloc = 5;
        let small = 1e-10;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 5.0, 6.0, 8.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.5, 0.5, 2.0, 3.0, 5.0, 1.0]);
        let mut alist = vec![0.0, 0.01, 1.0, 2.0, 3.0, 5.0];
        let mut flist = vec![1.0, 1.01, 2.0, 3.0, 4.0, 23.0];
        let (mut amin, mut amax, mut alp, mut abest, mut fbest, mut fmed) = (-1.0, 5.0, 1.0, 0.5, 1.5, 2.0);
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let mut monotone = true;
        let mut minima = vec![true];
        let (mut nmin, mut unitlen, mut s) = (1, 1.0, 3);
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        let expected_alist = vec![-1.0, 0.0, 0.01, 1.0, 2.0, 3.0, 5.0];
        let expected_flist = vec![-6.455214899134537e-154, 1.0, 1.01, 2.0, 3.0, 4.0, 23.0];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert!(up.iter().all(|&u| u));
        assert!(down.iter().all(|&d| !d));
        assert_eq!(amin, -1.0);
        assert_eq!(amax, 5.0);
        assert_eq!(alp, -1.0);
        assert_eq!(abest, -1.0);
        assert_eq!(fbest, -6.455214899134537e-154);
        assert_eq!(fmed, 2.0);
        assert!(monotone);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 6.0);
        assert_eq!(s, 7);
        assert!(!saturated);
    }

    #[test]
    fn test_1() {
        let nloc = 5;
        let small = 1e-10;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 0.5, 1.0, 2.0, 3.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.5, 0.5, 0.0, 2.0, 4.0, 1.0]);
        let mut alist = vec![-1.0, 0.0, 2.0, 4.0, 5.0, 3.0];
        let mut flist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
        let (mut amin, mut amax, mut alp, mut abest, mut fbest, mut fmed) = (-1.0, 5.0, 2.0, 2.0, 3.0, 3.0);
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let mut monotone = true;
        let mut minima = vec![true; 1];
        let (mut nmin, mut unitlen, mut s) = (1, 1.0, 4);
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        let expected_alist = vec![-1.0, 0.0, 2.0, 3.0, 4.0, 5.0];
        let expected_flist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(up, Vec::from([true, true, true]));
        assert_eq!(down, Vec::from([false, false, false]));
        assert_eq!(minima, Vec::from([true, false, false, false]));
        assert_eq!(monotone, true);
        assert_eq!(alp, -0.6666666666666666);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 1.0);
        assert_eq!(s, 4);
        assert!(saturated);
    }

    #[test]
    fn test_2() {
        let nloc = 5;
        let small = 1e-10;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 1.0 + 1e-9, 1.0 + 2e-9, 1.0 + 3e-9, 1.0 + 4e-9]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.5, 0.45, 5.5034, 0.5387, 5.8, 2.734]);
        let mut alist = vec![1.0, 1.0 + 1e-5, 1.0 + 2e-4, 1.0 + 3e-4, 3.0 + 4e-5, -2.0];
        let mut flist = vec![1.0, 1.1, 1.2, 1.3, 1.4, -2.0];
        let (mut amin, mut amax, mut alp, mut abest, mut fbest, mut fmed) = (0.0, 2.0, 1.0, 1.0, 1.0, 1.2);
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let mut monotone = true;
        let mut minima = vec![true; 1];
        let (mut nmin, mut unitlen, mut s) = (1, 1.0, 2);
        let mut saturated = true;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        let expected_alist = vec![-2.0, 0.0, 1.0, 1.00001, 1.0002, 1.0003, 3.00004];
        let expected_flist = vec![1.0, -7.70489595691683e-12, 1.1, 1.2, 1.3, 1.4, -2.0];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(up, Vec::from([false, true, true, true, true, false]));
        assert_eq!(down, Vec::from([true, false, false, false, false, true]));
        assert_eq!(minima, Vec::from([false, true, false, false, false, false, true]));
        assert_eq!(amin, 0.0);
        assert_eq!(amax, 2.0);
        assert_eq!(alp, 0.0);
        assert!(!monotone);
    }

    #[test]
    fn test_3() {
        let nloc = 3;
        let small = 1e-5;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 0.2325, 0.2, 1.0, 0.423, 0.4]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.5, 0.3, 0.25, 1.45, 0.537, 1.24]);
        let mut alist = vec![10.101, 1.0, 2.0, 3.0, 2.21, 1.4];
        let mut flist = vec![1.0, 0.5, 2.0, 1.5, 1.01, 1.5];
        let (mut amin, mut amax, mut alp, mut abest, mut fbest, mut fmed) = (-1.0, 5.0, 2.0, 1.0, 0.5, 1.5);
        let mut up = vec![false, false, true, false];
        let mut down = vec![true, true, true, true];
        let mut monotone = false;
        let mut minima = vec![true, false];
        let (mut nmin, mut unitlen, mut s) = (2, 1.0, 3);
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        let expected_alist = vec![1.0, 1.4, 2.0, 2.21, 3.0, 5.0, 10.101];
        let expected_flist = vec![0.5, 1.5, 2.0, 1.01, 1.5, -4.018396477612926e-236, 1.0];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(up, Vec::from([true, true, false, true, false, true]));
        assert_eq!(down, Vec::from([false, false, true, false, true, false]));
        assert_eq!(minima, Vec::from([true, false, false, true, false, true, false]));
        assert_eq!(amin, -1.0);
        assert_eq!(amax, 5.0);
        assert_eq!(alp, 5.0);
        assert_eq!(nmin, 3);
        assert_eq!(unitlen, 2.79);
        assert_eq!(s, 7);
    }
}

