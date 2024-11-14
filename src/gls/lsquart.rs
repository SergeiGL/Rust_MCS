use crate::feval::feval;
use crate::gls::lsguard::lsguard;
use crate::gls::lslocal::lslocal;
use crate::gls::lssort::lssort;
use crate::gls::quartic::quartic;
use nalgebra::Matrix3;
use ndarray::Array1;

pub fn lsquart(
    nloc: i32,
    small: f64,
    x: &Array1<f64>,
    p: &Array1<f64>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    mut alp: f64,
    mut abest: f64,
    mut fbest: f64,
    mut fmed: f64,
    up: &mut Vec<bool>,
    down: &mut Vec<bool>,
    mut monotone: bool,
    minima: &mut Vec<bool>,
    mut nmin: usize,
    mut unitlen: f64,
    mut s: usize,
    mut saturated: bool,
) -> (
    Vec<f64>, //alist
    Vec<f64>, //flist
    f64, //amin
    f64, //amax
    f64, //alp
    f64, // abest
    f64, // fbest
    f64, // fmed
    bool, //monotone
    usize, //nmin
    f64, //unitlen,
    usize, //s,
    f64, //good,
    bool //saturated
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

    let mut good = f64::INFINITY;
    let mut quart = false;

    if f12345 <= 0.0 {
        good = 0.0;
        let (
            new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin,
            new_unitlen, new_s, new_saturated
        ) = lslocal(
            nloc, small, x, p, alist, flist, amin, amax, alp, abest, fbest, fmed,
            up, down, monotone, minima, nmin, unitlen, s, saturated,
        );

        alp = new_alp;
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        monotone = new_monotone;
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;
        saturated = new_saturated;
    } else {
        quart = true;
    }

    if quart {
        // Expanding around alist[2]
        let mut c = vec![0.0; alist.len()];
        c[0] = f12345;
        c[1] = f1234 + c[0] * (alist[2] - alist[0]);
        c[2] = f234 + c[1] * (alist[2] - alist[3]);
        c[1] += c[0] * (alist[2] - alist[3]);
        c[3] = f23 + c[2] * (alist[2] - alist[1]);
        c[2] += c[1] * (alist[2] - alist[1]);
        c[1] += c[0] * (alist[2] - alist[1]);
        c[4] = flist[2];

        let cmax = c.iter().cloned().fold(f64::NAN, f64::max);
        for ci in &mut c {
            *ci /= cmax;
        }

        let hk = 4.0 * c[0];
        let compmat = Matrix3::new(
            0.0, 0.0, -c[3],
            hk, 0.0, -2.0 * c[2],
            0.0, hk, -3.0 * c[1],
        );

        // Calculate eigenvalues (complex).
        let real_ev: Vec<f64> = compmat.eigenvalues().unwrap().iter().copied().collect();

        if real_ev.len() == 1 {
            alp = alist[2] + real_ev[0];
        } else if real_ev.len() >= 3 {
            let ev_sorted = &real_ev;

            let alp1 = lsguard(alist[2] + ev_sorted[0], alist, amax, amin, small);
            let alp2 = lsguard(alist[2] + ev_sorted[2], alist, amax, amin, small);

            let c_arr = Array1::from_vec(c.clone());
            let f1 = cmax * quartic(&c_arr, alp1 - alist[2]);
            let f2 = cmax * quartic(&c_arr, alp2 - alist[2]);

            alp = if alp2 > alist[4] && f2 < *flist.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() {
                alp2
            } else if alp1 < alist[0] && f1 < *flist.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() {
                alp1
            } else if f2 <= f1 {
                alp2
            } else {
                alp1
            };
        }

        if alist.iter().any(|&v| v == alp) {
            quart = false;
        }

        if quart {
            alp = lsguard(alp, alist, amax, amin, small);
            let falp = feval(&(x + &(p * alp))); // Ensure correct feval usage here

            alist.push(alp);
            flist.push(falp);

            let (
                sorted_alist, sorted_flist, new_abest, new_fbest, new_fmed, new_up, new_down,
                new_monotone, new_minima, new_nmin, new_unitlen, new_s
            ) = lssort(alist, flist);

            *alist = sorted_alist;
            *flist = sorted_flist;
            abest = new_abest;
            fbest = new_fbest;
            fmed = new_fmed;
            *up = new_up;
            *down = new_down;
            monotone = new_monotone;
            *minima = new_minima;
            nmin = new_nmin;
            unitlen = new_unitlen;
            s = new_s;
        }
    }

    (alist.clone(), flist.clone(), amin, amax, alp, abest, fbest, fmed, monotone,
     nmin, unitlen, s, good, saturated)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_0() {
        let nloc = 5;
        let small = 1e-10;
        let x = Array1::from(vec![1.0, 2.0, 3.0, 5.0, 6.0, 8.0]);
        let p = Array1::from(vec![0.5, 0.5, 2.0, 3.0, 5.0, 1.0]);
        let mut alist = vec![0.0, 0.01, 1.0, 2.0, 3.0, 5.0];
        let mut flist = vec![1.0, 1.01, 2.0, 3.0, 4.0, 23.0];
        let (amin, amax, alp, abest, fbest, fmed) = (-1.0, 5.0, 1.0, 0.5, 1.5, 2.0);
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let monotone = true;
        let mut minima = vec![true];
        let (nmin, unitlen, s) = (1, 1.0, 3);
        let saturated = false;

        let (alist_res, flist_res, amin_res, amax_res, alp_res, abest_res, fbest_res, fmed_res, monotone_res, nmin_res, unitlen_res, s_res, good, saturated_res)
            = lsquart(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        let expected_alist = vec![-1.0, 0.0, 0.01, 1.0, 2.0, 3.0, 5.0];
        let expected_flist = vec![-6.455214899134537e-154, 1.0, 1.01, 2.0, 3.0, 4.0, 23.0];

        for (a_exp, a_res) in expected_alist.iter().zip(alist_res.iter()) {
            assert_relative_eq!(a_exp, a_res);
        }

        for (f_exp, f_res) in expected_flist.iter().zip(flist_res.iter()) {
            assert_relative_eq!(f_exp, f_res);
        }

        assert!(up.iter().all(|&u| u));
        assert!(down.iter().all(|&d| !d));
        assert_relative_eq!(amin_res, -1.0);
        assert_relative_eq!(amax_res, 5.0);
        assert_relative_eq!(alp_res, -1.0);
        assert_relative_eq!(abest_res, -1.0);
        assert_relative_eq!(fbest_res, -6.455214899134537e-154);
        assert_relative_eq!(fmed_res, 2.0);
        assert!(monotone_res);
        assert_eq!(nmin_res, 1);
        assert_relative_eq!(unitlen_res, 6.0);
        assert_eq!(s_res, 7);
        assert_eq!(good, 0.0);
        assert!(!saturated_res);
    }

    #[test]
    fn test_1() {
        let nloc = 5;
        let small = 1e-10;
        let x = Array1::from(vec![1.0, 2.0, 0.5, 1.0, 2.0, 3.0]);
        let p = Array1::from(vec![0.5, 0.5, 0.0, 2.0, 4.0, 1.0]);
        let mut alist = vec![-1.0, 0.0, 2.0, 4.0, 5.0, 3.0];
        let mut flist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
        let (amin, amax, alp, abest, fbest, fmed) = (-1.0, 5.0, 2.0, 2.0, 3.0, 3.0);
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let monotone = true;
        let mut minima = vec![true; 1];
        let (nmin, unitlen, s) = (1, 1.0, 4);
        let saturated = false;

        let (alist_res, flist_res, amin_res, amax_res, alp_res, abest_res, fbest_res, fmed_res, monotone_res, nmin_res, unitlen_res, s_res, good, saturated_res)
            = lsquart(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        let expected_alist = vec![-1.0, 0.0, 2.0, 3.0, 4.0, 5.0];
        let expected_flist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];

        for (a_exp, a_res) in expected_alist.iter().zip(alist_res.iter()) {
            assert_relative_eq!(a_exp, a_res);
        }

        for (f_exp, f_res) in expected_flist.iter().zip(flist_res.iter()) {
            assert_relative_eq!(f_exp, f_res);
        }

        assert_eq!(up, Vec::from([true, true, true]));
        assert_eq!(down, Vec::from([false, false, false]));
        assert_eq!(minima, Vec::from([true, false, false, false]));
        assert_eq!(monotone, true);
        assert_relative_eq!(alp_res, -0.6666666666666666);
        assert_eq!(nmin_res, 1);
        assert_relative_eq!(unitlen_res, 1.0);
        assert_eq!(s_res, 4);
        assert!(saturated_res);
    }

    #[test]
    fn test_2() {
        let nloc = 5;
        let small = 1e-10;
        let x = Array1::from(vec![1.0, 2.0, 1.0 + 1e-9, 1.0 + 2e-9, 1.0 + 3e-9, 1.0 + 4e-9]);
        let p = Array1::from(vec![0.5, 0.45, 5.5034, 0.5387, 5.8, 2.734]);
        let mut alist = vec![1.0, 1.0 + 1e-5, 1.0 + 2e-4, 1.0 + 3e-4, 3.0 + 4e-5, -2.0];
        let mut flist = vec![1.0, 1.1, 1.2, 1.3, 1.4, -2.0];
        let (amin, amax, alp, abest, fbest, fmed) = (0.0, 2.0, 1.0, 1.0, 1.0, 1.2);
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let monotone = true;
        let mut minima = vec![true; 1];
        let (nmin, unitlen, s) = (1, 1.0, 2);
        let saturated = true;

        let (alist_res, flist_res, amin_res, amax_res, alp_res, abest_res, fbest_res, fmed_res, monotone_res, nmin_res, unitlen_res, s_res, good, saturated_res)
            = lsquart(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        let expected_alist = vec![-2.0, 0.0, 1.0, 1.00001, 1.0002, 1.0003, 3.00004];
        let expected_flist = vec![1.0, -7.70489595691683e-12, 1.1, 1.2, 1.3, 1.4, -2.0];

        for (a_exp, a_res) in expected_alist.iter().zip(alist_res.iter()) {
            assert_relative_eq!(a_exp, a_res);
        }

        for (f_exp, f_res) in expected_flist.iter().zip(flist_res.iter()) {
            assert_relative_eq!(*f_exp, *f_res);
        }

        assert_eq!(up, Vec::from([false, true, true, true, true, false]));
        assert_eq!(down, Vec::from([true, false, false, false, false, true]));
        assert_eq!(minima, Vec::from([false, true, false, false, false, false, true]));
        assert_relative_eq!(amin_res, 0.0);
        assert_relative_eq!(amax_res, 2.0);
        assert_relative_eq!(alp_res, 0.0);
        assert!(!monotone_res);
    }

    #[test]
    fn test_3() {
        let nloc = 3;
        let small = 1e-5;
        let x = Array1::from(vec![1.0, 0.2325, 0.2, 1.0, 0.423, 0.4]);
        let p = Array1::from(vec![0.5, 0.3, 0.25, 1.45, 0.537, 1.24]);
        let mut alist = vec![10.101, 1.0, 2.0, 3.0, 2.21, 1.4];
        let mut flist = vec![1.0, 0.5, 2.0, 1.5, 1.01, 1.5];
        let (amin, amax, alp, abest, fbest, fmed) = (-1.0, 5.0, 2.0, 1.0, 0.5, 1.5);
        let mut up = vec![false, false, true, false];
        let mut down = vec![true, true, true, true];
        let monotone = false;
        let mut minima = vec![true, false];
        let (nmin, unitlen, s) = (2, 1.0, 3);
        let saturated = false;

        let (alist_res, flist_res, amin_res, amax_res, alp_res, abest_res, fbest_res, fmed_res, monotone_res, nmin_res, unitlen_res, s_res, good, saturated_res)
            = lsquart(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        let expected_alist = vec![1.0, 1.4, 2.0, 2.21, 3.0, 5.0, 10.101];
        let expected_flist = vec![0.5, 1.5, 2.0, 1.01, 1.5, -4.018396477612926e-236, 1.0];

        for (a_exp, a_res) in expected_alist.iter().zip(alist_res.iter()) {
            assert_relative_eq!(a_exp, a_res);
        }

        for (f_exp, f_res) in expected_flist.iter().zip(flist_res.iter()) {
            assert_relative_eq!(*f_exp, *f_res);
        }

        assert_eq!(up, Vec::from([true, true, false, true, false, true]));
        assert_eq!(down, Vec::from([false, false, true, false, true, false]));
        assert_eq!(minima, Vec::from([true, false, false, true, false, true, false]));
        assert_relative_eq!(amin_res, -1.0);
        assert_relative_eq!(amax_res, 5.0);
        assert_relative_eq!(alp_res, 5.0);
        assert_eq!(nmin_res, 3);
        assert_relative_eq!(unitlen_res, 2.79);
        assert_eq!(s_res, 7);
    }
}

