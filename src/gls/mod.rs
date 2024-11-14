mod lsrange;
mod lsinit;
mod lssort;
mod lssplit;
mod lsnew;
mod lsguard;
mod lspar;
mod quartic;
mod lslocal;
mod lsquart;
mod lsdescent;
mod lsconvex;
mod lssat;
mod lssep;


use lsconvex::lsconvex;
use lsdescent::lsdescent;
use lsinit::lsinit;
use lslocal::lslocal;
use lsnew::lsnew;
use lspar::lspar;
use lsquart::lsquart;
use lsrange::lsrange;
use lssat::lssat;
use lssep::lssep;
use lssort::lssort;

use ndarray::Array1;


pub fn gls(
    x: &Array1<f64>,
    p: &Array1<f64>,
    mut alist: &mut Vec<f64>,
    mut flist: &mut Vec<f64>,
    xl: &Array1<f64>,
    xu: &Array1<f64>,
    nloc: i32,
    small: f64,
    smax: usize,
) -> (
    Vec<f64>, //alist
    Vec<f64>, //flist
    usize //nf
) {

    // If alist is scalar, initialize it with a single value list
    if alist.len() == 1 {
        *alist = vec![alist[0]];
        *flist = vec![flist[0]];
    }

    // Golden section fraction is (3 - sqrt(5)) / 2
    let short = 0.381966;

    // Save information for nf computation and extrapolation decision
    let sinit = alist.len();  // Initial list size

    // Get 5 starting points (needed for lslocal)
    let bend = false;

    // Find range of useful alp
    let (mut amin, mut amax, scale) = lsrange(&x, &p, xl, xu, bend).unwrap();

    // Call `lsinit` to get initial alist, flist, etc.
    let (mut alp, _, _, _) = lsinit(&x, &p, alist, flist, amin, amax, scale);

    // Sort alist and flist and get relevant values
    let (mut alist, mut flist, mut abest, mut fbest, mut fmed, mut up, mut down, mut monotone, mut minima, mut nmin, mut unitlen, mut s) = lssort(alist, flist);

    // Initialize number of functions used
    let mut nf = s - sinit;

    // The main search loop
    while s < std::cmp::min(5, smax) {
        if nloc == 1 {
            // Parabolic interpolation step
            let (new_alist, new_flist, new_abest, new_fbest, new_fmed, new_up, new_down, new_monotone, new_minima, new_nmin, new_unitlen, new_s, new_alp, _) = lspar(
                nloc, small, sinit as i32, short, &x, &p, &mut alist, &mut flist, amin, amax,
                alp, abest, fmed, unitlen, s,
            );

            // Update the variables based on the result
            alist = new_alist;
            flist = new_flist;
            abest = new_abest;
            fbest = new_fbest;
            fmed = new_fmed;
            up = new_up;
            down = new_down;
            monotone = new_monotone;
            minima = new_minima;
            nmin = new_nmin;
            unitlen = new_unitlen;
            s = new_s;
            alp = new_alp;

            // If no true parabolic step has been done and it's monotonic
            if s > 3 && monotone && (abest == amin || abest == amax) {
                nf = s - sinit;
                return (alist, flist, nf);
            }
        } else {
            // Extrapolation or split
            let (new_alp, _) = lsnew(
                nloc, small, sinit as i32, short, &x, &p, s, &mut alist, &mut flist, amin,
                amax, abest, fmed, unitlen,
            );
            alp = new_alp;

            let sorted_result = lssort(&mut alist, &mut flist);
            alist = sorted_result.0;
            flist = sorted_result.1;
            abest = sorted_result.2;
            fbest = sorted_result.3;
            fmed = sorted_result.4;
            up = sorted_result.5;
            down = sorted_result.6;
            monotone = sorted_result.7;
            minima = sorted_result.8;
            nmin = sorted_result.9;
            unitlen = sorted_result.10;
            s = sorted_result.11;
        }
    }

    let mut saturated = false;
    if nmin == 1 {
        if monotone && (abest == amin || abest == amax) {
            nf = s - sinit;
            return (alist, flist, nf);
        }

        // Try quartic interpolation step if s == 5
        if s == 5 {
            let (new_alist, new_flist, new_amin, new_amax, new_alp,
                new_abest, new_fbest, new_fmed, new_monotone, new_nmin,
                new_unitlen, new_s, _, new_saturated
            ) = lsquart(
                nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest,
                fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated,
            );

            alist = new_alist;
            flist = new_flist;
            amin = new_amin;
            amax = new_amax;
            alp = new_alp;
            abest = new_abest;
            fbest = new_fbest;
            fmed = new_fmed;
            monotone = new_monotone;
            nmin = new_nmin;
            unitlen = new_unitlen;
            s = new_s;
            saturated = new_saturated;
        }

        // Check the descent condition
        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) = lsdescent(
            &x, &p, &mut alist, &mut flist, alp, abest, fbest, fmed, &mut up,
            &mut down, monotone, &mut minima, nmin, unitlen, s,
        );

        alp = new_alp;
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        monotone = new_monotone;
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;

        // Check convexity condition
        if lsconvex(&alist, &flist, nmin, s) {
            nf = s - sinit;
            return (alist, flist, nf);
        }
    }

    let mut sold = 0;
    'inner: loop {
        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) = lsdescent(
            &x, &p, &mut alist, &mut flist, alp, abest, fbest, fmed, &mut up,
            &mut down, monotone, &mut minima, nmin, unitlen, s);
        alp = new_alp;
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        monotone = new_monotone;
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;

        // Check saturation
        let (new_alp, is_saturated) = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        alp = new_alp;
        saturated = is_saturated;

        if saturated || s == sold || s >= smax {
            break 'inner;
        }

        sold = s;
        let nminold = nmin;

        if !saturated && nloc > 1 {
            let sep_result = lssep(
                nloc, small, sinit as i32, short, &x, &p, &mut alist, &mut flist, amin,
                amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone,
                &mut minima, nmin, unitlen, s,
            );

            let (new_amin, new_amax, new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) = sep_result;
            amin = new_amin;
            amax = new_amax;
            abest = new_abest;
            alp = new_alp;
            fbest = new_fbest;
            fmed = new_fmed;
            monotone = new_monotone;
            nmin = new_nmin;
            unitlen = new_unitlen;
            s = new_s;
        }

        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s, new_saturated) = lslocal(
            nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest,
            fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);
        alp = new_alp;
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        monotone = new_monotone;
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;
        saturated = new_saturated;

        if nmin > nminold { saturated = false }
    }

    return (alist, flist, s - sinit);
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_minimum_valid_input() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let xu = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let nloc = 1;
        let small = 1e-5;
        let smax = 10;

        let (alist_result, flist_result, nf) = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![-1.0, -0.618034, -0.381966025156, 0.0];
        let expected_flist = vec![
            -0.00508911288366444,
            -1.871372446840273e-18,
            -1.902009314142582e-62,
            -3.391970967769076e-191,
        ];
        let expected_nf = 4;

        for (a, a_exp) in alist_result.iter().zip(expected_alist.iter()) {
            assert_abs_diff_eq!(a, a_exp, epsilon = 1e-15);
        }

        for (f, f_exp) in flist_result.iter().zip(expected_flist.iter()) {
            assert_abs_diff_eq!(f, f_exp, epsilon = 1e-15);
        }

        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_small_vector_values() {
        let x = Array1::from_vec(vec![-1e-8, 1e-8, -1e-8, 1e-8, -1e-8, 1e-8]);
        let p = Array1::from_vec(vec![1e-8, -1e-8, 1e-8, -1e-8, 1e-8, -1e-8]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = Array1::from_vec(vec![-1e-8, -1e-8, -1e-8, -1e-8, -1e-8, -1e-8]);
        let xu = Array1::from_vec(vec![1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]);
        let nloc = 1;
        let small = 1e-12;
        let smax = 5;

        let (alist_result, flist_result, nf) = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![0.0, 0.127322, 0.381966, 1.0];
        let expected_flist = vec![
            -0.00508911309741038,
            -0.00508911307019582,
            -0.005089113015766702,
            -0.00508911288366444,
        ];
        let expected_nf = 4;

        for (a, a_exp) in alist_result.iter().zip(expected_alist.iter()) {
            assert_abs_diff_eq!(a, a_exp, epsilon = 1e-15);
        }

        for (f, f_exp) in flist_result.iter().zip(expected_flist.iter()) {
            assert_abs_diff_eq!(f, f_exp, epsilon = 1e-15);
        }

        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_edge_case_maximum_search_steps() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = Array1::from_vec(vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let xu = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let nloc = 1;
        let small = 1e-5;
        let smax = 1;

        let (alist_result, flist_result, nf) = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![0.0, 1.0];
        let expected_flist = vec![
            -3.391970967769076e-191,
            -0.00508911288366444,
        ];
        let expected_nf = 2;

        for (a, a_exp) in alist_result.iter().zip(expected_alist.iter()) {
            assert_abs_diff_eq!(a, a_exp, epsilon = 1e-15);
        }

        for (f, f_exp) in flist_result.iter().zip(expected_flist.iter()) {
            assert_abs_diff_eq!(f, f_exp, epsilon = 1e-15);
        }

        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_empty_lists() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let xu = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let nloc = 1;
        let small = 1e-5;
        let smax = 10;

        let (alist_result, flist_result, nf) = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![-1.0, -0.618034, -0.381966025156, 0.0];
        let expected_flist = vec![
            -0.00508911288366444,
            -1.871372446840273e-18,
            -1.902009314142582e-62,
            -3.391970967769076e-191,
        ];
        let expected_nf = 4;

        for (a, a_exp) in alist_result.iter().zip(expected_alist.iter()) {
            assert_abs_diff_eq!(a, a_exp, epsilon = 1e-15);
        }

        for (f, f_exp) in flist_result.iter().zip(expected_flist.iter()) {
            assert_abs_diff_eq!(f, f_exp, epsilon = 1e-15);
        }

        assert_eq!(nf, expected_nf);
    }
}
