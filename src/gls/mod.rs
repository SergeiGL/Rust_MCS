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


pub fn gls(
    x: &[f64],
    p: &[f64],
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    xl: &[f64],
    xu: &[f64],
    nloc: i32,
    small: f64,
    smax: usize,
) ->
    usize //nf
{
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
    let (mut abest, mut fbest, mut fmed, mut up, mut down, mut monotone, mut minima, mut nmin, mut unitlen, mut s) = lssort(alist, flist);


    // The main search loop
    while s < std::cmp::min(5, smax) {
        if nloc == 1 {
            // Parabolic interpolation step
            (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s, alp, _) =
                lspar(nloc, small, sinit as i32, short, &x, &p, alist, flist, amin, amax,
                      alp, abest, fmed, unitlen, s);
            // If no true parabolic step has been done and it's monotonic
            if s > 3 && monotone && (abest == amin || abest == amax) {
                return s - sinit;
            }
        } else {
            // Extrapolation or split
            let (new_alp, _) = lsnew(
                nloc, small, sinit as i32, short, &x, &p, s, alist, flist, amin,
                amax, abest, fmed, unitlen,
            );
            alp = new_alp;

            (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(alist, flist);
        }
    }

    let mut saturated = false;
    if nmin == 1 {
        if monotone && (abest == amin || abest == amax) {
            return s - sinit;
        }
        // println!("1 {nloc}, {small}, {x:?}, {p:?} {alist:?}, {flist:?}, {amin:?}, {amax}, {alp}, {abest}, {fbest}, {fmed}, {up:?}, {down:?}, {monotone}, {minima:?}, {nmin}, {unitlen}, {s}, {saturated}");
        if s == 5 {
            (amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s, _, saturated) = lsquart(nloc, small, &x, &p, alist, flist, amin, amax, alp, abest, fbest,
                                                                                                      fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);
        }
        // println!("2 {alist:?}");

        // Check the descent condition
        (alp, abest, fbest, fmed, monotone, nmin, unitlen, s) = lsdescent(&x, &p, alist, flist, alp, abest, fbest, fmed, &mut up,
                                                                          &mut down, monotone, &mut minima, nmin, unitlen, s);

        // println!("3 {alist:?}");

        // Check convexity condition
        if lsconvex(&alist, &flist, nmin, s) {
            return s - sinit;
        }
    }
    // println!("{alist:?}");

    let mut sold = 0;
    'inner: loop {
        (alp, abest, fbest, fmed, monotone, nmin, unitlen, s) = lsdescent(
            &x, &p, alist, flist, alp, abest, fbest, fmed, &mut up,
            &mut down, monotone, &mut minima, nmin, unitlen, s);

        // Check saturation
        (alp, saturated) = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);

        if saturated || s == sold || s >= smax {
            break 'inner;
        }

        sold = s;
        let nminold = nmin;

        if !saturated && nloc > 1 {
            (amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s) =
                lssep(nloc, small, sinit as i32, short, &x, &p, alist, flist, amin,
                      amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone,
                      &mut minima, nmin, unitlen, s);
        }

        (alp, abest, fbest, fmed, monotone, nmin, unitlen, s, saturated) =
            lslocal(nloc, small, &x, &p, alist, flist, amin, amax, alp, abest,
                    fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        if nmin > nminold { saturated = false }
    }

    s - sinit
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimum_valid_input() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let xu = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let nloc = 1;
        let small = 1e-5;
        let smax = 10;

        let nf = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![-1.0, -0.618034, -0.381966025156, 0.0];
        let expected_flist = vec![
            -0.00508911288366444,
            -1.871372446840273e-18,
            -1.902009314142582e-62,
            -3.391970967769076e-191,
        ];
        let expected_nf = 4;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_small_vector_values() {
        let x = vec![-1e-8, 1e-8, -1e-8, 1e-8, -1e-8, 1e-8];
        let p = vec![1e-8, -1e-8, 1e-8, -1e-8, 1e-8, -1e-8];
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = vec![-1e-8, -1e-8, -1e-8, -1e-8, -1e-8, -1e-8];
        let xu = vec![1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8];
        let nloc = 1;
        let small = 1e-12;
        let smax = 5;

        let nf = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![0.0, 0.12732200000000002, 0.381966, 1.0];
        let expected_flist = vec![
            -0.00508911309741038,
            -0.00508911307019582,
            -0.005089113015766702,
            -0.00508911288366444,
        ];
        let expected_nf = 4;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_edge_case_maximum_search_steps() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let xu = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let nloc = 1;
        let small = 1e-5;
        let smax = 1;

        let nf = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![0.0, 1.0];
        let expected_flist = vec![
            -3.391970967769076e-191,
            -0.00508911288366444,
        ];
        let expected_nf = 2;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_empty_lists() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let xl = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let xu = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let nloc = 1;
        let small = 1e-5;
        let smax = 10;

        let nf = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![-1.0, -0.618034, -0.381966025156, 0.0];
        let expected_flist = vec![
            -0.00508911288366444,
            -1.871372446840273e-18,
            -1.902009314142582e-62,
            -3.391970967769076e-191,
        ];
        let expected_nf = 4;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }


    #[test]
    fn test_real_mistake_0() {
        let x = vec![0.2, 0.2, 0.45, 0.56, 0.68, 0.72];
        let p = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut alist = vec![0.0];
        let mut flist = vec![-2.7];
        let xl = vec![0.0; 6];
        let xu = vec![1.0; 6];
        let nloc = 1;
        let small = 0.1;
        let smax = 6;

        let nf = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);

        let expected_alist = vec![
            -0.2,
            -0.1,
            -0.02602472805313486,
            -0.008674909351044953,
            0.0,
            0.016084658451990714,
            0.048253975355972145,
            0.2];
        let expected_flist = vec![
            -0.3098962997361745,
            -0.35807529391557985,
            -0.36128396643179006,
            -0.3577687209054916,
            -2.7,
            -0.3501457040105073,
            -0.33610446976533986,
            -0.23322360512233206
        ];
        let expected_nf = 7;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }
    #[test]
    fn test_real_mistake_1() {
        let x = vec![0.190983, 0.6, 0.7, 0.8, 0.9, 1.0];
        let p = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut alist = vec![0.0];
        let mut flist = vec![-0.01523227097945881];
        let xl = vec![-1.0; 6];
        let xu = vec![1.0; 6];
        let nloc = 1;
        let small = 0.1;
        let smax = 6;

        let nf = gls(&x, &p, &mut alist, &mut flist, &xl, &xu, nloc, small, smax);


        let expected_alist = vec![
            -1.6,
            -1.1,
            -0.6,
            -0.40767007775845987,
            -0.30750741391479774,
            -0.10250247130493258,
            0.0
        ];

        let expected_flist = vec![
            -0.00033007085742366247,
            -0.005222762849281021,
            -0.019362461793975085,
            -0.02327501776110989,
            -0.023740401281451076,
            -0.019450678518268937,
            -0.01523227097945881
        ];
        let expected_nf = 6;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }
}
