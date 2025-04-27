use crate::gls::lsnew::lsnew;
use crate::gls::lssort::lssort;
use nalgebra::SVector;

pub(super) fn lssep<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    nloc: usize,
    small: f64,
    sinit: usize,
    short: f64,
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
) {
    debug_assert!((down.len() == up.len()) && (down.len() == *s - 1));

    let mut nsep = 0;  // the original separation counter
    let mut ind = Vec::with_capacity(alist.len().max(*s) * 10);

    // Loop for separation points based on the differences in behavior
    while nsep < *nmin {
        debug_assert!(down.len() == up.len());

        // Find intervals where the behavior of adjacent intervals is opposite (monotonicity behavior switches)
        *down = (0..(*s - 1)).map(|i| flist[i + 1] < flist[i]).collect::<Vec<bool>>();

        // sep = ([1,1,down] & [0,up,0] & [down,1,1]) | ([1,1,up] & [0,down,0] & [up,1,1])
        // ind=find(sep);
        //
        // Note: down.len() == s-1
        // Note: that is s<=1, the code looks like this: ([1,1] & [0,0] & [1,1]) | ([1,1] & [0,0] & [1,1]) == ([0,0]) | ([0,0]) == [0,0] =>  ind=find(sep) is empty
        // we basically disregard first and last elements as they are always 0
        // But we need to keep enumeration as if the first element is still present (so +1)
        ind.clear();
        for i in 0..(*s - 1) {
            let prev_down = if i != 0 { down[i - 1] } else { true };
            let next_down = if i < down.len() - 1 { down[i + 1] } else { true };
            let current_up = up[i];
            let condition1 = prev_down && current_up && next_down;

            let prev_up = if i > 0 { up[i - 1] } else { true };
            let next_up = if i < up.len() - 1 { up[i + 1] } else { true };
            let current_down = down[i];
            let condition2 = prev_up && current_down && next_up;

            if condition1 || condition2 {
                ind.push(i + 1);
            }
        }

        if ind.is_empty() { break; }

        let aa = if ind.len() > nloc { // aa has len(aa) = len(ind)
            ind.sort_unstable_by(|&ind_i, &ind_j|
                flist[ind_i].min(flist[ind_i - 1]).total_cmp(
                    &flist[ind_j].min(flist[ind_j - 1]))
            );

            ind.iter().take(nloc).map(|&ind_i| 0.5 * (alist[ind_i] + alist[ind_i - 1]))
                .collect::<Vec<_>>()
        } else {
            ind.iter().map(|&ind_i| 0.5 * (alist[ind_i] + alist[ind_i - 1])).collect::<Vec<_>>()
        };

        // For each midpoint alp, evaluate the function and update lists
        for alp_i in aa {
            *alp = alp_i;
            let falp = func(&(x + p.scale(alp_i)));
            alist.push(alp_i);
            flist.push(falp);
            nsep += 1;
            if nsep >= *nmin {
                break;
            }
        }

        (*abest, *fbest, *fmed, *up, *down, *monotone, *minima, *nmin, *unitlen, *s) = lssort(alist, flist);
    }

    // instead of unnecessary separation, add some global points
    for _ in 0..(*nmin - nsep) {
        *alp = lsnew(func, nloc, small, sinit, short, x, p, *s, alist, flist, *amin, *amax, *abest, *fmed, *unitlen);
        (*abest, *fbest, *fmed, *up, *down, *monotone, *minima, *nmin, *unitlen, *s) = lssort(alist, flist);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_0() {
        // Matlab version of the test:
        // nloc = 1;
        // small = 1e-6;
        // sinit = 1;
        // short = 0.1;
        // x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // p = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        // alist = [3.0, 2.0, 2.0];
        // flist = [3.0, 2.0, 2.0];
        // amin = 0.0;
        // amax = 1.0;
        // alp = 1.5;
        // abest = 0.2;
        // fbest = 0.1;
        // fmed = 0.15;
        // up = [true, false];
        // down = [false, true];
        // monotone = false;
        // minima = [false, true];
        // nmin = 1;
        // unitlen = 0.1;
        // s = 3;
        //
        // lssep;
        //
        // disp(alist);
        // disp(flist);
        // disp(up);
        // disp(down);
        // disp(minima);
        // disp([amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s]);

        let nloc = 1;
        let small = 1e-6;
        let sinit = 1;
        let short = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        let mut alist = vec![3.0, 2.0, 2.0];
        let mut flist = vec![3.0, 2.0, 2.0];
        let mut amin = 0.0;
        let mut amax = 1.0;
        let mut alp = 1.5;
        let mut abest = 0.2;
        let mut fbest = 0.1;
        let mut fmed = 0.15;
        let mut up = vec![true, false];
        let mut down = vec![false, true];
        let mut monotone = false;
        let mut minima = vec![false, true];
        let mut nmin = 1;
        let mut unitlen = 0.1;
        let mut s = 3;

        lssep(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        assert_eq!(alist, vec![2., 2., 3., 1000003.]);
        assert_eq!(flist, vec![2., 2., 3., 0.]);
        assert_eq!(up, vec![false, true, false]);
        assert_eq!(down, vec![true, false, true]);
        assert_eq!(minima, vec![false, true, false, true]);
        assert_eq!((amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s), (0.0, 1.0, 1000003., 1000003., 0.0, 2.0, false, 2, 1000001., 4));
    }

    #[test]
    fn test_1() {
        // Matlab version of the test:
        // nloc = 2;
        // small = 1e-8;
        // sinit = 2;
        // short = 0.05;
        // x = [0.5, 1.5, 2.5, -3.5, 4.5, 5.5];
        // p = [0.5, -0.5, 0.5, 0.5, 0.5, 0.5];
        // alist = [0.0, -0.5, 1.0, -0.1, 0.5, -1.0, 2.0, -0.5, 1.0];
        // flist = [5.0, 3.0, -4.0, 5.0, 23.0, 1.1, 3.0, 4.0, -4.0];
        // amin = 0.0;
        // amax = 1.0;
        // alp = 0.0;
        // abest = 0.5;
        // fbest = 3.0;
        // fmed = 4.0;
        // up = [false, true, false, true, false, false, true, false];
        // down = [true, false, true, false, true, true, false, true];
        // monotone = false;
        // minima = [false, true, false];
        // nmin = 2;
        // unitlen = 0.5;
        // s = 9;
        //
        // lssep;
        //
        // disp(alist);
        // disp(flist);
        // disp(up);
        // disp(down);
        // disp(minima);
        // disp([amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s]);

        let nloc = 2;
        let small = 1e-8;
        let sinit = 2;
        let short = 0.05;
        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 1.5, 2.5, -3.5, 4.5, 5.5]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.5, -0.5, 0.5, 0.5, 0.5, 0.5]);
        let mut alist = vec![0.0, -0.5, 1.0, -0.1, 0.5, -1.0, 2.0, -0.5, 1.0];
        let mut flist = vec![5.0, 3.0, -4.0, 5.0, 23.0, 1.1, 3.0, 4.0, -4.0];
        let mut amin = 0.0;
        let mut amax = 1.0;
        let mut alp = 0.0;
        let mut abest = 0.5;
        let mut fbest = 3.0;
        let mut fmed = 4.0;
        let mut up = vec![false, true, false, true, false, false, true, false];
        let mut down = vec![true, false, true, false, true, true, false, true];
        let mut monotone = false;
        let mut minima = vec![false, true, false];
        let mut nmin = 2;
        let mut unitlen = 0.5;
        let mut s = 9;

        lssep(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        assert_eq!(alist, vec![-1.0, -0.5, -0.5, -0.375, -0.25, -0.1, 0., 0.25, 0.375, 0.5, 1., 1., 2., ]);
        assert_eq!(flist, vec![1.1000, 3., 4., -6.282368808737644e-134, -3.830914843431675e-137, 5., 5., -1.8124870068759153e-151, -2.0444760800101757e-155, 23., -4., -4., 3.]);
        assert_eq!(up, vec![true, true, false, true, true, false, false, true, true, false, false, true, ]);
        assert_eq!(down, vec![false, false, true, false, false, true, true, false, false, true, true, false, ]);
        assert_eq!(minima, vec![true, false, false, true, false, false, false, true, false, false, false, true, false, ]);
        assert_eq!((amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s), (0., 1., 0.375, 1.0, -4., 1.1, false, 4, 0.7500, 13));
    }


    #[test]
    fn test_2() {
        // Matlab version of the test:
        // nloc = 1;
        // small = 1e-6;
        // sinit = 1;
        // short = 0.1;
        // x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // alist = [0.0, 1.0];
        // flist = [2.0, 2.0];
        // amin = 0.0;
        // amax = 1.0;
        // alp = 0.5;
        // abest = 0.0;
        // fbest = 2.0;
        // fmed = 2.0;
        // up = [false];
        // down = [false];
        // monotone = true;
        // minima = [true, true];
        // nmin = 1;
        // unitlen = 1.0;
        // s = 2;
        //
        // lssep;
        //
        // disp(alist);
        // disp(flist);
        // disp(up);
        // disp(down);
        // disp(minima);
        // disp([amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s]);

        let nloc = 1;
        let small = 1e-6;
        let sinit = 1;
        let short = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);  // Zero direction vector
        let mut alist = vec![0.0, 1.0];
        let mut flist = vec![2.0, 2.0];  // Constant function values
        let mut amin = 0.0;
        let mut amax = 1.0;
        let mut alp = 0.5;
        let mut abest = 0.0;
        let mut fbest = 2.0;
        let mut fmed = 2.0;
        let mut up = vec![false];
        let mut down = vec![false];
        let mut monotone = true;  // Function is constant
        let mut minima = vec![true, true];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = 2;

        lssep(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        assert_eq!(alist, vec![0., 0.5000, 1.0000]);
        assert_eq!(flist, vec![2.0000, -3.391970967769076e-191, 2.0000]);
        assert_eq!(up, vec![false, true]);
        assert_eq!(down, vec![true, false]);
        assert_eq!(minima, vec![false, true, false]);
        assert_eq!((amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s), (0., 1., 0.5, 0.5, -3.391970967769076e-191, 2., false, 1, 0.5, 3));
    }


    #[test]
    fn test_3() {
        // Matlab version of the test:
        // nloc = 3;
        // small = 1e-7;
        // sinit = 3;
        // short = 0.2;
        // x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // p = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // alist = [0.0, 0.25, 0.5, 0.75, 1.0];
        // flist = [6.0, 4.0, 2.0, 4.0, 6.0];
        // amin = 0.0;
        // amax = 1.0;
        // alp = 0.0;
        // abest = 0.5;
        // fbest = 2.0;
        // fmed = 4.0;
        // up = [false, false, true, true];
        // down = [true, true, false, false];
        // monotone = false;
        // minima = [false, false, true, false, false];
        // nmin = 3;
        // unitlen = 0.25;
        // s = 5;
        //
        // lssep;
        //
        // disp(alist);
        // disp(flist);
        // disp(up);
        // disp(down);
        // disp(minima);
        // disp([amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s]);

        let nloc = 3;
        let small = 1e-7;
        let sinit = 3;
        let short = 0.2;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut alist = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut flist = vec![6.0, 4.0, 2.0, 4.0, 6.0];
        let mut amin = 0.0;
        let mut amax = 1.0;
        let mut alp = 0.0;
        let mut abest = 0.5;
        let mut fbest = 2.0;
        let mut fmed = 4.0;
        let mut up = vec![false, false, true, true];
        let mut down = vec![true, true, false, false];
        let mut monotone = false;
        let mut minima = vec![false, false, true, false, false];
        let mut nmin = 3;
        let mut unitlen = 0.25;
        let mut s = 5;

        lssep(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        assert_eq!(alist, vec![0., 0.16000000000000003, 0.2000, 0.2500, 0.4500, 0.5000, 0.7500, 1.0000, ]);
        assert_eq!(flist, vec![6.0000, -0.00000005684182795618631, -0.000000007754020958385428, 4.0000, -8.607772814964489e-16, 2.0000, 4.0000, 6.0000]);
        assert_eq!(up, vec![false, true, true, false, true, true, true, ]);
        assert_eq!(down, vec![true, false, false, true, false, false, false, ]);
        assert_eq!(minima, vec![false, true, false, false, true, false, false, false, ]);
        assert_eq!((amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s), (0., 1., 0.16000000000000003, 0.16000000000000003, -0.00000005684182795618631, 3., false, 2, 0.29, 8));
    }

    #[test]
    fn test_4() {
        // Matlab version of the test:
        // nloc = 1;
        // small = 1e-7;
        // sinit = 3;
        // short = 0.2;
        // x = [1.1, -1.2, 1.3, -1.4, -1.5, -1.6];
        // p = [-1.0, -2.0, -3.0, 4.0, 1.1, -1.0];
        // alist = [1.0, -1.25, 0.5, 0.5, -1.0, 3.];
        // flist = [-6.0, -24.0, -30.0, -40.0, -100., -102.];
        // amin = 0.0;
        // amax = 1.0;
        // alp = 0.0;
        // abest = 0.5;
        // fbest = 2.0;
        // fmed = 4.0;
        // up = [true, true, true, true, true];
        // down = [true, true, true, true, true];
        // monotone = false;
        // minima = [false, false, true, false, false];
        // nmin = 300;
        // unitlen = 0.25;
        // s = 6;
        //
        // lssep;
        //
        // disp(alist);
        // disp(flist);
        // disp(up);
        // disp(down);
        // disp(minima);
        // disp([amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s]);

        let nloc = 1;
        let small = 1e-7;
        let sinit = 3;
        let short = 0.2;
        let x = SVector::<f64, 6>::from_row_slice(&[1.1, -1.2, 1.3, -1.4, -1.5, -1.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, -2.0, -3.0, 4.0, 1.1, -1.0]);
        let mut alist = vec![1.0, -1.25, 0.5, 0.5, -1.0, 3.];
        let mut flist = vec![-6.0, -24.0, -30.0, -40.0, -100., -102.];
        let mut amin = 0.0;
        let mut amax = 1.0;
        let mut alp = 0.0;
        let mut abest = 0.5;
        let mut fbest = 2.0;
        let mut fmed = 4.0;
        let mut up = vec![true, true, true, true, true];
        let mut down = vec![true, true, true, true, true];
        let mut monotone = false;
        let mut minima = vec![false, false, true, false, false];
        let mut nmin = 300;
        let mut unitlen = 0.25;
        let mut s = 6;

        lssep(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        assert_eq!(alist, vec![-1.250000000000000, -1., 0.5, 0.5, 1., 1., 2., 2.5, 3.]);
        assert_eq!(flist, vec![-24., -100., -30., -40., -6., -2.6488386195657585e-84, -1.4509615388147326e-301, -0.0, -102.0]);
        assert_eq!(up, vec![false, true, false, true, true, true, true, false, ]);
        assert_eq!(down, vec![true, false, true, false, false, false, false, true, ]);
        assert_eq!(minima, vec![false, true, false, true, false, false, false, false, true, ]);
        assert_eq!((amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s), (0., 1., 2.5, 3., -102., -24., false, 3, 2.5, 9));
    }

    #[test]
    fn test_5() {
        // Matlab version of the test:
        // nloc = 1;
        // small = 1e-7;
        // sinit = 4;
        // short = 0.2;
        // x = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6];
        // p = [-1.0, -2.0, -3.0, -4.0, -1.1, -1.0];
        // alist = [-6.1, 6.1, 5.1, 4.1, -0.2, -1.1];
        // flist = [-6.0, 6.0, 5.0, 4.0, -0.1, -1.];
        // amin = 4.0;
        // amax = 23.0;
        // alp = -2.0;
        // abest = 1.5;
        // fbest = 3.0;
        // fmed = 4.1;
        // up = [true, true, true, true, true];
        // down = [true, true, true, true, true];
        // monotone = false;
        // minima = [false, false, true, false, false];
        // nmin = 300;
        // unitlen = 0.25;
        // s = 6;
        //
        // lssep;
        //
        //
        // disp(alist);
        // disp(flist);
        // disp(up);
        // disp(down);
        // disp(minima);
        // disp([amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s]);

        let nloc = 1;
        let small = 1e-7;
        let sinit = 4;
        let short = 0.2;
        let x = SVector::<f64, 6>::from_row_slice(&[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, -2.0, -3.0, -4.0, -1.1, -1.0]);
        let mut alist = vec![-6.1, 6.1, 5.1, 4.1, -0.2, -1.1];
        let mut flist = vec![-6.0, 6.0, 5.0, 4.0, -0.1, -1.];
        let mut amin = 4.0;
        let mut amax = 23.0;
        let mut alp = -2.0;
        let mut abest = 1.5;
        let mut fbest = 3.0;
        let mut fmed = 4.1;
        let mut up = vec![true, true, true, true, true];
        let mut down = vec![true, true, true, true, true];
        let mut monotone = false;
        let mut minima = vec![false, false, true, false, false];
        let mut nmin = 300;
        let mut unitlen = 0.25;
        let mut s = 6;

        lssep(hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        assert_eq!(alist, vec![-6.1, -1.1, -0.2, 0., 4.1, 5.1, 6.1]);
        assert_eq!(flist, vec![-6., -1., -0.1, -5.0547897966598666e-14, 4., 5., 6.]);
        assert_eq!(up, vec![true, true, true, true, true, true, ]);
        assert_eq!(down, vec![false, false, false, false, false, false, ]);
        assert_eq!(minima, vec![true, false, false, false, false, false, false, ]);
        assert_eq!((amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s), (4., 23., 0., -6.1, -6., -5.0547897966598666e-14, true, 1, 12.199999999999999, 7));
    }
}

