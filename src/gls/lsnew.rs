use crate::gls::lssplit::lssplit;
use nalgebra::SVector;

/// find one new point by extrapolation or split of wide interval
pub fn lsnew<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    nloc: usize,
    small: f64,
    sinit: usize,
    short: f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    s: usize,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    abest: f64,
    fmed: f64,
    unitlen: f64,
) ->
    f64 // alp
{
    let fmed_max_f1 = fmed.max(flist[1]);
    let condition = sinit == 1 || nloc > 1;
    let last_idx = s - 1;

    #[inline]
    const fn check_boundary(val: bool, f: f64, fmax: f64, cond: bool) -> bool {
        val && (f < fmax || cond)
    }

    // Compute leftok and rightok using optimized checks
    let leftok = check_boundary(alist[0] > amin, flist[0], fmed_max_f1, condition);

    let rightok = check_boundary(alist[last_idx] < amax, flist[last_idx], fmed.max(flist[s - 2]), condition);

    let step = if sinit == 1 { last_idx } else { 1 };

    let alp = if leftok && (flist[0] < flist[last_idx] || !rightok) {
        // Left branch optimization
        let al = alist[0] - (alist[step] - alist[0]) / small;
        amin.max(al)
    } else if rightok {
        // Right branch optimization
        let au = alist[last_idx] + (alist[last_idx] - alist[last_idx - step]) / small;
        amax.min(au)
    } else {
        // Single-pass optimization: Combine the windows iteration and max finding
        let mut max_ratio = f64::NEG_INFINITY;
        let mut i_max = 0;

        // Process windows and track maximum in a single pass
        for (i, window) in alist.windows(2).enumerate() {
            let length = window[1] - window[0];
            let d = f64::max(
                window[1] - abest,
                f64::max(abest - window[0], unitlen),
            );

            // Track maximum ratio during iteration
            let ratio = length / d;
            if ratio.total_cmp(&max_ratio) == std::cmp::Ordering::Greater {
                max_ratio = ratio;
                i_max = i;
            }
        }

        lssplit(alist[i_max], alist[i_max + 1], flist[i_max], flist[i_max + 1], short).0
    };

    let falp = func(&(x + p.scale(alp)));

    alist.push(alp);
    flist.push(falp);

    alp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_lsnew_basic() {
        let nloc = 2;
        let small = 0.1;
        let sinit = 1;
        let short = 0.5;
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.66]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let s = 3;
        let mut alist = vec![0.0, 0.5, 1.0];
        let mut flist = vec![3.0, 2.0, 1.0];
        let amin = -1.0;
        let amax = 2.0;
        let abest = 0.5;
        let fmed = 2.5;
        let unitlen = 1.0;

        let result_alp = lsnew(hm6, nloc, small, sinit, short, &x, &p, s,
                               &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, 2.0);
        assert_eq!(alist, vec![0.0, 0.5, 1.0, 2.0]);
        assert_eq!(flist, vec![3.0, 2.0, 1.0, -0.00033660405573826796]);
    }

    #[test]
    fn test_leftmost_boundary() {
        let nloc = 1;
        let small = 0.1;
        let sinit = 1;
        let short = 0.5;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let s = 3;
        let mut alist = vec![-1.0, 0.0, 1.0];
        let mut flist = vec![4.0, 3.0, 2.0];
        let amin = -1.0;
        let amax = 2.0;
        let abest = 0.0;
        let fmed = 3.5;
        let unitlen = 1.0;

        let result_alp = lsnew(hm6, nloc, small, sinit, short, &x, &p, s,
                               &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, 2.0);
        assert_eq!(alist, vec![-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(flist, vec![4.0, 3.0, 2.0, -6.021808044063958e-206]);
    }

    #[test]
    fn test_rightmost_boundary() {
        let nloc = 1;
        let small = 0.1;
        let sinit = 1;
        let short = 0.5;
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.4, 0.6, 0.8, 1.0, 0.42]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let s = 3;
        let mut alist = vec![0.5, 1.5, 2.0];
        let mut flist = vec![1.0, 0.5, 0.3];
        let amin = 0.0;
        let amax = 2.0;
        let abest = 1.0;
        let fmed = 0.75;
        let unitlen = 1.0;

        let result_alp = lsnew(hm6, nloc, small, sinit, short, &x, &p, s,
                               &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, 0.0);
        assert_eq!(alist, vec![0.5, 1.5, 2.0, 0.0]);
        assert_eq!(flist, vec![1.0, 0.5, 0.3, -0.08558359712218677]);
    }

    #[test]
    fn test_no_extrapolation() {
        let nloc = 2;
        let small = 0.05;
        let sinit = 1;
        let short = 0.6;
        let x = SVector::<f64, 6>::from_row_slice(&[0.15, 0.25, 0.35, 0.45, 0.55, 0.65]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let s = 2;
        let mut alist = vec![-10.0, 1.2, 9.0];
        let mut flist = vec![2.0, 1.8, 10.0];
        let amin = 0.1;
        let amax = 1.0;
        let abest = 0.15;
        let fmed = 1.9;
        let unitlen = 1.0;

        let result_alp = lsnew(hm6, nloc, small, sinit, short, &x, &p, s,
                               &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, -5.5200000000000005);
        assert_eq!(alist, vec![-10.0, 1.2, 9.0, -5.5200000000000005]);
        assert_eq!(flist, vec![2.0, 1.8, 10.0, -6.269043201318317e-79])
    }

    #[test]
    fn test_no_extrapolation_2() {
        let nloc = 2;
        let small = 0.1;
        let sinit = 1;
        let short = 0.7;
        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 0.3, 0.2, 0.4, 0.1, 0.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let s = 4;
        let mut alist = vec![0.1, 0.3, 1.5, 1.7];
        let mut flist = vec![1.9, 1.4, 1.2, 1.0];
        let amin = 0.1;
        let amax = 0.9;
        let abest = 0.5;
        let fmed = 1.5;
        let unitlen = 1.0;

        let result_alp = lsnew(hm6, nloc, small, sinit, short, &x, &p, s,
                               &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, 0.66);
        assert_eq!(alist, vec![0.1, 0.3, 1.5, 1.7, 0.66]);
        assert_eq!(flist, vec![1.9, 1.4, 1.2, 1.0, -0.005635405914473198]);
    }


    #[test]
    fn test_sinit_0() {
        let nloc = 1;
        let small = 0.1;
        let sinit = 0;
        let short = 0.5;
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.4, 0.6, 0.8, 1.0, 0.42]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let s = 3;
        let mut alist = vec![0.5, 1.5, 2.0];
        let mut flist = vec![1.0, 0.5, 0.3];
        let amin = 0.0;
        let amax = 2.0;
        let abest = 1.0;
        let fmed = 0.75;
        let unitlen = 1.0;

        let result_alp = lsnew(hm6, nloc, small, sinit, short, &x, &p, s,
                               &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, 1.0);
        assert_eq!(alist, vec![0.5, 1.5, 2.0, 1.0]);
        assert_eq!(flist, vec![1.0, 0.5, 0.3, -3.674762767159534e-09]);
    }
}