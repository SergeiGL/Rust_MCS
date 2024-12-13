use crate::feval::feval;
use crate::gls::lssplit::lssplit;
use nalgebra::SVector;

pub fn lsnew<const N: usize>(
    nloc: usize,
    small: f64,
    sinit: usize,
    short: f64,
    x: &[f64; N],
    p: &SVector<f64, N>,
    s: usize,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    abest: f64,
    fmed: f64,
    unitlen: f64,
) -> (
    f64,  // alp
    f64   // fac
) {
    let leftok = if alist[0] <= amin {
        false
    } else if flist[0] >= fmed.max(flist[1]) {
        sinit == 1 || nloc > 1
    } else {
        true
    };

    let rightok = if alist[s - 1] >= amax {
        false
    } else if flist[s - 1] >= fmed.max(flist[s - 2]) {
        sinit == 1 || nloc > 1
    } else {
        true
    };

    let step = if sinit == 1 { s - 1 } else { 1 };
    let mut fac = short;

    let alp = if leftok && (flist[0] < flist[s - 1] || !rightok) {
        let al = alist[0] - (alist[step] - alist[0]) / small;
        amin.max(al)
    } else if rightok {
        let au = alist[s - 1] + (alist[s - 1] - alist[s - 1 - step]) / small;
        amax.min(au)
    } else {
        let lenth = alist[1..s].iter().zip(&alist[0..s - 1])
            .map(|(i, j)| i - j).collect::<Vec<f64>>();
        let dist = alist[1..s].iter()
            .zip(&alist[0..s - 1])
            .map(|(i, j)| f64::max(i - abest, f64::max(abest - j, unitlen)))
            .collect::<Vec<f64>>();
        let wid: Vec<f64> = lenth.iter().zip(dist.iter()).map(|(l, d)| l / d).collect();

        // Properly use enumerate to get index with max value
        let (i_max, _) = wid.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();

        let new_alp;
        (new_alp, fac) = lssplit(i_max, alist, flist, short).unwrap();
        new_alp
    };


    let falp = feval(&std::array::from_fn::<f64, N, _>(|i| x[i] + alp * p[i]));

    alist.push(alp);
    flist.push(falp);

    (alp, fac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsnew_basic() {
        let nloc = 2;
        let small = 0.1;
        let sinit = 1;
        let short = 0.5;
        let x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.66];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let s = 3;
        let mut alist = vec![0.0, 0.5, 1.0];
        let mut flist = vec![3.0, 2.0, 1.0];
        let amin = -1.0;
        let amax = 2.0;
        let abest = 0.5;
        let fmed = 2.5;
        let unitlen = 1.0;

        let (result_alp, result_fac) = lsnew(nloc, small, sinit, short, &x, &p, s,
                                             &mut alist, &mut flist, amin, amax, abest, fmed, unitlen,
        );

        assert_eq!(result_alp, 2.0);
        assert_eq!(result_fac, 0.5);
        assert_eq!(alist, vec![0.0, 0.5, 1.0, 2.0]);
        assert_eq!(flist, vec![3.0, 2.0, 1.0, -0.00033660405573826796]);
    }

    #[test]
    fn test_leftmost_boundary() {
        let nloc = 1;
        let small = 0.1;
        let sinit = 1;
        let short = 0.5;
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let s = 3;
        let mut alist = vec![-1.0, 0.0, 1.0];
        let mut flist = vec![4.0, 3.0, 2.0];
        let amin = -1.0;
        let amax = 2.0;
        let abest = 0.0;
        let fmed = 3.5;
        let unitlen = 1.0;

        let (result_alp, result_fac) = lsnew(nloc, small, sinit, short, &x, &p, s,
                                             &mut alist, &mut flist, amin, amax, abest, fmed, unitlen,
        );

        assert_eq!(result_alp, 2.0);
        assert_eq!(result_fac, 0.5);
        assert_eq!(alist, vec![-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(flist, vec![4.0, 3.0, 2.0, -6.021808044063958e-206]);
    }

    #[test]
    fn test_rightmost_boundary() {
        let nloc = 1;
        let small = 0.1;
        let sinit = 1;
        let short = 0.5;
        let x = [0.2, 0.4, 0.6, 0.8, 1.0, 0.42];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let s = 3;
        let mut alist = vec![0.5, 1.5, 2.0];
        let mut flist = vec![1.0, 0.5, 0.3];
        let amin = 0.0;
        let amax = 2.0;
        let abest = 1.0;
        let fmed = 0.75;
        let unitlen = 1.0;

        let (result_alp, result_fac) = lsnew(nloc, small, sinit, short, &x, &p, s,
                                             &mut alist, &mut flist, amin, amax, abest, fmed, unitlen,
        );

        assert_eq!(result_alp, 0.0);
        assert_eq!(result_fac, 0.5);
        assert_eq!(alist, vec![0.5, 1.5, 2.0, 0.0]);
        assert_eq!(flist, vec![1.0, 0.5, 0.3, -0.08558359712218677]);
    }

    #[test]
    fn test_no_extrapolation() {
        let nloc = 2;
        let small = 0.05;
        let sinit = 1;
        let short = 0.6;
        let x = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65];
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let s = 2;
        let mut alist = vec![-10.0, 1.2, 9.0];
        let mut flist = vec![2.0, 1.8, 10.0];
        let amin = 0.1;
        let amax = 1.0;
        let abest = 0.15;
        let fmed = 1.9;
        let unitlen = 1.0;

        let (result_alp, result_fac) = lsnew(nloc, small, sinit, short, &x, &p, s,
                                             &mut alist, &mut flist, amin, amax, abest, fmed, unitlen,
        );

        assert_eq!(result_alp, -5.5200000000000005);
        assert_eq!(result_fac, 0.4);
        assert_eq!(alist, vec![-10.0, 1.2, 9.0, -5.5200000000000005]);
        assert_eq!(flist, vec![2.0, 1.8, 10.0, -6.269043201318317e-79])
    }

    #[test]
    fn test_no_extrapolation_2() {
        let nloc = 2;
        let small = 0.1;
        let sinit = 1;
        let short = 0.7;
        let x = [0.5, 0.3, 0.2, 0.4, 0.1, 0.6];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let s = 4;
        let mut alist = vec![0.1, 0.3, 1.5, 1.7];
        let mut flist = vec![1.9, 1.4, 1.2, 1.0];
        let amin = 0.1;
        let amax = 0.9;
        let abest = 0.5;
        let fmed = 1.5;
        let unitlen = 1.0;

        let (result_alp, result_fac) = lsnew(nloc, small, sinit, short, &x, &p, s,
                                             &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, 0.66);
        assert_eq!(result_fac, 0.30000000000000004);
        assert_eq!(alist, vec![0.1, 0.3, 1.5, 1.7, 0.66]);
        assert_eq!(flist, vec![1.9, 1.4, 1.2, 1.0, -0.005635405914473198]);
    }


    #[test]
    fn test_sinit_0() {
        let nloc = 1;
        let small = 0.1;
        let sinit = 0;
        let short = 0.5;
        let x = [0.2, 0.4, 0.6, 0.8, 1.0, 0.42];
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let s = 3;
        let mut alist = vec![0.5, 1.5, 2.0];
        let mut flist = vec![1.0, 0.5, 0.3];
        let amin = 0.0;
        let amax = 2.0;
        let abest = 1.0;
        let fmed = 0.75;
        let unitlen = 1.0;

        let (result_alp, result_fac) = lsnew(nloc, small, sinit, short, &x, &p, s,
                                             &mut alist, &mut flist, amin, amax, abest, fmed, unitlen);

        assert_eq!(result_alp, 1.0);
        assert_eq!(result_fac, 0.5);
        assert_eq!(alist, vec![0.5, 1.5, 2.0, 1.0]);
        assert_eq!(flist, vec![1.0, 0.5, 0.3, -3.674762767159534e-09]);
    }
}