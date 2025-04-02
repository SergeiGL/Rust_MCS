use crate::gls::lssort::lssort;
use nalgebra::SVector;

pub fn lsdescent<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
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
    if alist.iter().any(|&i| i == 0.0) {
        let i;

        (i, *fbest) = flist.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, &val)| (i, val))
            .unwrap();

        if alist[i] < 0.0 {
            if alist[i] >= 4.0 * alist[i + 1] {
                return;
            }
        } else if alist[i] > 0.0 {
            if alist[i] < 4.0 * alist[i - 1] {
                return;
            }
        }

        if alist[i] != 0.0 {
            *alp = alist[i] / 3.0;
        } else if i == *s - 1 {
            *alp = alist[*s - 2] / 3.0;
        } else if i == 0 {
            *alp = alist[1] / 3.0;
        } else {
            // Split wider adjacent interval.
            *alp = if alist[i + 1] - alist[i] > alist[i] - alist[i - 1] {
                alist[i + 1] / 3.0
            } else {
                alist[i - 1] / 3.0
            }
        }

        let falp = func(&(x + p.scale(*alp)));

        // Insert the new alp and falp into the lists.
        alist.push(*alp);
        flist.push(falp);

        (*abest, *fbest, *fmed, *up, *down, *monotone, *minima, *nmin, *unitlen, *s) = lssort(alist, flist);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_case_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        let mut alist = vec![3.0, 2.0];
        let mut flist = vec![3.0, 2.0];
        let mut alp = 0.0;
        let mut abest = 3.0;
        let mut fbest = flist[0];
        let mut fmed = flist[0];
        let mut up = vec![false, false, false, false, false, false];
        let mut down = vec![false, false, false, false, false, false];
        let mut monotone = true;
        let mut minima = vec![];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = 1;


        lsdescent(
            hm6, &x, &p, &mut alist, &mut flist, &mut alp, &mut abest, &mut fbest, &mut fmed,
            &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s,
        );

        assert_eq!(alist, vec![3.0, 2.0]);
        assert_eq!(flist, vec![3.0, 2.0]);
        assert_eq!(alp, 0.0);
        assert_eq!(abest, 3.0);
        assert_eq!(fbest, 3.0);
        assert_eq!(fmed, 3.0);
        assert_eq!(up, vec![false, false, false, false, false, false]);
        assert_eq!(down, vec![false, false, false, false, false, false]);
        assert_eq!(monotone, true);
        assert_eq!(minima, Vec::<bool>::new());
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 1.0);
        assert_eq!(s, 1);
    }

    #[test]
    fn test_case_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
        let mut alist = vec![1.0, 2.0, -1.0, 0.5];
        let mut flist = vec![5.0, 7.0, 3.0, 10.0];
        let mut alp = 1.5;
        let mut abest = 2.0;
        let mut fbest = 3.0;
        let mut fmed = 6.0;
        let mut up = vec![false, false, false, false, false, false];
        let mut down = vec![true, false, true, false, true, false];
        let mut monotone = false;
        let mut minima = vec![false, true];
        let mut nmin = 2;
        let mut unitlen = 1.0;
        let mut s = 4;


        lsdescent(
            hm6, &x, &p, &mut alist, &mut flist, &mut alp, &mut abest, &mut fbest, &mut fmed,
            &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s,
        );

        assert_eq!(alist, vec![1.0, 2.0, -1.0, 0.5]);
        assert_eq!(flist, vec![5.0, 7.0, 3.0, 10.0]);
        assert_eq!(alp, 1.5);
        assert_eq!(abest, 2.0);
        assert_eq!(fbest, 3.0);
        assert_eq!(fmed, 6.0);
        assert_eq!(up, vec![false, false, false, false, false, false]);
        assert_eq!(down, vec![true, false, true, false, true, false]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, true]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 1.0);
        assert_eq!(s, 4);
    }

    #[test]
    fn test_case_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
        let mut alist = vec![-1.2, -3.4, -4.0, 2.5];
        let mut flist = vec![1.0, 4.5, 6.5, 2.0];
        let mut alp = 1.0;
        let mut abest = 1.5;
        let mut fbest = 1.0;
        let mut fmed = 3.25;
        let mut down = vec![true, false, true, false, true, false];
        let mut up = vec![false, false, false, false, false, false];
        let mut monotone = false;
        let mut minima = vec![];
        let mut nmin = 0;
        let mut unitlen = 1.0;
        let mut s = 4;


        lsdescent(
            hm6, &x, &p, &mut alist, &mut flist, &mut alp, &mut abest, &mut fbest, &mut fmed,
            &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s,
        );

        assert_eq!(alist, vec![-1.2, -3.4, -4.0, 2.5]);
        assert_eq!(flist, vec![1.0, 4.5, 6.5, 2.0]);
        assert_eq!(alp, 1.0);
        assert_eq!(abest, 1.5);
        assert_eq!(fbest, 1.0);
        assert_eq!(fmed, 3.25);
        assert_eq!(up, vec![false, false, false, false, false, false]);
        assert_eq!(down, vec![true, false, true, false, true, false]);
        assert_eq!(monotone, false);
        assert!(minima.is_empty());
        assert_eq!(nmin, 0);
        assert_eq!(unitlen, 1.0);
        assert_eq!(s, 4);
    }


    #[test]
    fn test_real_mistake_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.45, 0.56, 0.68, 0.72]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-0.2, -0.1, -0.02602472805313486, 0.0, 0.048253975355972145, 0.2];
        let mut flist = vec![-0.3098962997361745, -0.35807529391557985, -0.36128396643179006, -2.7, -0.33610446976533986, -0.23322360512233206];
        let mut alp = -0.02602472805313486;
        let mut abest = 0.0;
        let mut fbest = -2.7;
        let mut fmed = -0.34708988184045986;
        let mut down = vec![false, false, false, true, true];
        let mut up = vec![true, true, true, false, false];
        let mut monotone = false;
        let mut minima = vec![false, false, false, true, false, false];
        let mut nmin = 1;
        let mut unitlen = 0.2;
        let mut s = 6;


        lsdescent(
            hm6, &x, &p, &mut alist, &mut flist, &mut alp, &mut abest, &mut fbest, &mut fmed,
            &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s,
        );

        assert_eq!(alist, vec![-0.2, -0.1, -0.02602472805313486, 0.0, 0.016084658451990714, 0.048253975355972145, 0.2]);
        assert_eq!(flist, vec![-0.3098962997361745, -0.35807529391557985, -0.36128396643179006, -2.7, -0.3501457040105073, -0.33610446976533986, -0.23322360512233206]);
        assert_eq!(alp, 0.016084658451990714);
        assert_eq!(abest, 0.0);
        assert_eq!(fbest, -2.7);
        assert_eq!(fmed, -0.3501457040105073);
        assert_eq!(up, vec![false, false, false, true, true, true]);
        assert_eq!(down, vec![true, true, true, false, false, false]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, false, false, true, false, false, false]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 0.2);
        assert_eq!(s, 7);
    }
}
