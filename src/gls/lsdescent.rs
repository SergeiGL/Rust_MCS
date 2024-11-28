use crate::feval::feval;
use crate::gls::lssort::lssort;

pub fn lsdescent(
    x: &[f64],
    p: &[f64],
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
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
) -> (f64, //alp
      f64, //abest
      f64, //fbest
      f64, //fmed
      bool, //monotone
      usize, //nmin
      f64, //unitlen
      usize //s
) {
    if alist.iter().any(|&i| i == 0.0) {
        fbest = flist.iter().cloned().fold(flist[0], f64::min);
        let i = flist.iter().position(|&f| f == fbest).unwrap();
        if alist[i] < 0.0 {
            if alist[i] >= 4.0 * alist[i + 1] {
                return (alp, abest, fbest, fmed, monotone, nmin, unitlen, s);
            }
        } else if alist[i] > 0.0 {
            if alist[i] < 4.0 * alist[i - 1] {
                return (alp, abest, fbest, fmed, monotone, nmin, unitlen, s);
            }
        } else {
            if i == 0 {
                fbest = flist[1];
            } else if i == s - 1 {
                fbest = flist[s - 2];
            } else {
                fbest = flist[i - 1].min(flist[i + 1]);
            }
        }

        if alist[i] != 0.0 {
            alp = alist[i] / 3.0;
        } else if i == s - 1 {
            alp = alist[s - 2] / 3.0;
        } else if i == 0 {
            alp = alist[1] / 3.0;
        } else {
            // Split wider adjacent interval.
            if alist[i + 1] - alist[i] > alist[i] - alist[i - 1] {
                alp = alist[i + 1] / 3.0;
            } else {
                alp = alist[i - 1] / 3.0;
            }
        }

        let falp = feval(&(x.iter().zip(p).map(|(xi, pi)| *xi + *pi * alp).collect::<Vec<f64>>()));

        // Insert the new alp and falp into the lists.
        alist.push(alp);
        flist.push(falp);

        // Call lssort and update necessary references.
        (abest, fbest, fmed, *up, *down, monotone, *minima, nmin, unitlen, s) = lssort(alist, flist);
    }

    (alp, abest, fbest, fmed, monotone, nmin, unitlen, s)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_0() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let mut alist = vec![3.0, 2.0];
        let mut flist = vec![3.0, 2.0];
        let alp = 0.0;
        let abest = 3.0;
        let fbest = flist[0];
        let fmed = flist[0];
        let mut up = vec![false, false, false, false, false, false];
        let mut down = vec![false, false, false, false, false, false];
        let monotone = true;
        let mut minima = vec![];
        let nmin = 1;
        let unitlen = 1.0;
        let s = 1;

        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) =
            lsdescent(
                &x, &p, &mut alist, &mut flist, alp, abest, fbest, fmed,
                &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
            );

        assert_eq!(alist, vec![3.0, 2.0]);
        assert_eq!(flist, vec![3.0, 2.0]);
        assert_eq!(new_alp, 0.0);
        assert_eq!(new_abest, 3.0);
        assert_eq!(new_fbest, 3.0);
        assert_eq!(new_fmed, 3.0);
        assert_eq!(up, vec![false, false, false, false, false, false]);
        assert_eq!(down, vec![false, false, false, false, false, false]);
        assert_eq!(new_monotone, true);
        assert_eq!(minima, Vec::<bool>::new());
        assert_eq!(new_nmin, 1);
        assert_eq!(new_unitlen, 1.0);
        assert_eq!(new_s, 1);
    }

    #[test]
    fn test_case_1() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let p = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let mut alist = vec![1.0, 2.0, -1.0, 0.5];
        let mut flist = vec![5.0, 7.0, 3.0, 10.0];
        let alp = 1.5;
        let abest = 2.0;
        let fbest = 3.0;
        let fmed = 6.0;
        let mut up = vec![false, false, false, false, false, false];
        let mut down = vec![true, false, true, false, true, false];
        let monotone = false;
        let mut minima = vec![false, true];
        let nmin = 2;
        let unitlen = 1.0;
        let s = 4;

        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) =
            lsdescent(
                &x, &p, &mut alist, &mut flist, alp, abest, fbest, fmed,
                &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
            );

        assert_eq!(alist, vec![1.0, 2.0, -1.0, 0.5]);
        assert_eq!(flist, vec![5.0, 7.0, 3.0, 10.0]);
        assert_eq!(new_alp, 1.5);
        assert_eq!(new_abest, 2.0);
        assert_eq!(new_fbest, 3.0);
        assert_eq!(new_fmed, 6.0);
        assert_eq!(up, vec![false, false, false, false, false, false]);
        assert_eq!(down, vec![true, false, true, false, true, false]);
        assert_eq!(new_monotone, false);
        assert_eq!(minima, vec![false, true]);
        assert_eq!(new_nmin, 2);
        assert_eq!(new_unitlen, 1.0);
        assert_eq!(new_s, 4);
    }

    #[test]
    fn test_case_2() {
        let x = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let p = vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let mut alist = vec![-1.2, -3.4, -4.0, 2.5];
        let mut flist = vec![1.0, 4.5, 6.5, 2.0];
        let alp = 1.0;
        let abest = 1.5;
        let fbest = 1.0;
        let fmed = 3.25;
        let mut down = vec![true, false, true, false, true, false];
        let mut up = vec![false, false, false, false, false, false];
        let monotone = false;
        let mut minima = vec![];
        let nmin = 0;
        let unitlen = 1.0;
        let s = 4;

        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) =
            lsdescent(
                &x, &p, &mut alist, &mut flist, alp, abest, fbest, fmed,
                &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
            );

        assert_eq!(alist, vec![-1.2, -3.4, -4.0, 2.5]);
        assert_eq!(flist, vec![1.0, 4.5, 6.5, 2.0]);
        assert_eq!(new_alp, 1.0);
        assert_eq!(new_abest, 1.5);
        assert_eq!(new_fbest, 1.0);
        assert_eq!(new_fmed, 3.25);
        assert_eq!(up, vec![false, false, false, false, false, false]);
        assert_eq!(down, vec![true, false, true, false, true, false]);
        assert_eq!(new_monotone, false);
        assert_eq!(minima, vec![]);
        assert_eq!(new_nmin, 0);
        assert_eq!(new_unitlen, 1.0);
        assert_eq!(new_s, 4);
    }


    #[test]
    fn test_real_mistake_0() {
        let x = vec![0.2, 0.2, 0.45, 0.56, 0.68, 0.72];
        let p = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut alist = vec![-0.2, -0.1, -0.02602472805313486, 0.0, 0.048253975355972145, 0.2];
        let mut flist = vec![-0.3098962997361745, -0.35807529391557985, -0.36128396643179006, -2.7, -0.33610446976533986, -0.23322360512233206];
        let alp = -0.02602472805313486;
        let abest = 0.0;
        let fbest = -2.7;
        let fmed = -0.34708988184045986;
        let mut down = vec![false, false, false, true, true];
        let mut up = vec![true, true, true, false, false];
        let monotone = false;
        let mut minima = vec![false, false, false, true, false, false];
        let nmin = 1;
        let unitlen = 0.2;
        let s = 6;

        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) =
            lsdescent(
                &x, &p, &mut alist, &mut flist, alp, abest, fbest, fmed,
                &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
            );

        assert_eq!(alist, vec![-0.2, -0.1, -0.02602472805313486, 0.0, 0.016084658451990714, 0.048253975355972145, 0.2]);
        assert_eq!(flist, vec![-0.3098962997361745, -0.35807529391557985, -0.36128396643179006, -2.7, -0.3501457040105073, -0.33610446976533986, -0.23322360512233206]);
        assert_eq!(new_alp, 0.016084658451990714);
        assert_eq!(new_abest, 0.0);
        assert_eq!(new_fbest, -2.7);
        assert_eq!(new_fmed, -0.3501457040105073);
        assert_eq!(up, vec![false, false, false, true, true, true]);
        assert_eq!(down, vec![true, true, true, false, false, false]);
        assert_eq!(new_monotone, false);
        assert_eq!(minima, vec![false, false, false, true, false, false, false]);
        assert_eq!(new_nmin, 1);
        assert_eq!(new_unitlen, 0.2);
        assert_eq!(new_s, 7);
    }
}
