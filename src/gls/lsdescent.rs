use crate::feval::feval;
use crate::gls::lssort::lssort;
use ndarray::Array1;

pub fn lsdescent(
    x: &Array1<f64>,
    p: &Array1<f64>,
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
    let cont = alist.iter().any(|&i| i == 0.0);

    let i = flist.iter().position(|&f| f == fbest).unwrap();

    if cont {
        // Find current fbest (smallest flist value) and retrieve corresponding index `i`.
        fbest = flist.iter().cloned().fold(flist[0], f64::min);

        if alist[i] < 0.0 {
            if i < s - 1 && alist[i] >= 4.0 * alist[i + 1] {
                // Terminate if condition holds for negative alist value.
                return (alp, abest, fbest, fmed, monotone, nmin, unitlen, s);
            }
        } else if alist[i] > 0.0 {
            if i > 0 && alist[i] < 4.0 * alist[i - 1] {
                // Terminate if condition holds for positive alist value.
                return (alp, abest, fbest, fmed, monotone, nmin, unitlen, s);
            }
        } else {
            // Handle alist[i] = 0 conditions.
            if i == 0 {
                fbest = flist[1];
            } else if i == s - 1 {
                fbest = flist[s - 2];
            } else {
                fbest = flist[i - 1].min(flist[i + 1]);
            }
        }
    }

    if cont {
        // Force a local descent step by adjusting `alp`.
        if alist[alist.len() - 1] != 0.0 {
            alp = alist.last().unwrap() / 3.0;
        } else if s == 1 {
            alp = alist[s - 2] / 3.0;
        } else if s == 0 {
            alp = alist[1] / 3.0;
        } else {
            // Split wider adjacent interval.
            if alist[i + 1] - alist[i] > alist[i] - alist[i - 1] {
                alp = alist[i + 1] / 3.0;
            } else {
                alp = alist[i - 1] / 3.0;
            }
        }

        // Evaluate the function at the new step alp.
        let new_x = x + &(alp * p);
        let falp = feval(&new_x);

        // Insert the new alp and falp into the lists.
        alist.push(alp);
        flist.push(falp);

        // Call lssort and update necessary references.
        let (sorted_alist, sorted_flist, new_abest, new_fbest, new_fmed, new_up, new_down, new_monotone, new_minima, new_nmin, new_unitlen, new_s) =
            lssort(alist, flist);

        alist.clone_from(&sorted_alist); // inplace update of alist
        flist.clone_from(&sorted_flist); // inplace update of flist
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        *up = new_up; // inplace update of up
        *down = new_down; // inplace update of down
        monotone = new_monotone;
        *minima = new_minima.clone(); // inplace update of minima
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;
    }

    (alp, abest, fbest, fmed, monotone, nmin, unitlen, s)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_case_0() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = array![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
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

        // Assertions for correctness
        assert_eq!(alist, vec![3.0, 2.0]); // check modified alist
        assert_eq!(flist, vec![3.0, 2.0]); // check modified flist
        assert_eq!(new_alp, 0.0);
        assert_eq!(new_abest, 3.0);
        assert_eq!(new_fbest, 3.0);
        assert_eq!(new_fmed, 3.0);
        assert_eq!(up, vec![false, false, false, false, false, false]); // check modified in place
        assert_eq!(down, vec![false, false, false, false, false, false]); // check modified in place
        assert_eq!(new_monotone, true); // whether it changed
        assert_eq!(minima, Vec::<bool>::new()); // minima same vec!
        assert_eq!(new_nmin, 1);
        assert_eq!(new_unitlen, 1.0);
        assert_eq!(new_s, 1);
    }

    #[test]
    fn test_case_1() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let p = array![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let mut alist = vec![1.0, 2.0, -1.0, 0.5];
        let mut flist = vec![5.0, 7.0, 3.0, 10.0];
        let alp = 1.5;
        let abest = 2.0;
        let fbest = 3.0; // min in flist
        let fmed = 6.0; // calculated manually
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

        // Assertions for correctness
        assert_eq!(alist, vec![1.0, 2.0, -1.0, 0.5]); // check modified alist
        assert_eq!(flist, vec![5.0, 7.0, 3.0, 10.0]); // check modified flist
        assert_eq!(new_alp, 1.5);
        assert_eq!(new_abest, 2.0);
        assert_eq!(new_fbest, 3.0);
        assert_eq!(new_fmed, 6.0);
        assert_eq!(up, vec![false, false, false, false, false, false]); // check modified in place
        assert_eq!(down, vec![true, false, true, false, true, false]); // check modified in place
        assert_eq!(new_monotone, false);
        assert_eq!(minima, vec![false, true]); // minima same vec
        assert_eq!(new_nmin, 2);
        assert_eq!(new_unitlen, 1.0);
        assert_eq!(new_s, 4);
    }

    #[test]
    fn test_case_2() {
        // Test inputs
        let x = array![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let p = array![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let mut alist = vec![-1.2, -3.4, -4.0, 2.5];
        let mut flist = vec![1.0, 4.5, 6.5, 2.0];
        let alp = 1.0;
        let abest = 1.5;
        let fbest = 1.0; // minimum flist value
        let fmed = 3.25; // calculated median
        let mut down = vec![true, false, true, false, true, false];
        let mut up = vec![false, false, false, false, false, false];
        let monotone = false;
        let mut minima = vec![];
        let nmin = 0;
        let unitlen = 1.0;
        let s = 4;

        // Call to `lsdescent`
        let (new_alp, new_abest, new_fbest, new_fmed, new_monotone, new_nmin, new_unitlen, new_s) =
            lsdescent(
                &x, &p, &mut alist, &mut flist, alp, abest, fbest, fmed,
                &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
            );

        // Assertions for correctness
        assert_eq!(alist, vec![-1.2, -3.4, -4.0, 2.5]); // check modified alist
        assert_eq!(flist, vec![1.0, 4.5, 6.5, 2.0]); // check modified flist
        assert_eq!(new_alp, 1.0);
        assert_eq!(new_abest, 1.5);
        assert_eq!(new_fbest, 1.0);
        assert_eq!(new_fmed, 3.25);
        assert_eq!(up, vec![false, false, false, false, false, false]); // check modified in place
        assert_eq!(down, vec![true, false, true, false, true, false]); // check modified in place
        assert_eq!(new_monotone, false);
        assert_eq!(minima, vec![]); // minima same vec
        assert_eq!(new_nmin, 0);
        assert_eq!(new_unitlen, 1.0);
        assert_eq!(new_s, 4);
    }
}
