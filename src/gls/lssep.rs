use crate::feval::feval;
use crate::gls::lsnew::lsnew;
use crate::gls::lssort::lssort;

use ndarray::Array1;
use std::cmp::Ordering;


pub fn lssep(
    nloc: i32,
    small: f64,
    sinit: i32,
    short: f64,
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
) -> (f64,  //amin
      f64,  //amax
      f64,  //alp
      f64,  //abest
      f64,  //fbest
      f64,  //fmed
      bool, //monotone
      usize, //nmin
      f64,  //unitlen
      usize //s
) {
    let mut nsep = 0;  // the original separation counter

    // Loop for separation points based on the differences in behavior
    while nsep < nmin {
        // Find intervals where the behavior of adjacent intervals is opposite (monotonicity behavior switches)
        *down = flist[1..s]
            .iter()
            .zip(&flist[0..s - 1])
            .map(|(&i, &j)| i < j)
            .collect();

        let mut sep: Vec<bool> = [vec![true, true], down.clone(), vec![true, true]].concat()
            .iter()
            .zip([vec![false], up.clone(), vec![false]].concat())
            .zip([down.clone(), vec![true, true]].concat())
            .map(|((i, j), k)| *i && j && k)
            .collect();

        let temp_sep: Vec<bool> = [vec![true, true], up.clone(), vec![true, true]].concat()
            .iter()
            .zip([vec![false], down.clone(), vec![false]].concat())
            .zip([up.clone(), vec![true, true]].concat())
            .map(|((i, j), k)| *i && j && k)
            .collect();

        // Combine temp_sep and sep
        sep = sep.iter().zip(&temp_sep).map(|(i, j)| *i || *j).collect();

        // Indices where separation occurs
        let ind: Vec<usize> = sep.iter().enumerate().filter_map(|(i, &val)| if val { Some(i) } else { None }).collect();
        if ind.len() == 0 {
            break;
        }

        // Calculating midpoints for the intervals to be checked
        let mut aa: Vec<f64> = ind.iter().map(|&i| 0.5 * (alist[i] + alist[i - 1])).collect();

        // If there are more midpoints than `nloc`, select the best `nloc`
        if aa.len() > nloc as usize {
            let mut ff: Vec<f64> = ind
                .iter()
                .map(|&i| flist[i].min(flist[i - 1])) // select minimum flist for those pairs
                .collect();

            let mut indices: Vec<usize> = (0..ff.len()).collect();
            indices.sort_by(|&i, &j| ff[i].partial_cmp(&ff[j]).unwrap_or(Ordering::Equal)); // sort by f values
            aa = indices.iter().take(nloc as usize).map(|&i| aa[i]).collect(); // pick the top nloc values
        }

        // For each midpoint alp, evaluate the function and update lists
        for &alp_elem in &aa {
            alp = alp_elem;
            let falp = feval(&(x + &(p * alp_elem)));
            alist.push(alp_elem);
            flist.push(falp);
            nsep += 1;
            if nsep >= nmin {
                break;
            }
        }

        // Sort the lists using `lssort`
        let (
            sorted_alist, permuted_flist, new_abest, new_fbest, new_fmed, sorted_up,
            sorted_down, new_monotone, sorted_minima, new_nmin, new_unitlen, new_s
        ) = lssort(alist, flist);

        // Reassign all sorted values after sorting
        *alist = sorted_alist;
        *flist = permuted_flist;
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        *up = sorted_up;
        *down = sorted_down;
        monotone = new_monotone;
        *minima = sorted_minima;
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;
    }

    // To account for missing separations, add points globally using lsnew
    for _ in 0..(nmin - nsep) {
        println!("2323");
        let res = lsnew(
            nloc, small, sinit, short, x, p, s, alist, flist, amin, amax, abest, fmed, unitlen,
        );
        alp = res.0;

        let (
            sorted_alist, permuted_flist, new_abest, new_fbest, new_fmed, sorted_up,
            sorted_down, new_monotone, sorted_minima, new_nmin, new_unitlen, new_s
        ) = lssort(alist, flist);

        // Reassign variables based on `lssort` results
        *alist = sorted_alist;
        *flist = permuted_flist;
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        *up = sorted_up;
        *down = sorted_down;
        monotone = new_monotone;
        *minima = sorted_minima;
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;
    }

    (amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_0() {
        let nloc = 1;
        let small = 1e-6;
        let sinit = 1;
        let short = 0.1;
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = array![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let mut alist = vec![3.0, 2.0];
        let mut flist = vec![3.0, 2.0];
        let amin = 0.0;
        let amax = 1.0;
        let alp = 1.5;
        let abest = 0.2;
        let fbest = 0.1;
        let fmed = 0.15;
        let mut up = vec![true, false];
        let mut down = vec![false, true];
        let monotone = false;
        let mut minima = vec![false, true];
        let nmin = 1;
        let unitlen = 0.1;
        let s = 2;

        let output = lssep(
            nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
        );

        assert_eq!(alist, vec![2.0, 2.5, 3.0]);
        assert_eq!(flist, vec![2.0, -0.0, 3.0]);
        assert_eq!(up, vec![false, true]);
        assert_eq!(down, vec![true, false]);
        assert_eq!(minima, vec![false, true, false]);
        assert_eq!(output, (0.0, 1.0, 2.5, 2.5, -0.0, 2.0, false, 1, 0.5, 3));
    }

    #[test]
    fn test_1() {
        let nloc = 1;
        let small = 1e-6;
        let sinit = 1;
        let short = 0.1;
        let x = array![1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0];
        let p = array![-1000.0, -2000.0, -3000.0, -4000.0, -5000.0, -6000.0];
        let mut alist = vec![3000.0, 2000.0];
        let mut flist = vec![3000.0, 2000.0];
        let amin = 0.0;
        let amax = 10000.0;
        let alp = 15000.0;
        let abest = 200.0;
        let fbest = 100.0;
        let fmed = 150.0;
        let mut up = vec![true, true];
        let mut down = vec![false, false];
        let monotone = false;
        let mut minima = vec![false, true];
        let nmin = 2;
        let unitlen = 0.1;
        let s = 1;

        let output = lssep(
            nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
        );

        assert_eq!(alist, vec![2000.0, 2500.0, 3000.0]);
        assert_eq!(flist, vec![2000.0, -0.0, 3000.0]);
        assert_eq!(up, vec![false, true]);
        assert_eq!(down, vec![true, false]);
        assert_eq!(minima, vec![false, true, false]);

        assert_eq!(output, (0.0, 10000.0, 2500.0, 2500.0, -0.0, 2000.0, false, 1, 500.0, 3));
    }

    #[test]
    fn test_2() {
        let nloc = 0;
        let small = 0.0;
        let sinit = 0;
        let short = 0.0;
        let x = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut alist = vec![0.0, 0.0];
        let mut flist = vec![0.0, 0.0];
        let amin = 0.0;
        let amax = 0.0;
        let alp = 0.0;
        let abest = 0.0;
        let fbest = 0.0;
        let fmed = 0.0;
        let mut up = vec![false, false];
        let mut down = vec![false, false];
        let monotone = false;
        let mut minima = vec![false, false];
        let nmin = 0;
        let unitlen = 0.0;
        let s = 1;

        let output = lssep(
            nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
        );

        assert_eq!(alist, vec![0.0, 0.0]);
        assert_eq!(flist, vec![0.0, 0.0]);
        assert_eq!(up, vec![false, false]);
        assert_eq!(down, vec![false, false]);
        assert_eq!(minima, vec![false, false]);

        assert_eq!(output, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, 0.0, 1));
    }
}
