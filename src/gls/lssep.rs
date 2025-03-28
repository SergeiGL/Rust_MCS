use crate::gls::lsnew::lsnew;
use crate::gls::lssort::lssort;
use nalgebra::SVector;


pub fn lssep<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    nloc: usize,
    small: f64,
    sinit: usize,
    short: f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
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
) -> (
    f64,   // amin
    f64,   // amax
    f64,   // alp
    f64,   // abest
    f64,   // fbest
    f64,   // fmed
    bool,  // monotone
    usize, // nmin
    f64,   // unitlen
    usize  // s
) {
    let mut nsep = 0;  // the original separation counter

    // Pre-allocate reusable vectors to minimize allocations within the loop
    let cap = down.len().max(up.len()).max(nloc) + 10; // +2 needed, +10 just in case
    let mut ind = Vec::with_capacity(cap);
    let mut aa = Vec::with_capacity(cap);
    let mut ff = Vec::with_capacity(cap);

    let mut x_alp_p: SVector<f64, N>;

    // Loop for separation points based on the differences in behavior
    while nsep < nmin {
        // Find intervals where the behavior of adjacent intervals is opposite (monotonicity behavior switches)
        down.clear();
        for i in 0..s - 1 {
            down.push(flist[i + 1] < flist[i]);  // Using strict < comparison
        }

        let down_len = down.len();
        let up_len = up.len();

        let sep_len = 2 + std::cmp::min(down_len, up_len);

        ind.clear();
        for n in 0..sep_len {
            // Compute `sep[n]` equivalent:
            let i_sep = if n < 2 {
                true
            } else {
                down[n - 2]
            };

            let j_sep = if n == 0 {
                false
            } else {
                up.get(n - 1).copied().unwrap_or(false)
            };

            let k_sep = if n < down_len {
                down[n]
            } else {
                true
            };

            let sep_val = i_sep && j_sep && k_sep;

            // Compute `temp_sep[n]` equivalent:
            let i_temp_sep = if n < 2 {
                true
            } else {
                up[n - 2]
            };

            let j_temp_sep = if n == 0 {
                false
            } else {
                down.get(n - 1).copied().unwrap_or(false)
            };

            // k_temp_sep = up[n] if within bounds, else true
            let k_temp_sep = if n < up_len {
                up[n]
            } else {
                true
            };

            let temp_sep_val = i_temp_sep && j_temp_sep && k_temp_sep;

            // If either `sep_val` or `temp_sep_val` is true, record the index.
            if sep_val || temp_sep_val {
                ind.push(n);
            }
        }

        if ind.is_empty() {
            break;
        }

        // Calculating midpoints for the intervals to be checked
        aa.clear();
        ff.clear();
        for &i in ind.iter() {
            aa.push(0.5 * (alist[i] + alist[i - 1]));
            ff.push(flist[i].min(flist[i - 1]));
        };

        if aa.len() > nloc {
            let mut indices_with_values: Vec<(usize, f64)> = ff.iter().cloned()
                .enumerate()
                .collect();
            indices_with_values.sort_unstable_by(|(_, f_i), (_, f_j)| f_i.total_cmp(f_j));

            aa.clear();
            for &(i, _) in indices_with_values.iter().take(nloc) {
                aa.push(aa[i]);
            }
        }

        // For each midpoint alp, evaluate the function and update lists
        for &alp_elem in &aa {
            alp = alp_elem;
            x_alp_p = x + p.scale(alp);
            let falp = func(&x_alp_p);
            alist.push(alp_elem);
            flist.push(falp);
            nsep += 1;
            if nsep >= nmin {
                break;
            }
        }

        // Sort the lists using `lssort`
        (abest, fbest, fmed, *up, *down, monotone, *minima, nmin, unitlen, s) = lssort(alist, flist);
    }

    // To account for missing separations, add points globally using lsnew
    for _ in 0..(nmin - nsep) {
        alp = lsnew(func, nloc, small, sinit, short, x, p, s, alist, flist, amin, amax, abest, fmed, unitlen);
        (abest, fbest, fmed, *up, *down, monotone, *minima, nmin, unitlen, s) = lssort(alist, flist);
    }

    (amin, amax, alp, abest, fbest, fmed, monotone, nmin, unitlen, s)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_0() {
        let nloc = 1;
        let small = 1e-6;
        let sinit = 1;
        let short = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
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
            hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
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
        let x = SVector::<f64, 6>::from_row_slice(&[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1000.0, -2000.0, -3000.0, -4000.0, -5000.0, -6000.0]);
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
            hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
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
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
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
            hm6, nloc, small, sinit, short, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed, &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s,
        );

        assert_eq!(alist, vec![0.0, 0.0]);
        assert_eq!(flist, vec![0.0, 0.0]);
        assert_eq!(up, vec![false, false]);
        assert_eq!(down, vec![false, false]);
        assert_eq!(minima, vec![false, false]);

        assert_eq!(output, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, 0.0, 1));
    }
}
