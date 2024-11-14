use crate::feval::feval;
use crate::gls::lsguard::lsguard;
use crate::gls::lssort::lssort;

use ndarray::Array1;

pub fn lslocal(
    nloc: i32,
    small: f64,
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
    mut saturated: bool,
) -> (f64, //alp
      f64, //abest
      f64, //fbest
      f64, //fmed
      bool, //monotone
      usize, //nmin
      f64, //unitlen
      usize, //s
      bool //saturated
) {
    // Calculate up and down vectors
    up.clear();
    for i in 0..s - 1 {
        up.push(flist[i] < flist[i + 1]);
    }

    down.clear();
    for i in 1..s {
        down.push(flist[i] <= flist[i - 1]);
    }
    // Fix the last element of down as in Python
    if down.len() > 0 {
        down[s - 2] = flist[s - 1] < flist[s - 2];
    }

    // Calculate minima using Python's logic of padding with True
    minima.clear();
    let mut padded_up = up.clone();
    padded_up.push(true);
    let mut padded_down = vec![true];
    padded_down.extend(down.iter().cloned());

    for i in 0..s {
        minima.push(padded_up[i] && padded_down[i]);
    }

    // Get indices of minima
    let mut imin: Vec<usize> = minima.iter()
        .enumerate()
        .filter(|(_, &is_min)| is_min)
        .map(|(i, _)| i)
        .collect();

    // Sort minima by function values and get permutation
    let mut ff: Vec<f64> = imin.iter().map(|&i| flist[i]).collect();
    let mut perm: Vec<usize> = (0..ff.len()).collect();
    perm.sort_by(|&a, &b| ff[a].partial_cmp(&ff[b]).unwrap());

    // Apply permutation to imin
    imin = perm.iter().map(|&i| imin[i]).collect();
    let nind = std::cmp::min(nloc as usize, imin.len());
    // Match Python's slice exactly: imin[nind-1::-1]
    if nind > 0 {
        imin = imin[..nind].iter().rev().cloned().collect();
    } else {
        imin.clear();
    }

    let mut nadd = 0;
    let mut nsat = 0;

    for &i in &imin {
        // Select nearest five points for local formula
        let (ind, ii) = if i <= 1 {
            ((0..5).collect::<Vec<usize>>(), i)
        } else if i >= s - 2 {
            ((s - 5..s).collect::<Vec<usize>>(), i - (s - 1) + 4)
        } else {
            ((i - 2..i + 3).collect::<Vec<usize>>(), 2)
        };

        let aa: Vec<f64> = ind.iter().map(|&j| alist[j]).collect();
        let ff: Vec<f64> = ind.iter().map(|&j| flist[j]).collect();

        // Get divided differences
        let f12 = (ff[1] - ff[0]) / (aa[1] - aa[0]);
        let f23 = (ff[2] - ff[1]) / (aa[2] - aa[1]);
        let f34 = (ff[3] - ff[2]) / (aa[3] - aa[2]);
        let f45 = (ff[4] - ff[3]) / (aa[4] - aa[3]);
        let f123 = (f23 - f12) / (aa[2] - aa[0]);
        let f234 = (f34 - f23) / (aa[3] - aa[1]);
        let f345 = (f45 - f34) / (aa[4] - aa[2]);

        // Decide on action using the same logic as Python
        let mut cas = 0;

        if ii == 0 {
            if f123 > 0.0 && f123.is_finite() {
                alp = 0.5 * (aa[1] + aa[2] - f23 / f123);
                if alp < amin {
                    cas = -1;
                }
            } else {
                alp = f64::NEG_INFINITY;
                if (alist[0] - amin).abs() < f64::EPSILON && flist[1] < flist[2] {
                    cas = -1;
                }
            }
            alp = lsguard(alp, alist, amax, amin, small);
        } else if ii == 4 {
            if f345 > 0.0 && f345.is_finite() {
                alp = 0.5 * (aa[2] + aa[3] - f34 / f345);
                if alp > amax {
                    cas = -1;
                }
            } else {
                alp = f64::INFINITY;
                if (alist[s - 1] - amax).abs() < f64::EPSILON && flist[s - 2] < flist[s - 3] {
                    cas = -1;
                }
            }
            alp = lsguard(alp, alist, amax, amin, small);
        } else if !(f234 > 0.0 && f234.is_finite()) {
            cas = 0;
            if ii < 2 {
                alp = 0.5 * (aa[1] + aa[2] - f23 / f123);
            } else {
                alp = 0.5 * (aa[2] + aa[3] - f34 / f345);
            }
        } else if !(f123 > 0.0 && f123.is_finite()) {
            if f345 > 0.0 && f345.is_finite() {
                cas = 5;
            } else {
                cas = 0;
                alp = 0.5 * (aa[2] + aa[3] - f34 / f234);
            }
        } else if f345 > 0.0 && f345.is_finite() && ff[1] > ff[3] {
            cas = 5;
        } else {
            cas = 1;
        }

        match cas {
            0 => {
                alp = alp.max(amin).min(amax);
            }
            1 => {
                let f1x4 = if ff[1] < ff[2] {
                    let f13 = (ff[2] - ff[0]) / (aa[2] - aa[0]);
                    (f34 - f13) / (aa[3] - aa[0])
                } else {
                    let f24 = (ff[3] - ff[1]) / (aa[3] - aa[1]);
                    (f24 - f12) / (aa[3] - aa[0])
                };
                alp = 0.5 * (aa[1] + aa[2] - f23 / (f123 + f234 - f1x4));
                if alp <= *aa.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() ||
                    alp >= *aa.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() {
                    cas = 0;
                    alp = 0.5 * (aa[1] + aa[2] - f23 / f123.max(f234));
                }
            }
            5 => {
                let f2x5 = if ff[2] < ff[3] {
                    let f24 = (ff[3] - ff[1]) / (aa[3] - aa[1]);
                    (f45 - f24) / (aa[4] - aa[1])
                } else {
                    let f35 = (ff[4] - ff[2]) / (aa[4] - aa[2]);
                    (f35 - f23) / (aa[4] - aa[1])
                };
                alp = 0.5 * (aa[2] + aa[3] - f34 / (f234 + f345 - f2x5));
                if alp <= *aa.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() ||
                    alp >= *aa.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() {
                    cas = 0;
                    alp = 0.5 * (aa[2] + aa[3] - f34 / f234.max(f345));
                }
            }
            _ => {}
        }

        // Calculate tolerance for accepting new step
        let alptol = if cas < 0 || flist[i] > fmed {
            0.0
        } else if i == 0 {
            small * (alist[2] - alist[0])
        } else if i == s - 1 {
            small * (alist[s - 1] - alist[s - 3])
        } else {
            small * (alist[i + 1] - alist[i - 1])
        };

        let close = alist.iter().any(|&a| (a - alp).abs() <= alptol);

        if cas < 0 || close {
            nsat += 1;
        }

        saturated = nsat == nind;
        let final_check = saturated && !alist.iter().any(|&a| (a - alp).abs() < f64::EPSILON);

        if cas >= 0 && (final_check || !close) {
            nadd += 1;
            let falp = feval(&(x + &(alp * p)));
            alist.push(alp);
            flist.push(falp);
        }
    }

    if nadd > 0 {
        let (new_alist, new_flist, new_abest, new_fbest, new_fmed, new_up,
            new_down, new_monotone, new_minima, new_nmin, new_unitlen, new_s) =
            lssort(alist, flist);

        // Update all vectors and values
        alist.clear();
        alist.extend(new_alist);
        flist.clear();
        flist.extend(new_flist);
        up.clear();
        up.extend(new_up);
        down.clear();
        down.extend(new_down);
        minima.clear();
        minima.extend(new_minima);

        // Update scalar values
        abest = new_abest;
        fbest = new_fbest;
        fmed = new_fmed;
        monotone = new_monotone;
        nmin = new_nmin;
        unitlen = new_unitlen;
        s = new_s;
    }

    (alp, abest, fbest, fmed, monotone, nmin, unitlen, s, saturated)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;


    #[test]
    fn test_basic_minimum() {
        let nloc = 3;
        let small = 1e-6;
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut flist = vec![10.0, 8.0, 7.0, 8.0, 10.0];
        let amin = 0.0;
        let amax = 2.0;
        let alp = 1.0;
        let abest = 1.0;
        let fbest = 7.0;
        let fmed = 8.0;
        let mut up = vec![true, true, false, false];
        let mut down = vec![true, true, false, false];
        let monotone = false;
        let mut minima = vec![false, false, true, false, false];
        let nmin = 1;
        let unitlen = 1.0;
        let s = 5;
        let saturated = false;

        let expected_alist = vec![0.0, 0.5, 1.0, 1.05, 1.5, 2.0];
        let expected_flist = vec![10.0, 8.0, 7.0, -2.3696287132470126e-202, 8.0, 10.0];
        let expected_up = vec![false, false, false, true, true];
        let expected_down = vec![true, true, true, false, false];
        let expected_minima = vec![false, false, false, true, false, false];

        let (alp_out, abest_out, fbest_out, fmed_out, monotone_out, nmin_out, unitlen_out, s_out, saturated_out) =
            lslocal(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed,
                    &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        // Check return values
        assert_relative_eq!(alp_out, 1.05);
        assert_relative_eq!(abest_out, 1.05);
        assert_relative_eq!(fbest_out, -2.3696287132470126e-202);
        assert_relative_eq!(fmed_out, 8.0);
        assert_eq!(monotone_out, false);
        assert_eq!(nmin_out, 1);
        assert_relative_eq!(unitlen_out, 1.05);
        assert_eq!(s_out, 6);
        assert_eq!(saturated_out, false);

        // Check modified vectors
        assert_eq!(alist, expected_alist);
        for (f1, f2) in flist.iter().zip(expected_flist.iter()) {
            assert_relative_eq!(*f1, *f2);
        }
        assert_eq!(up, expected_up);
        assert_eq!(down, expected_down);
        assert_eq!(minima, expected_minima);
    }

    #[test]
    fn test_multiple_minima() {
        let nloc = 3;
        let small = 1e-6;
        let x = Array1::from_vec(vec![1.0; 6]);
        let p = Array1::from_vec(vec![0.2; 6]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut flist = vec![10.0, 8.0, 6.0, 9.0, 5.0, 7.0, 10.0];
        let amin = 0.0;
        let amax = 3.0;
        let alp = 2.0;
        let abest = 2.0;
        let fbest = 5.0;
        let fmed = 8.0;
        let mut up = vec![true, true, false, true, false, false];
        let mut down = vec![true, true, false, true, false, false];
        let monotone = false;
        let mut minima = vec![false, false, true, false, true, false, false];
        let nmin = 2;
        let unitlen = 1.0;
        let s = 7;
        let saturated = false;

        let expected_alist = vec![0.0, 0.5, 0.95, 1.0, 1.5, 2.0, 2.019230769230769, 2.5, 3.0];
        let expected_flist = vec![10.0, 8.0, -1.2948375496133751e-8, 6.0, 9.0, 5.0, -2.628133731425332e-14, 7.0, 10.0];
        let expected_up = vec![false, false, true, true, false, false, true, true];
        let expected_down = vec![true, true, false, false, true, true, false, false];
        let expected_minima = vec![false, false, true, false, false, false, true, false, false];

        let (alp_out, abest_out, fbest_out, fmed_out, monotone_out, nmin_out, unitlen_out, s_out, saturated_out) =
            lslocal(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed,
                    &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        assert_relative_eq!(alp_out, 2.019230769230769);
        assert_relative_eq!(abest_out, 0.95);
        assert_relative_eq!(fbest_out, -1.2948375496133751e-8);
        assert_relative_eq!(fmed_out, 7.0);
        assert_eq!(monotone_out, false);
        assert_eq!(nmin_out, 2);
        assert_relative_eq!(unitlen_out, 1.0692307692307692);
        assert_eq!(s_out, 9);
        assert_eq!(saturated_out, false);

        assert_eq!(alist, expected_alist);
        for (f1, f2) in flist.iter().zip(expected_flist.iter()) {
            assert_relative_eq!(*f1, *f2);
        }
        assert_eq!(up, expected_up);
        assert_eq!(down, expected_down);
        assert_eq!(minima, expected_minima);
    }

    #[test]
    fn test_boundary_minimum_left() {
        let nloc = 3;
        let small = 1e-6;
        let x = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let p = Array1::from_vec(vec![-0.1; 6]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut flist = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let amin = 0.0;
        let amax = 2.0;
        let alp = 0.0;
        let abest = 0.0;
        let fbest = 5.0;
        let fmed = 7.0;
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let monotone = true;
        let mut minima = vec![true, false, false, false, false];
        let nmin = 1;
        let unitlen = 1.0;
        let s = 5;
        let saturated = false;

        let expected_up = vec![true; 4];
        let expected_down = vec![false; 4];

        let (alp_out, abest_out, fbest_out, fmed_out, monotone_out, nmin_out, unitlen_out, s_out, saturated_out) =
            lslocal(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed,
                    &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        assert_relative_eq!(alp_out, 0.16666666666666666);
        assert_relative_eq!(abest_out, 0.0);
        assert_relative_eq!(fbest_out, 5.0);
        assert_relative_eq!(fmed_out, 7.0);
        assert_eq!(monotone_out, true);
        assert_eq!(nmin_out, 1);
        assert_relative_eq!(unitlen_out, 1.0);
        assert_eq!(s_out, 5);
        assert_eq!(saturated_out, true);

        // Original lists should remain unchanged in this case
        assert_eq!(alist, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(flist, vec![5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(up, expected_up);
        assert_eq!(down, expected_down);
        assert_eq!(minima, vec![true, false, false, false, false]);
    }

    #[test]
    fn test_boundary_minimum_right() {
        let nloc = 3;
        let small = 1e-6;
        let x = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let p = Array1::from_vec(vec![0.1; 6]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut flist = vec![9.0, 8.0, 7.0, 6.0, 5.0];
        let amin = 0.0;
        let amax = 2.0;
        let alp = 2.0;
        let abest = 2.0;
        let fbest = 5.0;
        let fmed = 7.0;
        let mut up = vec![true; 4];
        let mut down = vec![false; 4];
        let monotone = true;
        let mut minima = vec![false, false, false, false, true];
        let nmin = 1;
        let unitlen = 1.0;
        let s = 5;
        let saturated = false;

        let expected_up = vec![false; 4];
        let expected_down = vec![true; 4];

        let (alp_out, abest_out, fbest_out, fmed_out, monotone_out, nmin_out, unitlen_out, s_out, saturated_out) =
            lslocal(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed,
                    &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        assert_relative_eq!(alp_out, 1.8333333333333333);
        assert_relative_eq!(abest_out, 2.0);
        assert_relative_eq!(fbest_out, 5.0);
        assert_relative_eq!(fmed_out, 7.0);
        assert_eq!(monotone_out, true);
        assert_eq!(nmin_out, 1);
        assert_relative_eq!(unitlen_out, 1.0);
        assert_eq!(s_out, 5);
        assert_eq!(saturated_out, true);

        assert_eq!(alist, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(flist, vec![9.0, 8.0, 7.0, 6.0, 5.0]);
        assert_eq!(up, expected_up);
        assert_eq!(down, expected_down);
        assert_eq!(minima, vec![false, false, false, false, true]);
    }

    #[test]
    fn test_saturated_case() {
        let nloc = 2;
        let small = 1e-6;
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = Array1::from_vec(vec![0.1; 6]);
        let mut alist = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut flist = vec![10.0, 8.0, 7.0, 8.0, 10.0];
        let amin = 0.0;
        let amax = 1.0;
        let alp = 0.5;
        let abest = 0.5;
        let fbest = 7.0;
        let fmed = 8.0;
        let mut up = vec![true, true, false, false];
        let mut down = vec![true, true, false, false];
        let monotone = false;
        let mut minima = vec![false, false, true, false, false];
        let nmin = 1;
        let unitlen = 0.25;
        let s = 5;
        let saturated = true;

        let expected_alist = vec![0.0, 0.25, 0.5, 0.525, 0.75, 1.0];
        let expected_flist = vec![10.0, 8.0, 7.0, -1.00989834152592e-196, 8.0, 10.0];
        let expected_up = vec![false, false, false, true, true];
        let expected_down = vec![true, true, true, false, false];
        let expected_minima = vec![false, false, false, true, false, false];

        let (alp_out, abest_out, fbest_out, fmed_out, monotone_out, nmin_out, unitlen_out, s_out, saturated_out) =
            lslocal(nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, alp, abest, fbest, fmed,
                    &mut up, &mut down, monotone, &mut minima, nmin, unitlen, s, saturated);

        assert_relative_eq!(alp_out, 0.525);
        assert_relative_eq!(abest_out, 0.525);
        assert_relative_eq!(fbest_out, -1.00989834152592e-196);
    }
}