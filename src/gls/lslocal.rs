use crate::gls::lsguard::lsguard;
use crate::gls::lssort::lssort;
use nalgebra::SVector;


pub fn lslocal<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    nloc: usize,
    small: f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
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
    saturated: &mut bool,
) {
    // Calculate up and down vectors
    up.clear();
    down.clear();
    flist.windows(2).take(*s - 1).for_each(|win| {
        up.push(win[0] < win[1]);
        down.push(win[1] <= win[0]);
    });

    // Fix the last element of down as in Python
    if down.len() > 0 {
        down[*s - 2] = flist[*s - 1] < flist[*s - 2];
    }

    // Calculate minima using Python's logic of padding with True
    minima.clear();
    minima.extend(
        up.iter()
            .chain(std::iter::once(&true))
            .zip(std::iter::once(&true).chain(down.iter()))
            .take(*s)
            .map(|(up_val, down_val)| *up_val && *down_val)
    );

    // Pre-allocate with capacity to avoid reallocations
    let mut index_value_pairs = Vec::with_capacity(minima.len());

    // Single pass through data to collect indices and values together
    for (idx, &is_min) in minima.iter().enumerate() {
        if is_min {
            index_value_pairs.push((idx, flist[idx]));
        }
    }

    // Sort by values directly, avoiding separate permutation vector
    index_value_pairs.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    // Take needed number of elements, reverse, and extract indices
    let nind = std::cmp::min(nloc, index_value_pairs.len());

    // Preallocate result vector with exact size
    let mut imin = Vec::with_capacity(nind);
    for i in (0..nind).rev() {
        imin.push(index_value_pairs[i].0);
    }


    let mut nadd = 0;
    let mut nsat = 0;

    let mut x_alp_p: SVector<f64, N>;

    for i in imin {
        // Select nearest five points for local formula
        let (ind, ii) = if i <= 1 {
            ([0, 1, 2, 3, 4], i)
        } else if i >= *s - 2 {
            ([*s - 5, *s - 4, *s - 3, *s - 2, *s - 1], i + 5 - *s)
        } else {
            ([i - 2, i - 1, i, i + 1, i + 2], 2)
        };

        let aa: [f64; 5] = std::array::from_fn(|i| alist[ind[i]]);
        let ff: [f64; 5] = std::array::from_fn(|i| flist[ind[i]]);

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
                *alp = 0.5 * (aa[1] + aa[2] - f23 / f123);
                if *alp < amin {
                    cas = -1;
                }
            } else {
                *alp = f64::NEG_INFINITY;
                if (alist[0] - amin).abs() < f64::EPSILON && flist[1] < flist[2] {
                    cas = -1;
                }
            }
            lsguard(alp, alist, amax, amin, small);
        } else if ii == 4 {
            if f345 > 0.0 && f345.is_finite() {
                *alp = 0.5 * (aa[2] + aa[3] - f34 / f345);
                if *alp > amax {
                    cas = -1;
                }
            } else {
                *alp = f64::INFINITY;
                if (alist[*s - 1] - amax).abs() < f64::EPSILON && flist[*s - 2] < flist[*s - 3] {
                    cas = -1;
                }
            }
            lsguard(alp, alist, amax, amin, small);
        } else if !(f234 > 0.0 && f234.is_finite()) {
            cas = 0;
            if ii < 2 {
                *alp = 0.5 * (aa[1] + aa[2] - f23 / f123);
            } else {
                *alp = 0.5 * (aa[2] + aa[3] - f34 / f345);
            }
        } else if !(f123 > 0.0 && f123.is_finite()) {
            if f345 > 0.0 && f345.is_finite() {
                cas = 5;
            } else {
                cas = 0;
                *alp = 0.5 * (aa[2] + aa[3] - f34 / f234);
            }
        } else if f345 > 0.0 && f345.is_finite() && ff[1] > ff[3] {
            cas = 5;
        } else {
            cas = 1;
        }

        match cas {
            0 => {
                *alp = alp.max(amin).min(amax);
            }
            1 => {
                let f1x4 = if ff[1] < ff[2] {
                    let f13 = (ff[2] - ff[0]) / (aa[2] - aa[0]);
                    (f34 - f13) / (aa[3] - aa[0])
                } else {
                    let f24 = (ff[3] - ff[1]) / (aa[3] - aa[1]);
                    (f24 - f12) / (aa[3] - aa[0])
                };
                *alp = 0.5 * (aa[1] + aa[2] - f23 / (f123 + f234 - f1x4));
                if *alp <= *aa.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
                    || *alp >= *aa.iter().max_by(|a, b| a.total_cmp(b)).unwrap() {
                    cas = 0;
                    *alp = 0.5 * (aa[1] + aa[2] - f23 / f123.max(f234));
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
                *alp = 0.5 * (aa[2] + aa[3] - f34 / (f234 + f345 - f2x5));
                if *alp <= *aa.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
                    || *alp >= *aa.iter().max_by(|a, b| a.total_cmp(b)).unwrap() {
                    cas = 0;
                    *alp = 0.5 * (aa[2] + aa[3] - f34 / f234.max(f345));
                }
            }
            _ => {}
        }

        // Calculate tolerance for accepting new STEP
        let alptol = if cas < 0 || flist[i] > *fmed {
            0.0
        } else if i == 0 {
            small * (alist[2] - alist[0])
        } else if i == *s - 1 {
            small * (alist[*s - 1] - alist[*s - 3])
        } else {
            small * (alist[i + 1] - alist[i - 1])
        };

        let close = alist.iter().any(|&a| (a - *alp).abs() <= alptol);

        if cas < 0 || close {
            nsat += 1;
        }

        *saturated = nsat == nind;
        let final_check = *saturated && !alist.iter().any(|&a| a == *alp);

        if cas >= 0 && (final_check || !close) {
            nadd += 1;
            x_alp_p = x + p.scale(*alp);
            let falp = func(&x_alp_p);
            alist.push(*alp);
            flist.push(falp);
        }
    }

    if nadd > 0 {
        (*abest, *fbest, *fmed, *up, *down, *monotone, *minima, *nmin, *unitlen, *s) = lssort(alist, flist);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_basic_minimum() {
        let nloc = 3;
        let small = 1e-6;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut flist = vec![10.0, 8.0, 7.0, 8.0, 10.0];
        let amin = 0.0;
        let amax = 2.0;
        let mut alp = 1.0;
        let mut abest = 1.0;
        let mut fbest = 7.0;
        let mut fmed = 8.0;
        let mut up = vec![true, true, false, false];
        let mut down = vec![true, true, false, false];
        let mut monotone = false;
        let mut minima = vec![false, false, true, false, false];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = 5;
        let mut saturated = false;

        let expected_alist = vec![0.0, 0.5, 1.0, 1.05, 1.5, 2.0];
        let expected_flist = vec![10.0, 8.0, 7.0, -2.3696287132470126e-202, 8.0, 10.0];
        let expected_up = vec![false, false, false, true, true];
        let expected_down = vec![true, true, true, false, false];
        let expected_minima = vec![false, false, false, true, false, false];

        lslocal(hm6, nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, &mut alp, &mut abest, &mut fbest, &mut fmed,
                &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alp, 1.05);
        assert_eq!(abest, 1.05);
        assert_eq!(fbest, -2.3696287132470126e-202);
        assert_eq!(fmed, 8.0);
        assert_eq!(monotone, false);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 1.05);
        assert_eq!(s, 6);
        assert_eq!(saturated, false);
        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(up, expected_up);
        assert_eq!(down, expected_down);
        assert_eq!(minima, expected_minima);
    }

    #[test]
    fn test_multiple_minima() {
        let nloc = 3;
        let small = 1e-6;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.2; 6]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut flist = vec![10.0, 8.0, 6.0, 9.0, 5.0, 7.0, 10.0];
        let amin = 0.0;
        let amax = 3.0;
        let mut alp = 2.0;
        let mut abest = 2.0;
        let mut fbest = 5.0;
        let mut fmed = 8.0;
        let mut up = vec![true, true, false, true, false, false];
        let mut down = vec![true, true, false, true, false, false];
        let mut monotone = false;
        let mut minima = vec![false, false, true, false, true, false, false];
        let mut nmin = 2;
        let mut unitlen = 1.0;
        let mut s = 7;
        let mut saturated = false;

        let expected_alist = vec![0.0, 0.5, 0.95, 1.0, 1.5, 2.0, 2.019230769230769, 2.5, 3.0];
        let expected_flist = vec![10.0, 8.0, -1.2948375496133751e-8, 6.0, 9.0, 5.0, -2.628133731425332e-14, 7.0, 10.0];
        let expected_up = vec![false, false, true, true, false, false, true, true];
        let expected_down = vec![true, true, false, false, true, true, false, false];
        let expected_minima = vec![false, false, true, false, false, false, true, false, false];

        lslocal(hm6, nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, &mut alp, &mut abest, &mut fbest, &mut fmed,
                &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alp, 2.019230769230769);
        assert_eq!(abest, 0.95);
        assert_eq!(fbest, -1.2948375496133751e-8);
        assert_eq!(fmed, 7.0);
        assert_eq!(monotone, false);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 1.0692307692307692);
        assert_eq!(s, 9);
        assert_eq!(saturated, false);
        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(up, expected_up);
        assert_eq!(down, expected_down);
        assert_eq!(minima, expected_minima);
    }

    #[test]
    fn test_boundary_minimum_left() {
        let nloc = 3;
        let small = 1e-6;
        let x = SVector::<f64, 6>::from_row_slice(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-0.1; 6]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut flist = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let amin = 0.0;
        let amax = 2.0;
        let mut alp = 0.0;
        let mut abest = 0.0;
        let mut fbest = 5.0;
        let mut fmed = 7.0;
        let mut up = vec![false; 4];
        let mut down = vec![true; 4];
        let mut monotone = true;
        let mut minima = vec![true, false, false, false, false];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = 5;
        let mut saturated = false;

        let expected_up = vec![true; 4];
        let expected_down = vec![false; 4];

        lslocal(hm6, nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, &mut alp, &mut abest, &mut fbest, &mut fmed,
                &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alp, 0.16666666666666666);
        assert_eq!(abest, 0.0);
        assert_eq!(fbest, 5.0);
        assert_eq!(fmed, 7.0);
        assert_eq!(monotone, true);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 1.0);
        assert_eq!(s, 5);
        assert_eq!(saturated, true);
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
        let x = SVector::<f64, 6>::from_row_slice(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.1; 6]);
        let mut alist = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut flist = vec![9.0, 8.0, 7.0, 6.0, 5.0];
        let amin = 0.0;
        let amax = 2.0;
        let mut alp = 2.0;
        let mut abest = 2.0;
        let mut fbest = 5.0;
        let mut fmed = 7.0;
        let mut up = vec![true; 4];
        let mut down = vec![false; 4];
        let mut monotone = true;
        let mut minima = vec![false, false, false, false, true];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = 5;
        let mut saturated = false;

        let expected_up = vec![false; 4];
        let expected_down = vec![true; 4];

        lslocal(hm6, nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, &mut alp, &mut abest, &mut fbest, &mut fmed,
                &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alp, 1.8333333333333333);
        assert_eq!(abest, 2.0);
        assert_eq!(fbest, 5.0);
        assert_eq!(fmed, 7.0);
        assert_eq!(monotone, true);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 1.0);
        assert_eq!(s, 5);
        assert_eq!(saturated, true);
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
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.1; 6]);
        let mut alist = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut flist = vec![10.0, 8.0, 7.0, 8.0, 10.0];
        let amin = 0.0;
        let amax = 1.0;
        let mut alp = 0.5;
        let mut abest = 0.5;
        let mut fbest = 7.0;
        let mut fmed = 8.0;
        let mut up = vec![true, true, false, false];
        let mut down = vec![true, true, false, false];
        let mut monotone = false;
        let mut minima = vec![false, false, true, false, false];
        let mut nmin = 1;
        let mut unitlen = 0.25;
        let mut s = 5;
        let mut saturated = true;

        let expected_alist = vec![0.0, 0.25, 0.5, 0.525, 0.75, 1.0];
        let expected_flist = vec![10.0, 8.0, 7.0, -1.00989834152592e-196, 8.0, 10.0];
        let expected_up = vec![false, false, false, true, true];
        let expected_down = vec![true, true, true, false, false];
        let expected_minima = vec![false, false, false, true, false, false];

        lslocal(hm6, nloc, small, &x, &p, &mut alist, &mut flist, amin, amax, &mut alp, &mut abest, &mut fbest, &mut fmed,
                &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alp, 0.525);
        assert_eq!(abest, 0.525);
        assert_eq!(fbest, -1.00989834152592e-196);
        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(up, expected_up);
        assert_eq!(down, expected_down);
        assert_eq!(minima, expected_minima);
    }
}