pub fn lsguard(alp: &mut f64, alist: &mut Vec<f64>, amax: f64, amin: f64, small: f64) {
    alist.sort_unstable_by(|a, b| a.total_cmp(&b));
    let s = alist.len();

    // Enforce extrapolation to be cautious
    let al: f64 = alist[0] - (alist[s - 1] - alist[0]) / small;
    let au: f64 = alist[s - 1] + (alist[s - 1] - alist[0]) / small;

    *alp = alp.max(al).min(au).max(amin).min(amax);

    // Enforce some distance from endpoints
    if (*alp - alist[0]).abs() < small * (alist[1] - alist[0]) {
        *alp = (2.0 * alist[0] + alist[1]) / 3.0;
    }

    if (*alp - alist[s - 1]).abs() < small * (alist[s - 1] - alist[s - 2]) {
        *alp = (2.0 * alist[s - 1] + alist[s - 2]) / 3.0;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_0() {
        let mut alp = -1.0;
        lsguard(&mut alp, &mut vec![0.1, 0.0, 7.0], 5.0, 4.0, 10.0);
        assert_eq!(alp, 4.7);
    }

    #[test]
    fn test_coverage_1() {
        let mut alp = 13.0;
        lsguard(&mut alp, &mut vec![0.1, 0.0, 7.0], 5.0, 4.0, 100.0);
        assert_eq!(alp, 4.7);
    }

    #[test]
    fn test_coverage_2() {
        let mut alp = 13.0;
        lsguard(&mut alp, &mut vec![1.1, 0.8, 8.0], -5.0, 10.0, 100.0);
        assert_eq!(alp, 5.7);
    }

    #[test]
    fn test_within_bounds() {
        let mut alp = 5.0;
        lsguard(&mut alp, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 5.0);
    }

    #[test]
    fn test_below_extrapolated_bounds() {
        let mut alp = -1.0;
        lsguard(&mut alp, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 0.0);
    }

    #[test]
    fn test_above_extrapolated_bounds() {
        let mut alp = 12.0;
        lsguard(&mut alp, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 10.0);
    }

    #[test]
    fn test_near_lower_bound() {
        let mut alp = 3.1;
        lsguard(&mut alp, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 4.333333333333333);
    }

    #[test]
    fn test_near_upper_bound() {
        let mut alp = 7.9;
        lsguard(&mut alp, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 7.666666666666667);
    }

    #[test]
    fn test_extrapolation() {
        let mut alp = 11.;
        let result = lsguard(&mut alp, &mut vec![2.0, 5.0], 20.0, 0.0, 0.5);
        assert_eq!(alp, 11.0);
    }

    #[test]
    fn test_max() {
        let mut alp = 15.0;
        lsguard(&mut alp, &mut vec![3.0, 8.0, 7.0], 8.0, 4.0, 0.1);
        assert_eq!(alp, 7.666666666666667);
    }

    #[test]
    fn test_min() {
        let mut alp = 1.0;
        lsguard(&mut alp, &mut vec![3.0, 8.0, 7.0], 10.0, 3.0, 0.1);
        assert_eq!(alp, 4.333333333333333);
    }
}