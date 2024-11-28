use std::cmp::Ordering;

pub fn lsguard(mut alp: f64, alist: &mut Vec<f64>, amax: f64, amin: f64, small: f64) -> f64 {
    alist.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let s = alist.len();

    // Enforce extrapolation to be cautious
    let al = alist[0] - (alist[s - 1] - alist[0]) / small;
    let au = alist[s - 1] + (alist[s - 1] - alist[0]) / small;
    alp = alp.max(al).min(au);
    alp = alp.max(amin).min(amax);

    // Enforce some distance from endpoints
    if (alp - alist[0]).abs() < small * (alist[1] - alist[0]) {
        alp = (2.0 * alist[0] + alist[1]) / 3.0;
    }

    if (alp - alist[s - 1]).abs() < small * (alist[s - 1] - alist[s - 2]) {
        alp = (2.0 * alist[s - 1] + alist[s - 2]) / 3.0;
    }

    alp
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_within_bounds() {
        assert_relative_eq!(lsguard(5.0, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1), 5.0);
    }

    #[test]
    fn test_below_extrapolated_bounds() {
        let result = lsguard(-1.0, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_relative_eq!(result, 0.0);
    }

    #[test]
    fn test_above_extrapolated_bounds() {
        let result = lsguard(12.0, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_relative_eq!(result, 10.0);
    }

    #[test]
    fn test_near_lower_bound() {
        let result = lsguard(3.1, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_relative_eq!(result, 4.333333333333333);
    }

    #[test]
    fn test_near_upper_bound() {
        let result = lsguard(7.9, &mut vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_relative_eq!(result, 7.666666666666666);
    }

    #[test]
    fn test_extrapolation() {
        let result = lsguard(11.0, &mut vec![2.0, 5.0], 20.0, 0.0, 0.5);
        assert_relative_eq!(result, 11.0);
    }

    #[test]
    fn test_max_and_min() {
        assert_relative_eq!(lsguard(15.0, &mut vec![3.0, 8.0, 7.0], 8.0, 4.0, 0.1), 7.666666666666666);
        assert_relative_eq!(lsguard(1.0, &mut vec![3.0, 8.0, 7.0], 10.0, 3.0, 0.1), 4.333333333333333);
    }
}