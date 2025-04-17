/**
 * Safeguards a step size within admissible bounds with additional safety constraints.
 *
 * This function safeguards a step size `alp` by:
 * 1. Computing extrapolation limits based on previous step sizes
 * 2. Enforcing both hard bounds (amin, amax) and soft bounds based on previous steps
 * 3. Ensuring sufficient distance from the minimum and maximum previous steps
 *
 * The implementation prevents steps too close to previous values, which helps
 * improve numerical stability in line search algorithms.
 *
 * # Arguments
 *
 * * `alp` - A mutable reference to the step size to be safeguarded
 * * `alist` - A reference to a vector of previous step sizes
 * * `amax` - The maximum allowed value for the step size
 * * `amin` - The minimum allowed value for the step size
 * * `small` - A relative distance factor that controls how close the new step can be to previous steps
 *
 * # Behavior
 *
 * The function:
 * - Finds the two smallest and two largest values in `alist`
 * - Creates soft bounds based on the range of previous steps and the `small` parameter
 * - Applies both hard bounds (`amin`/`amax`) and soft bounds to `alp`
 * - Ensures `alp` is sufficiently distant from the minimum and maximum previous steps
 *
 * # Assumptions
 *
 * * `alist` contains values within the admissible range [amin, amax]
 **/
pub(super) fn lsguard(alp: &mut f64, alist: &[f64], amax: f64, amin: f64, small: f64) {
    // We only need 2 minimums and 2 maximums, no need to sort()
    let (mut min, mut min2, mut max2, mut max) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &el in alist {
        // Note: DO not use else if as len of alist can be < 4
        if el < min {
            min2 = min;
            min = el;
        } else if el < min2 {
            min2 = el;
        }

        if el > max {
            max2 = max;
            max = el;
        } else if el > max2 {
            max2 = el;
        }
    }

    // Enforce extrapolation to be cautious
    let al: f64 = min - (max - min) / small;
    let au: f64 = max + (max - min) / small;

    *alp = alp.min(au).max(al).min(amax).max(amin);

    // Enforce some distance from end points
    // factor 1/3 ensures equal spacing if s=2 and the third point
    // in a safeguarded extrapolation is the maximum.
    if (*alp - min).abs() < small * (min2 - min) {
        *alp = (2.0 * min + min2) / 3.0;
    }

    if (*alp - max).abs() < small * (max - max2) {
        *alp = (2.0 * max + max2) / 3.0;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_0() {
        let mut alp = -1.0;
        lsguard(&mut alp, &vec![0.1, 0.0, 7.0], 5.0, 4.0, 10.0);
        assert_eq!(alp, 4.7);
    }

    #[test]
    fn test_coverage_1() {
        let mut alp = 13.0;
        lsguard(&mut alp, &vec![0.1, 0.0, 7.0], 5.0, 4.0, 100.0);
        assert_eq!(alp, 4.7);
    }

    #[test]
    fn test_coverage_2() {
        let mut alp = 13.0;
        lsguard(&mut alp, &vec![1.1, 0.8, 8.0], -5.0, 10.0, 100.0);
        assert_eq!(alp, 5.7);
    }

    #[test]
    fn test_within_bounds() {
        let mut alp = 5.0;
        lsguard(&mut alp, &vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 5.0);
    }

    #[test]
    fn test_below_extrapolated_bounds() {
        let mut alp = -1.0;
        lsguard(&mut alp, &vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 0.0);
    }

    #[test]
    fn test_above_extrapolated_bounds() {
        let mut alp = 12.0;
        lsguard(&mut alp, &vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 10.0);
    }

    #[test]
    fn test_near_lower_bound() {
        let mut alp = 3.1;
        lsguard(&mut alp, &vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 4.333333333333333);
    }

    #[test]
    fn test_near_upper_bound() {
        let mut alp = 7.9;
        lsguard(&mut alp, &vec![3.0, 8.0, 7.0], 10.0, 0.0, 0.1);
        assert_eq!(alp, 7.666666666666667);
    }

    #[test]
    fn test_extrapolation() {
        let mut alp = 11.;
        lsguard(&mut alp, &vec![2.0, 5.0], 20.0, 0.0, 0.5);
        assert_eq!(alp, 11.0);
    }

    #[test]
    fn test_max() {
        let mut alp = 15.0;
        lsguard(&mut alp, &vec![3.0, 8.0, 7.0], 8.0, 4.0, 0.1);
        assert_eq!(alp, 7.666666666666667);
    }

    #[test]
    fn test_min() {
        let mut alp = 1.0;
        lsguard(&mut alp, &vec![3.0, 8.0, 7.0], 10.0, 3.0, 0.1);
        assert_eq!(alp, 4.333333333333333);
    }
}