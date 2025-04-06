use std::cmp::Ordering;

/**
 * Computes an interpolated point and its associated interpolation factor within an interval.
 *
 * This function determines an interpolation factor (`fac`) based on a comparison between two function
 * values: `flist_i` and `flist_i_plus_1`.
 *
 *  The interpolated point (`alp`) is then calculated by linearly interpolating between `alist_i` and
 * `alist_i_plus_1` using the determined factor:
 *
 *  alp = alist_i + fac * (alist_i_plus_1 - alist_i)
 *
 * # Arguments
 *
 * * `alist_i` - The value at the beginning of the interval.
 * * `alist_i_plus_1` - The value at the end of the interval.
 * * `flist_i` - The function value corresponding to `alist_i`.
 * * `flist_i_plus_1` - The function value corresponding to `alist_i_plus_1`.
 * * `short` - The weight used when `flist_i` is less than `flist_i_plus_1`; influences the interpolation.
 *
 * # Returns
 *
 * * `alp` - The interpolated point.
 * * `fac` - The interpolation factor determined by the comparison of `flist_i` and `flist_i_plus_1`.
 */
#[inline]
pub fn lssplit(alist_i: f64, alist_i_plus_1: f64, flist_i: f64, flist_i_plus_1: f64, short: f64) -> (f64, f64) {
    let fac = match flist_i.total_cmp(&flist_i_plus_1) {
        Ordering::Less => short,
        Ordering::Greater => 1.0 - short,
        Ordering::Equal => 0.5,
    };

    let alp = alist_i + fac * (alist_i_plus_1 - alist_i);
    (alp, fac)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flist_increasing() {
        let result = lssplit(10.0, 20.0, 1.0, 2.0, 0.3);
        assert_eq!(result, (13.0, 0.3));
    }

    #[test]
    fn test_flist_decreasing() {
        let result = lssplit(10.0, 20.0, 2.0, 1.0, 0.3);
        assert_eq!(result, (17.0, 0.7));
    }

    #[test]
    fn test_flist_equal() {
        let result = lssplit(10.0, 20.0, 2.0, 2.0, 0.3);
        assert_eq!(result, (15.0, 0.5));
    }
}