use crate::mcs_utils::sign::sign;

/// Computes for real x and real or infinite y two points x1 and x2 in
/// \[min(x,y),max(x,y)\] that are neither too close nor too far away from x
/// # Arguments
/// * `const F: usize` - controls the threshold for when values are considered "too far" from each other.
/// A multiplier that helps to determine when the algorithm should adjust the second point to avoid points being too far apart
/// * `x` - Real number
/// * `y` - Real or infinite number
///
/// # Returns
///
/// * `x1` - First point in \[min(x,y),max(x,y)\] that is neither too close nor too far away from x
/// * `x2` - Second point in \[min(x,y),max(x,y)\] that is neither too close nor too far away from x
pub(super) fn subint(x: f64, y: f64) ->
(
    f64, // x1 new
    f64  // x2 new
) {
    let mut x2 = y;

    if 1000_f64 * x.abs() < 1.0 {
        if y.abs() > 1000_f64 { x2 = sign(y) } // very important that this if is nested ( && will not work)
    } else if y.abs() > 1000_f64 * x.abs() { // TODO: python version has a mistake here
        x2 = 10.0 * sign(y) * x.abs();
    }
    
    (x + (x2 - x) / 10.0, x2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subint_test_0() {
        let x1 = 1.0;
        let x2 = 2.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 1.1);
        assert_eq!(x2_new, 2.0);
    }

    #[test]
    fn subint_test_1() {
        let x1 = 0.0000001;
        let x2 = 2000.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 0.10000009000000001);
        assert_eq!(x2_new, 1.0);
    }


    #[test]
    fn subint_test_2() {
        let x1 = -1.0;
        let x2 = 50.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 4.1);
        assert_eq!(x2_new, 50.0);
    }

    #[test]
    fn subint_test_3() {
        let x1 = -0.00000001;
        let x2 = 132455.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, 0.099999991);
        assert_eq!(x2_new, 1.0);
    }

    #[test]
    fn subint_test_real_mistake_1() {
        let x1 = -0.0002;
        let x2 = -20.0;
        let (x1_new, x2_new) = subint(x1, x2);
        assert_eq!(x1_new, -2.0001800000000003);
        assert_eq!(x2_new, -20.0);
    }
}