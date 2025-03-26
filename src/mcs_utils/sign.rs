/// Returns SIGN(X)
///
/// # Arguments
///
/// * `x` - Any number
///
/// # Returns
///
/// 1 if x is greater than zero (including f64::INFINITY)
///
/// 0 if x equals 0.0 or -0.0
///
/// -1 if x is less than zero (including f64::NEG_INFINITY)
///
///
/// # Note
///
/// Default Rust .signum() function does not work as it maps -0.0 to -1.0 and 0.0 to 1.0
#[inline]
pub const fn sign(x: f64) -> f64 {
    match x {
        0.0 => 0.0, // captures both -0.0 and 0.0
        _ => x.signum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_1() {
        assert_eq!(sign(-0.0), 0.0);
    }

    #[test]
    fn sign_2() {
        assert_eq!(sign(0.0), 0.0);
    }
    #[test]
    fn sign_3() {
        assert_eq!(sign(f64::INFINITY), 1.0);
    }

    #[test]
    fn sign_4() {
        assert_eq!(sign(f64::NEG_INFINITY), -1.0);
    }

    #[test]
    fn sign_5() {
        assert_eq!(sign(123.43), 1.0);
    }

    #[test]
    fn sign_6() {
        assert_eq!(sign(-0.00023), -1.0);
    }
}