pub fn polint(x: &[f64; 3], f: &[f64; 3]) -> [f64; 3] {
    let mut d = [0.0; 3];
    // First step of interpolation
    d[0] = f[0];
    // Calculating the interpolated value for d[1]
    d[1] = (f[1] - f[0]) / (x[1] - x[0]);
    // Calculating f12 (the divided difference for f[2] and f[1])
    let f12 = (f[2] - f[1]) / (x[2] - x[1]);
    // Calculating the interpolated value for d[2]
    d[2] = (f12 - d[1]) / (x[2] - x[0]);

    d
}

pub fn polint1(x: &[f64; 3], f: &[f64; 3]) -> (f64, f64) {
    let f13 = (f[2] - f[0]) / (x[2] - x[0]);
    let f12 = (f[1] - f[0]) / (x[1] - x[0]);
    let f23 = (f[2] - f[1]) / (x[2] - x[1]);

    // Interpolated values following the formula from the Python code.
    let g = f13 + f12 - f23;
    let G = 2.0 * (f13 - f12) / (x[2] - x[1]);

    (g, G)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn polint_test_nominal_case() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let d = polint(&x, &f);
        let expected_d = [0.0, 0.23583295, 0.5989329];
        assert_abs_diff_eq!(&d[..], &expected_d[..], epsilon = 1e-6);
    }

    #[test]
    fn polint_test_nearly_identical_x_values() {
        let x = [0.5, 0.50000000001, 1.0];
        let f = [0.1, 0.2, 0.3];
        let d = polint(&x, &f);
        let expected_d = [1.00000000e-01, 9999999172.59636, -19999998344.792717];
        assert_abs_diff_eq!(&d[..], &expected_d[..], epsilon = 1e-6);
    }

    #[test]
    fn polint_test_large_numbers() {
        let x = [1e10, 1e10 + 1.0, 1e10 + 2.0];
        let f = [1e10, 1e10 + 0.5, 1e10 + 1.5];
        let d = polint(&x, &f);
        let expected_d = [1.0e+10, 5.0e-01, 2.5e-01];
        assert_abs_diff_eq!(&d[..], &expected_d[..], epsilon = 1e-6);
    }

    #[test]
    fn polint_test_small_numbers() {
        let x = [1e-10, 2e-10, 3e-10];
        let f = [1e-10, 1.5e-10, 2e-10];
        let d = polint(&x, &f);
        let expected_d = [1.00000000e-10, 5.00000000e-01, 8.32667268e-07];
        assert_abs_diff_eq!(&d[..], &expected_d[..], epsilon = 1e-6);
    }

    #[test]
    fn polint_test_mixed_sign_x() {
        let x = [-1.0, 0.5, 1.0];
        let f = [0.1, 0.2, 0.3];
        let d = polint(&x, &f);
        let expected_d = [0.1, 0.06666667, 0.06666667];
        assert_abs_diff_eq!(&d[..], &expected_d[..], epsilon = 1e-6);
    }

    #[test]
    fn polint_test_extra_elements_in_x_and_f() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let d = polint(&x, &f);
        let expected_d = [0.0, 0.23583295, 0.5989329];
        assert_abs_diff_eq!(&d[..], &expected_d[..], epsilon = 1e-6);
    }


    //---------------------------------------------------------
    #[test]
    fn polint1_test_nominal_case() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let (g, G) = polint1(&x, &f);
        let (expected_d1, expected_d2) = (-0.06363349972311094, 1.1978658059930671);
        assert_abs_diff_eq!(g, expected_d1, epsilon = 1e-5);
        assert_abs_diff_eq!(G, expected_d2, epsilon = 1e-5);
    }

    #[test]
    fn polint1_test_nearly_identical_x_values() {
        let x = [0.5, 0.50000000001, 1.0];
        let f = [0.1, 0.2, 0.3];
        let (g, G) = polint1(&x, &f);
        let (expected_d1, expected_d2) = (9999999172.796358, -39999996689.58544);
        assert_abs_diff_eq!(g, expected_d1, epsilon = 1e-5);
        assert_abs_diff_eq!(G, expected_d2, epsilon = 1e-5);
    }

    #[test]
    fn polint1_test_large_numbers() {
        let x = [1e10, 1e10 + 1.0, 1e10 + 2.0];
        let f = [1e10, 1e10 + 0.5, 1e10 + 1.5];
        let (g, G) = polint1(&x, &f);
        let (expected_d1, expected_d2) = (0.25, 0.5);
        assert_abs_diff_eq!(g, expected_d1, epsilon = 1e-5);
        assert_abs_diff_eq!(G, expected_d2, epsilon = 1e-5);
    }

    #[test]
    fn polint1_test_small_numbers() {
        let x = [1e-10, 2e-10, 3e-10];
        let f = [1e-10, 1.5e-10, 2e-10];
        let (g, G) = polint1(&x, &f);
        let (expected_d1, expected_d2) = (0.4999999999999999, 3.33066907387547e-06);
        assert_abs_diff_eq!(g, expected_d1, epsilon = 1e-5);
        assert_abs_diff_eq!(G, expected_d2, epsilon = 1e-5);
    }

    #[test]
    fn polint1_test_mixed_sign_x() {
        let x = [-1.0, 0.5, 1.0];
        let f = [0.1, 0.2, 0.3];
        let (g, G) = polint1(&x, &f);
        let (expected_d1, expected_d2) = (-0.0333333333333333, 0.1333333333333333);
        assert_abs_diff_eq!(g, expected_d1, epsilon = 1e-5);
        assert_abs_diff_eq!(G, expected_d2, epsilon = 1e-5);
    }

    #[test]
    fn polint1_test_extra_elements_in_x_and_f() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let (g, G) = polint1(&x, &f);
        let (expected_d1, expected_d2) = (-0.06363349972311094, 1.1978658059930671);
        assert_abs_diff_eq!(g, expected_d1, epsilon = 1e-5);
        assert_abs_diff_eq!(G, expected_d2, epsilon = 1e-5);
    }
}
