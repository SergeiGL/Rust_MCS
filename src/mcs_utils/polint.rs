pub(super) fn polint(x: &[f64; 3], f: &[f64; 3]) -> [f64; 3] {
    [
        f[0],
        (f[1] - f[0]) / (x[1] - x[0]),
        ((f[2] - f[1]) / (x[2] - x[1]) - (f[1] - f[0]) / (x[1] - x[0])) / (x[2] - x[0])
    ]
}

pub(super) fn polint1(x: &[f64; 3], f: &[f64; 3]) -> (f64, f64) {
    let f13: f64 = (f[2] - f[0]) / (x[2] - x[0]);
    let f12: f64 = (f[1] - f[0]) / (x[1] - x[0]);
    let f23: f64 = (f[2] - f[1]) / (x[2] - x[1]);

    // Interpolated values following the formula from the Python code.
    let g: f64 = f13 + f12 - f23;
    let G: f64 = 2.0 * (f13 - f12) / (x[2] - x[1]);

    (g, G)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polint_test_nominal_case() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let d = polint(&x, &f);
        let expected_d = [0., 0.23583295177515584, 0.5989329029965336];
        assert_eq!(d, expected_d);
    }

    #[test]
    fn polint_test_nearly_identical_x_values() {
        let x = [0.5, 0.50000000001, 1.0];
        let f = [0.1, 0.2, 0.3];
        let d = polint(&x, &f);
        let expected_d = [1.00000000e-01, 9999999172.59636, -19999998344.792717];
        assert_eq!(d, expected_d);
    }

    #[test]
    fn polint_test_large_numbers() {
        let x = [1e10, 1e10 + 1.0, 1e10 + 2.0];
        let f = [1e10, 1e10 + 0.5, 1e10 + 1.5];
        let d = polint(&x, &f);
        let expected_d = [1.0e+10, 5.0e-01, 2.5e-01];
        assert_eq!(d, expected_d);
    }

    #[test]
    fn polint_test_small_numbers() {
        let x = [1e-10, 2e-10, 3e-10];
        let f = [1e-10, 1.5e-10, 2e-10];
        let d = polint(&x, &f);
        let expected_d = [1.0000000000000000e-10, 4.9999999999999994e-01, 8.3266726846886751e-07];
        assert_eq!(d, expected_d);
    }

    #[test]
    fn polint_test_mixed_sign_x() {
        let x = [-1.0, 0.5, 1.0];
        let f = [0.1, 0.2, 0.3];
        let d = polint(&x, &f);
        let expected_d = [0.1, 0.06666666666666667, 0.06666666666666665];
        assert_eq!(d, expected_d);
    }

    #[test]
    fn polint_test_extra_elements_in_x_and_f() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let d = polint(&x, &f);
        let expected_d = [0., 0.23583295177515584, 0.5989329029965336];
        assert_eq!(d, expected_d);
    }


    //---------------------------------------------------------
    #[test]
    fn polint1_test_nominal_case() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let (g, G) = polint1(&x, &f);
        let (expected_g, expected_G) = (-0.06363349972311094, 1.1978658059930671);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
    }

    #[test]
    fn polint1_test_nearly_identical_x_values() {
        let x = [0.5, 0.50000000001, 1.0];
        let f = [0.1, 0.2, 0.3];
        let (g, G) = polint1(&x, &f);
        let (expected_g, expected_G) = (9999999172.796358, -39999996689.58544);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
    }

    #[test]
    fn polint1_test_large_numbers() {
        let x = [1e10, 1e10 + 1.0, 1e10 + 2.0];
        let f = [1e10, 1e10 + 0.5, 1e10 + 1.5];
        let (g, G) = polint1(&x, &f);
        let (expected_g, expected_G) = (0.25, 0.5);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
    }

    #[test]
    fn polint1_test_small_numbers() {
        let x = [1e-10, 2e-10, 3e-10];
        let f = [1e-10, 1.5e-10, 2e-10];
        let (g, G) = polint1(&x, &f);
        let (expected_g, expected_G) = (0.4999999999999999, 3.33066907387547e-06);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
    }

    #[test]
    fn polint1_test_mixed_sign_x() {
        let x = [-1.0, 0.5, 1.0];
        let f = [0.1, 0.2, 0.3];
        let (g, G) = polint1(&x, &f);
        let (expected_g, expected_G) = (-0.0333333333333333, 0.1333333333333333);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
    }

    #[test]
    fn polint1_test_extra_elements_in_x_and_f() {
        let x = [0.0, 0.5, 1.0];
        let f = [0.0, 0.11791647588757792, 0.5352994032734226];
        let (g, G) = polint1(&x, &f);
        let (expected_g, expected_G) = (-0.06363349972311094, 1.1978658059930671);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
    }
}
