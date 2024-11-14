use ndarray::Array1;

pub fn quartic(a: &Array1<f64>, x: f64) -> f64 {
    (((a[0] * x + a[1]) * x + a[2]) * x + a[3]) * x + a[4]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_positive_coefficients_and_positive_x() {
        let a = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let x = 2.0;
        let result = quartic(&a, x);
        assert_eq!(result, 31.0);
    }

    #[test]
    fn test_positive_coefficients_and_negative_x() {
        let a = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let x = -2.0;
        let result = quartic(&a, x);
        assert_eq!(result, 11.0);
    }

    #[test]
    fn test_mixed_coefficients_and_positive_x() {
        let a = array![2.0, -1.0, 3.0, -2.0, 1.0];
        let x = 1.0;
        let result = quartic(&a, x);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_mixed_coefficients_and_negative_x() {
        let a = array![-2.0, 3.0, -1.0, 4.0, -1.0];
        let x = -1.0;
        let result = quartic(&a, x);
        assert_eq!(result, -11.0);
    }

    #[test]
    fn test_zero_coefficients() {
        let a = array![0.0, 0.0, 0.0, 0.0, 0.0];
        let x = 5.0;
        let result = quartic(&a, x);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_zero_x() {
        let a = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let x = 0.0;
        let result = quartic(&a, x);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_large_numbers() {
        let a = array![1e10, 1e8, 1e6, 1e4, 1e2];
        let x = 1e3;
        let result = quartic(&a, x);
        assert_relative_eq!(result, 1.000010000100001e22);
    }
}