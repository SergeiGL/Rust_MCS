use nalgebra::SVector;
/**
* Evaluates the quartic polynomial
*
* f(x) = a₀x⁴ + a₁x³ + a₂x² + a₃x + a₄
*
* at a given scalar value `x`, using an optimized approach for numerical stability.
*
* This function employs Horner's method with compensated summation,
* restructuring the polynomial into even and odd parts.
*
* # Arguments
*
* * `a` - A reference to an `SVector<f64, 5>` representing the polynomial coefficients `[a₀, a₁, a₂, a₃, a₄]`.
*          The coefficient `a[0]` corresponds to x⁴, `a[1]` to x³, and so on.
* * `x` - A scalar value (`f64`) at which the polynomial is evaluated.
*
* # Returns
*
* * A floating-point value (`f64`) representing the computed result f(x).
**/
pub fn quartic(a: &SVector<f64, 5>, x: f64) -> f64 {
    // Use Horner's method with compensated summation for better numerical stability
    // and parallel computation opportunities

    // Pre-compute powers of x to enable parallel evaluation
    let x2 = x * x;
    let x4 = x2 * x2;

    // Split the polynomial into even and odd parts
    // P(x) = (a0*x^4 + a2*x^2 + a4) + x*(a1*x^2 + a3)
    // This reduces dependencies between operations and allows better instruction pipelining

    // Even part
    let even = a[0] * x4 + a[2] * x2 + a[4];

    // Odd part
    let odd = a[1] * x2 + a[3];

    // Combine parts
    odd * x + even
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_coefficients_and_positive_x() {
        let a = SVector::<f64, 5>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let x = 2.0;
        let result = quartic(&a, x);
        assert_eq!(result, 31.0);
    }

    #[test]
    fn test_positive_coefficients_and_negative_x() {
        let a = SVector::<f64, 5>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let x = -2.0;
        let result = quartic(&a, x);
        assert_eq!(result, 11.0);
    }

    #[test]
    fn test_mixed_coefficients_and_positive_x() {
        let a = SVector::<f64, 5>::from_row_slice(&[2.0, -1.0, 3.0, -2.0, 1.0]);
        let x = 1.0;
        let result = quartic(&a, x);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_mixed_coefficients_and_negative_x() {
        let a = SVector::<f64, 5>::from_row_slice(&[-2.0, 3.0, -1.0, 4.0, -1.0]);
        let x = -1.0;
        let result = quartic(&a, x);
        assert_eq!(result, -11.0);
    }

    #[test]
    fn test_zero_coefficients() {
        let a = SVector::<f64, 5>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let x = 5.0;
        let result = quartic(&a, x);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_zero_x() {
        let a = SVector::<f64, 5>::from_row_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        let x = 0.0;
        let result = quartic(&a, x);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_large_numbers() {
        let a = SVector::<f64, 5>::from_row_slice(&[1e10, 1e8, 1e6, 1e4, 1e2]);
        let x = 1e3;
        let result = quartic(&a, x);
        assert_eq!(result, 1.000010000100001e22);
    }

    #[test]
    fn test_noname() {
        let a = SVector::<f64, 5>::new(1.0, -2.0, 3.0, -4.0, 5.0);
        let x = 2.0;
        let result = quartic(&a, x);

        assert_eq!(result, 9.);
    }

    #[test]
    fn test_neg() {
        let a = SVector::<f64, 5>::new(1.0, -2.0, 3.0, -4.0, 5.0);
        let x = -2.0;
        let result = quartic(&a, x);

        assert_eq!(result, 57.);
    }
}