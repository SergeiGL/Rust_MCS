pub fn quadmin(a: f64, b: f64, d: &[f64; 3], x0: &[f64; 3]) -> f64 {
    if d[2] == 0.0 {
        if d[1] > 0.0 {
            a
        } else {
            b
        }
    } else if d[2] > 0.0 {
        let x1 = 0.5 * (x0[0] + x0[1]) - 0.5 * d[1] / d[2];
        if a <= x1 && x1 <= b {
            x1
        } else if quadpol(a, &d, &x0) < quadpol(b, &d, &x0) {
            a
        } else {
            b
        }
    } else {
        if quadpol(a, &d, &x0) < quadpol(b, &d, &x0) {
            a
        } else {
            b
        }
    }
}

pub fn quadpol(x: f64, d: &[f64; 3], x0: &[f64; 3]) -> f64 {
    d[0] + d[1] * (x - x0[0]) + d[2] * (x - x0[0]) * (x - x0[1])
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1_no_quadratic_term_d2_zero() {
        let a = 0.1;
        let b = 0.5;
        let d = [0.0, 1.0, 0.0];
        let x0 = [0.0, 0.5, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_2_no_quadratic_term_d2_zero() {
        let a = 0.1;
        let b = 0.5;
        let d = [0.0, -0.2, 0.0];  // d[2] is 0, d[1] is <= 0
        let x0 = [0.0, 0.5, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, b,);
    }

    #[test]
    fn test_3_quadratic_term_d2_positive() {
        let a = 0.1;
        let b = 1.0;
        let d = [0.01, 0.5, 0.3];
        let x0 = [0.1, 0.9, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_4_quadratic_term_d2_positive_x1_less_than_a() {
        let a = 0.5;
        let b = 1.0;
        let d = [0.01, -1.0, 2.0];
        let x0 = [0.1, 0.2, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_5_quadratic_term_d2_positive_x1_greater_than_b() {
        let a = 0.1;
        let b = 0.5;
        let d = [0.01, 1.0, 0.01];
        let x0 = [0.0, 0.5, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_6_d2_negative() {
        let a = -10.0;
        let b = 1.0;
        let d = [0.1, 1.0, -0.5];
        let x0 = [0.0, 0.5, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_7() {
        let a = -10.0;
        let b = 10.0;
        let d = [0.321, -1.0, 10.5];
        let x0 = [0.0, 0.5, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, 0.2976190476190476);
    }

    #[test]
    fn test_8() {
        let a = -13.5;
        let b = 13.0;
        let d = [0.0, 0.0, 10.5];
        let x0 = [0.0, 0.5, 1.0];
        let result = quadmin(a, b, &d, &x0);
        assert_eq!(result, 0.25);
    }
}
