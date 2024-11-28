#[derive(Debug)]
pub enum LineSearchError {
    ZeroSearchDirection,
    NoAdmissibleStep,
}


pub fn lsrange(
    x: &[f64],
    p: &[f64],
    xl: &[f64],
    xu: &[f64],
    bend: bool,
) -> Result<(f64, //amin
             f64, //amax
             f64, //scale
), LineSearchError> {
    // Check for zero search direction
    if p.into_iter().fold(0.0, |acc: f64, &x| acc.max(x.abs())) == 0.0 {
        return Err(LineSearchError::ZeroSearchDirection);
    }

    // Find sensible step size scale
    let nonzero_mask = p.iter().map(|&x| x != 0.0).collect::<Vec<_>>();
    let pp = p.iter()
        .zip(nonzero_mask.iter())
        .filter(|(_, &mask)| mask)
        .map(|(&val, _)| val.abs())
        .collect::<Vec<_>>();

    let u = x.iter()
        .zip(nonzero_mask.iter())
        .filter(|(_, &mask)| mask)
        .zip(pp.iter())
        .map(|((x_val, _), p_val)| x_val.abs() / p_val)
        .collect::<Vec<_>>();

    let mut scale = u.iter().copied().fold(f64::INFINITY, f64::min);

    if scale == 0.0 {
        let new_u: Vec<f64> = u.iter()
            .zip(pp.iter())
            .map(|(&u_val, &p_val)| if u_val == 0.0 { 1.0 / p_val } else { u_val })
            .collect();
        scale = new_u.iter().copied().fold(f64::INFINITY, f64::min);
    }

    if !bend {
        // Truncated line search
        let mut amin = f64::NEG_INFINITY;
        let mut amax = f64::INFINITY;

        for (i, &p_i) in p.iter().enumerate() {
            if p_i > 0.0 {
                amin = amin.max((xl[i] - x[i]) / p_i);
                amax = amax.min((xu[i] - x[i]) / p_i);
            } else if p_i < 0.0 {
                amin = amin.max((xu[i] - x[i]) / p_i);
                amax = amax.min((xl[i] - x[i]) / p_i);
            }
        }

        if amin > amax {
            return Err(LineSearchError::NoAdmissibleStep);
        }

        Ok((amin, amax, scale))
    } else {
        // Bent line search
        let mut amin = f64::INFINITY;
        let mut amax = f64::NEG_INFINITY;

        for (i, &p_i) in p.iter().enumerate() {
            if p_i > 0.0 {
                amin = amin.min((xl[i] - x[i]) / p_i);
                amax = amax.max((xu[i] - x[i]) / p_i);
            } else if p_i < 0.0 {
                amin = amin.min((xu[i] - x[i]) / p_i);
                amax = amax.max((xl[i] - x[i]) / p_i);
            }
        }

        Ok((amin, amax, scale))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_0() {
        let xl = vec![0.0, 0.0];
        let xu = vec![10.0, 10.0];
        let x = vec![5.0, 5.0];
        let p = vec![1.0, 1.0];
        let bend = false;

        let result = lsrange(&x, &p, &xl, &xu, bend).expect("Line search failed");

        let (amin, amax, scale) = result;

        // Check that returned values are as expected
        assert_abs_diff_eq!(amin, -5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(amax, 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(scale, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_3() {
        let xl = vec![0.0, 0.0];
        let xu = vec![10.0, 10.0];
        let x = vec![5.0, 5.0];
        let p = vec![0.0, 0.0]; // zero search direction
        let bend = false;

        // Expect an error indicating a zero search direction
        let result = lsrange(&x, &p, &xl, &xu, bend);
        assert!(matches!(result, Err(LineSearchError::ZeroSearchDirection)));
    }

    #[test]
    fn test_5() {
        let xl = vec![0.0, 2.0];
        let xu = vec![10.0, 20.0];
        let x = vec![-5.0, 0.5];
        let p = vec![-10.0, 1.0];
        let bend = true;

        let result = lsrange(&x, &p, &xl, &xu, bend).expect("Line search failed");

        let (amin, amax, scale) = result;

        // Check that returned values are as expected
        assert_abs_diff_eq!(amin, -1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(amax, 19.5, epsilon = 1e-6);
        assert_abs_diff_eq!(scale, 0.5, epsilon = 1e-6);
    }
}