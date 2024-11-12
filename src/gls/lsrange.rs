use ndarray::Array1;
use std::error::Error;

#[derive(Debug)]
pub enum LineSearchError {
    ZeroSearchDirection,
    NoAdmissibleStep,
    Generic(String),
}

pub struct LineSearchResult {
    pub amin: f64,
    pub amax: f64,
    pub scale: f64,
}

pub fn line_search(
    x: &Array1<f64>,
    p: &Array1<f64>,
    xl: &Array1<f64>,
    xu: &Array1<f64>,
    bend: bool,
) -> Result<LineSearchResult, LineSearchError> {
    // Check for zero search direction
    if p.fold(0.0, |acc: f64, &x| acc.max(x.abs())) == 0.0 {
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

        Ok(LineSearchResult { amin, amax, scale })
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

        Ok(LineSearchResult { amin, amax, scale })
    }
}

// Helper function to compute the next point
pub fn compute_next_point(
    x: &Array1<f64>,
    p: &Array1<f64>,
    alpha: f64,
    xl: &Array1<f64>,
    xu: &Array1<f64>,
) -> Array1<f64> {
    let mut next_point = x + &(p * alpha);
    for i in 0..next_point.len() {
        next_point[i] = next_point[i].max(xl[i]).min(xu[i]);
    }
    next_point
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_zero_search_direction() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let xl = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let xu = Array1::from_vec(vec![4.0, 4.0, 4.0]);

        let result = line_search(&x, &p, &xl, &xu, false);
        assert!(matches!(result, Err(LineSearchError::ZeroSearchDirection)));
    }

    #[test]
    fn test_truncated_line_search() {
        let x = Array1::from_vec(vec![1.0, 1.0]);
        let p = Array1::from_vec(vec![1.0, -1.0]);
        let xl = Array1::from_vec(vec![0.0, 0.0]);
        let xu = Array1::from_vec(vec![2.0, 2.0]);

        let result = line_search(&x, &p, &xl, &xu, false).unwrap();
        assert_relative_eq!(result.amin, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.amax, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bent_line_search() {
        let x = Array1::from_vec(vec![1.0, 1.0]);
        let p = Array1::from_vec(vec![1.0, -1.0]);
        let xl = Array1::from_vec(vec![0.0, 0.0]);
        let xu = Array1::from_vec(vec![2.0, 2.0]);

        let result = line_search(&x, &p, &xl, &xu, true).unwrap();
        assert_relative_eq!(result.amin, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.amax, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_higher_dim() {
        let x = Array1::from_vec(vec![1.0, 1.0]);
        let p = Array1::from_vec(vec![1.0, -1.0]);
        let xl = Array1::from_vec(vec![0.0, 0.0]);
        let xu = Array1::from_vec(vec![2.0, 2.0]);

        let result = line_search(&x, &p, &xl, &xu, true).unwrap();
        assert_relative_eq!(result.amin, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.amax, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_large_values() {
        let x = Array1::from_vec(vec![1e10, 1e12]);
        let p = Array1::from_vec(vec![1e10, -1e12]);
        let xl = Array1::from_vec(vec![1e9, 1e11]);
        let xu = Array1::from_vec(vec![1e11, 1e13]);

        let result = line_search(&x, &p, &xl, &xu, false).unwrap();
        assert!(result.amin.is_finite() && result.amax.is_finite());
        assert_relative_eq!(result.scale, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_all_negative_directions() {
        let x = Array1::from_vec(vec![2.0, 2.0]);
        let p = Array1::from_vec(vec![-1.0, -1.0]);
        let xl = Array1::from_vec(vec![0.0, 0.0]);
        let xu = Array1::from_vec(vec![3.0, 3.0]);

        let result = line_search(&x, &p, &xl, &xu, false).unwrap();
        assert_relative_eq!(result.amin, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.amax, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_different_shape_handling() {
        let x = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let p = Array1::from_vec(vec![-1.0, -1.0, -1.0]);
        let xl = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let xu = Array1::from_vec(vec![2.0, 2.0, 2.0]);

        let result = line_search(&x, &p, &xl, &xu, false).unwrap();
        assert_relative_eq!(result.amin, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.amax, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_scale_handling() {
        let x = Array1::from_vec(vec![0.0, 0.0]);
        let p = Array1::from_vec(vec![0.0, 0.0]);
        let xl = Array1::from_vec(vec![0.0, 0.0]);
        let xu = Array1::from_vec(vec![2.0, 2.0]);

        let result = line_search(&x, &p, &xl, &xu, false);
        assert!(matches!(result, Err(LineSearchError::ZeroSearchDirection)));
    }

    #[test]
    fn test_1d_regular() {
        let xl = Array1::from_vec(vec![0.0]);
        let xu = Array1::from_vec(vec![10.0]);
        let x = Array1::from_vec(vec![5.0]);
        let p = Array1::from_vec(vec![1.0]);

        let result = line_search(&x, &p, &xl, &xu, false).unwrap();
        assert_relative_eq!(result.amin, -5.0, epsilon = 1e-10);
        assert_relative_eq!(result.amax, 5.0, epsilon = 1e-10);
        assert_relative_eq!(result.scale, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_3d_bent() {
        let xl = Array1::from_vec(vec![0.0, -1.0, -5.0]);
        let xu = Array1::from_vec(vec![10.0, 5.0, 15.0]);
        let x = Array1::from_vec(vec![2.0, 1.0, 0.0]);
        let p = Array1::from_vec(vec![1.0, -2.0, 3.0]);

        let result = line_search(&x, &p, &xl, &xu, true).unwrap();
        assert_relative_eq!(result.amin, -2.0, epsilon = 1e-10);
        assert_relative_eq!(result.amax, 8.0, epsilon = 1e-10);
        assert_relative_eq!(result.scale, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_4d_zero_components() {
        let xl = Array1::from_vec(vec![0.0, -1.0, -5.0, -10.0]);
        let xu = Array1::from_vec(vec![10.0, 5.0, 15.0, 10.0]);
        let x = Array1::from_vec(vec![2.0, 1.0, 0.0, 5.0]);
        let p = Array1::from_vec(vec![1.0, 0.0, 3.0, -2.0]);

        let result = line_search(&x, &p, &xl, &xu, false).unwrap();
        assert!(result.amin.is_finite());
        assert!(result.amax.is_finite());
        assert!(result.scale > 0.0);
    }

    #[test]
    fn test_5d_edge_case() {
        let xl = Array1::from_vec(vec![0.0, -1.0, -5.0, -10.0, -100.0]);
        let xu = Array1::from_vec(vec![10.0, 5.0, 15.0, 10.0, 100.0]);
        let x = Array1::from_vec(vec![0.0, 5.0, 0.0, -10.0, 0.0]);
        let p = Array1::from_vec(vec![1.0, -1.0, 2.0, 3.0, 4.0]);

        let result = line_search(&x, &p, &xl, &xu, false).unwrap();
        assert!(result.amin.is_finite());
        assert!(result.amax.is_finite());
        assert!(result.scale > 0.0);
    }
}