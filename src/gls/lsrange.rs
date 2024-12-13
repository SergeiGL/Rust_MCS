use nalgebra::SVector;

#[derive(Debug)]
pub enum LineSearchError {
    ZeroSearchDirection,
    NoAdmissibleStep,
}

pub fn lsrange<const N: usize>(
    x: &[f64; N],
    p: &SVector<f64, N>,
    u: &[f64; N],
    v: &[f64; N],
    bend: bool,
) -> Result<(
    f64, // amin
    f64, // amax
    f64, // scale
), LineSearchError> {

    // Check for zero search direction
    if p.into_iter().fold(0.0, |acc: f64, &x| acc.max(x.abs())) == 0.0 {
        return Err(LineSearchError::ZeroSearchDirection);
    }

    // Find sensible STEP size scale
    let nonzero_mask = p.iter().map(|&x| x != 0.0).collect::<Vec<_>>();
    let pp = p.iter()
        .zip(nonzero_mask.iter())
        .filter(|(_, &mask)| mask)
        .map(|(&val, _)| val.abs())
        .collect::<Vec<_>>();

    let u_vec = x.iter()
        .zip(nonzero_mask.iter())
        .filter(|(_, &mask)| mask)
        .zip(pp.iter())
        .map(|((x_val, _), p_val)| x_val.abs() / p_val)
        .collect::<Vec<_>>();

    let mut scale = u_vec.iter().copied().fold(f64::INFINITY, f64::min);

    if scale == 0.0 {
        let new_u: Vec<f64> = u_vec.iter()
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
                amin = amin.max((u[i] - x[i]) / p_i);
                amax = amax.min((v[i] - x[i]) / p_i);
            } else if p_i < 0.0 {
                amin = amin.max((v[i] - x[i]) / p_i);
                amax = amax.min((u[i] - x[i]) / p_i);
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
                amin = amin.min((u[i] - x[i]) / p_i);
                amax = amax.max((v[i] - x[i]) / p_i);
            } else if p_i < 0.0 {
                amin = amin.min((v[i] - x[i]) / p_i);
                amax = amax.max((u[i] - x[i]) / p_i);
            }
        }

        Ok((amin, amax, scale))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let xl = [0.0, 0.0];
        let xu = [10.0, 10.0];
        let x = [5.0, 5.0];
        let p = SVector::<f64, 2>::from_row_slice(&[1.0, 1.0]);
        let bend = false;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend).expect("Line search failed");

        assert_eq!(amin, -5.0);
        assert_eq!(amax, 5.0);
        assert_eq!(scale, 5.0);
    }

    #[test]
    fn test_3() {
        let xl = [0.0, 0.0];
        let xu = [10.0, 10.0];
        let x = [5.0, 5.0];
        let p = SVector::<f64, 2>::repeat(0.0); // zero search direction
        let bend = false;

        let result = lsrange(&x, &p, &xl, &xu, bend);

        assert!(matches!(result, Err(LineSearchError::ZeroSearchDirection)));
    }

    #[test]
    fn test_5() {
        let xl = [0.0, 2.0];
        let xu = [10.0, 20.0];
        let x = [-5.0, 0.5];
        let p = SVector::<f64, 2>::from_row_slice(&[-10.0, 1.0]);
        let bend = true;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend).expect("Line search failed");

        assert_eq!(amin, -1.5);
        assert_eq!(amax, 19.5);
        assert_eq!(scale, 0.5);
    }
}