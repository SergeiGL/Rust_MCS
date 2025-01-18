use nalgebra::SVector;

pub fn lsrange<const N: usize>(
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    bend: bool,
) -> (
    f64, // amin
    f64, // amax
    f64, // scale
) {
    // Check for zero search direction
    debug_assert!(p.into_iter().fold(0.0_f64, |acc, &p_i| acc.max(p_i.abs())) != 0.0);

    let scale = x.iter()
        .zip(p)
        .filter(|(_, &p_i)| p_i != 0.0)
        .map(
            |(x_i, p_i)|
                match x_i.abs() / p_i.abs() {
                    0.0 => 1.0 / p_i.abs(),
                    num => num
                }
        )
        .min_by(|a, b| a.total_cmp(b)).unwrap();

    if !bend {
        let mut amin = f64::NEG_INFINITY;
        let mut amax = f64::INFINITY;

        for (i, &p_i) in p.iter().enumerate() {
            match p_i.partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => {
                    amin = amin.max((u[i] - x[i]) / p_i);
                    amax = amax.min((v[i] - x[i]) / p_i);
                }
                std::cmp::Ordering::Less => {
                    amin = amin.max((v[i] - x[i]) / p_i);
                    amax = amax.min((u[i] - x[i]) / p_i);
                }
            }
        }
        debug_assert!(amin < amax);
        (amin, amax, scale)
    } else {
        let mut amin = f64::INFINITY;
        let mut amax = f64::NEG_INFINITY;

        for (i, &p_i) in p.iter().enumerate() {
            match p_i.partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => {
                    amin = amin.min((u[i] - x[i]) / p_i);
                    amax = amax.max((v[i] - x[i]) / p_i);
                }
                std::cmp::Ordering::Less => {
                    amin = amin.min((v[i] - x[i]) / p_i);
                    amax = amax.max((u[i] - x[i]) / p_i);
                }
            }
        }
        (amin, amax, scale)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let xl = SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]);
        let xu = SVector::<f64, 2>::from_row_slice(&[10.0, 10.0]);
        let x = SVector::<f64, 2>::from_row_slice(&[5.0, 5.0]);
        let p = SVector::<f64, 2>::from_row_slice(&[1.0, 1.0]);
        let bend = false;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -5.0);
        assert_eq!(amax, 5.0);
        assert_eq!(scale, 5.0);
    }

    #[test]
    #[should_panic]
    fn test_3() {
        let xl = SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]);
        let xu = SVector::<f64, 2>::from_row_slice(&[10.0, 10.0]);
        let x = SVector::<f64, 2>::from_row_slice(&[5.0, 5.0]);
        let p = SVector::<f64, 2>::zeros(); // zero search direction
        let bend = false;

        let _ = lsrange(&x, &p, &xl, &xu, bend);
    }

    #[test]
    fn test_5() {
        let xl = SVector::<f64, 2>::from_row_slice(&[0.0, 2.0]);
        let xu = SVector::<f64, 2>::from_row_slice(&[10.0, 20.0]);
        let x = SVector::<f64, 2>::from_row_slice(&[-5.0, 0.5]);
        let p = SVector::<f64, 2>::from_row_slice(&[-10.0, 1.0]);
        let bend = true;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -1.5);
        assert_eq!(amax, 19.5);
        assert_eq!(scale, 0.5);
    }
}