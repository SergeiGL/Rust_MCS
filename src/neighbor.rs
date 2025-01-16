use nalgebra::SVector;

pub fn neighbor<const N: usize>(
    x: &SVector<f64, N>,
    delta: &SVector<f64, N>,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
) -> (SVector<f64, N>, SVector<f64, N>) {
    // Initialize x1 and x2 as zero vectors
    let mut x1 = SVector::<f64, N>::zeros();
    let mut x2 = SVector::<f64, N>::zeros();

    // Iterate over each component once
    for i in 0..N {
        let xi = x[i];
        let delta_i = delta[i];
        let u_i = u[i];
        let v_i = v[i];

        // Compute x1[i]
        x1[i] = if xi == u_i {
            xi + 2.0 * delta_i
        } else {
            u_i.max(xi - delta_i)
        };

        // Compute x2[i]
        x2[i] = if xi == v_i {
            xi - 2.0 * delta_i
        } else {
            v_i.min(xi + delta_i)
        };
    }

    (x1, x2)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let delta = SVector::<f64, 6>::from_row_slice(&[0.001, 0.001, 0.002, 0.001, 0.001, 0.002]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let (x1, x2) = neighbor(&x, &delta, &u, &v);
        assert_eq!(x1.as_slice(), [0.199, 0.149, 0.46799999999999997, 0.269, 0.309, 0.648]);
        assert_eq!(x2.as_slice(), [0.201, 0.151, 0.472, 0.271, 0.311, 0.652])
    }

    #[test]
    fn test_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 0.1, 0.0, 0.2, 0.0, 0.6]);
        let delta = SVector::<f64, 6>::from_row_slice(&[0.001, 0.001, 0.002, 0.001, 0.001, 0.002]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let (x1, x2) = neighbor(&x, &delta, &u, &v);

        assert_eq!(x1.as_slice(), [0.002, 0.099, 0.004, 0.199, 0.002, 0.598]);
        assert_eq!(x2.as_slice(), [0.001, 0.101, 0.002, 0.201, 0.001, 0.602])
    }

    #[test]
    fn test_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 0.15, 0.55, 1.0, 0.3, 1.0]);
        let delta = SVector::<f64, 6>::from_row_slice(&[0.001, 0.001, 0.002, 0.001, 0.001, 0.002]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let (x1, x2) = neighbor(&x, &delta, &u, &v);

        assert_eq!(x1.as_slice(), [0.999, 0.149, 0.548, 0.999, 0.299, 0.998]);
        assert_eq!(x2.as_slice(), [0.998, 0.151, 0.552, 0.998, 0.301, 0.996]);
    }

    #[test]
    fn test_3() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]);
        let delta = SVector::<f64, 6>::from_row_slice(&[0.001, 0.001, 0.002, 0.001, 0.001, 0.002]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let (x1, x2) = neighbor(&x, &delta, &u, &v);

        assert_eq!(x1.as_slice(), [0.002, 0.002, 0.498, 0.002, 0.002, 0.004]);
        assert_eq!(x2.as_slice(), [-0.002, -0.002, 0.502, -0.002, -0.002, -0.004]);
    }

    #[test]
    fn test_4() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0001, 0.9999, 0.5001, 0.2001, 0.2999, 0.6501]);
        let delta = SVector::<f64, 6>::from_row_slice(&[0.001, 0.001, 0.002, 0.001, 0.001, 0.002]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let (x1, x2) = neighbor(&x, &delta, &u, &v);

        assert_eq!(x1.as_slice(), [0.0, 0.9989, 0.4981, 0.1991, 0.2989, 0.6481]);
        assert_eq!(x2.as_slice(), [0.0011, 1.0, 0.5021, 0.2011, 0.3009, 0.6521]);
    }
}
