use nalgebra::{SMatrix, SVector};

pub fn hessian<const N: usize>(
    i: usize,
    k: usize,
    x: &[f64; N],
    x0: &[f64; N],
    f: f64,
    f0: f64,
    g: &SVector<f64, N>,
    G: &SMatrix<f64, N, N>,
) -> f64 {
    let h = f - f0 - g[i] * (x[i] - x0[i]) - g[k] * (x[k] - x0[k]) - 0.5 * G[(i, i)] * (x[i] - x0[i]).powi(2) - 0.5 * G[(k, k)] * (x[k] - x0[k]).powi(2);
    h / (x[i] - x0[i]) / (x[k] - x0[k])
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let i = 0;
        let k = 0;
        let x = [1.0f64, 1.0e-10];
        let x0 = [0.0f64, 1.0e-10];
        let f = 0.0f64;
        let f0 = 0.0f64;
        let g = SVector::<f64, 2>::from_row_slice(&[0.0f64, 0.0f64]);
        let G = SMatrix::<f64, 2, 2>::from_row_slice(&[0.0f64, 0.0f64, 0.0f64, 0.0f64]);

        assert_eq!(hessian(i, k, &x, &x0, f, f0, &g, &G), 0.0);
    }

    #[test]
    fn test_1() {
        let i = 0;
        let k = 1;
        let x = [-0.5f64, -0.75];
        let x0 = [-1.0f64, -1.0];
        let f = -2.0f64;
        let f0 = -3.0f64;
        let g = SVector::<f64, 2>::from_row_slice(&[-1.0f64, -1.5]);
        let G = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0f64, 0.5, 0.5f64, 2.0]);

        assert_eq!(hessian(i, k, &x, &x0, f, f0, &g, &G), 13.5);
    }

    #[test]
    fn test_2() {
        let i = 0;
        let k = 1;
        let x = [0.5f64, 0.75];
        let x0 = [1.0f64, 1.0];
        let f = 12.0f64;
        let f0 = 64.0f64;
        let g = SVector::<f64, 2>::from_row_slice(&[-1.0f64, 1.5]);
        let G = SMatrix::<f64, 2, 2>::from_row_slice(&[15.0f64, -0.5, 1.5f64, 2.0]);

        assert_eq!(hessian(i, k, &x, &x0, f, f0, &g, &G), -432.5);
    }

    #[test]
    fn test_3() {
        let i = 4;
        let k = 1;
        let x = [0.112, 11.0, -0.32, 0.0, -1.0, -0.4];
        let x0 = [0.112, -0.1, -0.32, 0.0, 10.0, -0.4];
        let f = -0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004289691165770153;
        let f0 = -4.5;
        let g = SVector::<f64, 6>::from_row_slice(&[-3.6334635451080715, -4.594594594594595, -6.220120557002464, -4.090909090909091, 4.090909096062172, 0.]);
        let G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.7433436057913404, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.9009009009009009, 0.5961844197086474, 0.40540540540540543, 0.0, 0.0,
            0.0, 0.5961844197086474, 1.1691955934221243, -6.617647058765987, 0.0, 0.0,
            0.0, 0.40540540540540543, -6.617647058765987, 0.8181818181818182, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.8181818078647831, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);

        assert_eq!(hessian(i, k, &x, &x0, f, f0, &g, &G), 0.03685503127875094);
    }
}
