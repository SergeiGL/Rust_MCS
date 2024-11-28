pub fn hessian(
    i: usize,
    k: usize,
    x: &Vec<f64>,
    x0: &Vec<f64>,
    f: f64,
    f0: f64,
    g: &Vec<f64>,
    G: &Vec<Vec<f64>>,
) -> f64 {
    let delta_xi = x[i] - x0[i];
    let delta_xk = x[k] - x0[k];

    let mut h = f - f0 - g[i] * delta_xi - g[k] * delta_xk;
    h -= 0.5 * G[i][i] * delta_xi * delta_xi;
    h -= 0.5 * G[k][k] * delta_xk * delta_xk;

    h / (delta_xi * delta_xk)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let i = 0;
        let k = 0;
        let x = vec![1.0f64, 1.0e-10];
        let x0 = vec![0.0f64, 1.0e-10];
        let f = 0.0f64;
        let f0 = 0.0f64;
        let g = vec![0.0f64, 0.0f64];
        let G = vec![vec![0.0f64, 0.0f64], vec![0.0f64, 0.0f64]];

        assert_eq!(hessian(i, k, &x, &x0, f, f0, &g, &G), 0.0);
    }

    #[test]
    fn test_1() {
        let i = 0;
        let k = 1;
        let x = vec![-0.5f64, -0.75];
        let x0 = vec![-1.0f64, -1.0];
        let f = -2.0f64;
        let f0 = -3.0f64;
        let g = vec![-1.0f64, -1.5];
        let G = vec![vec![1.0f64, 0.5], vec![0.5f64, 2.0]];

        assert_eq!(hessian(i, k, &x, &x0, f, f0, &g, &G), 13.5);
    }

    #[test]
    fn test_2() {
        let i = 0;
        let k = 1;
        let x = vec![0.5f64, 0.75];
        let x0 = vec![1.0f64, 1.0];
        let f = 12.0f64;
        let f0 = 64.0f64;
        let g = vec![-1.0f64, 1.5];
        let G = vec![vec![15.0f64, -0.5], vec![1.5f64, 2.0]];

        assert_eq!(hessian(i, k, &x, &x0, f, f0, &g, &G), -432.5);
    }
}
