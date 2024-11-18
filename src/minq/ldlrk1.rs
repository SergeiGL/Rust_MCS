pub fn ldlrk1(
    mut L: Vec<Vec<f64>>,
    mut d: Vec<f64>,
    mut alp: f64,
    mut u: Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut p: Vec<f64> = vec![];
    if alp == 0.0 {
        return (L, d, p);
    }

    let eps: f64 = 2.2204e-16;
    let n: usize = u.len();
    let neps: f64 = n as f64 * eps;

    // Save old factorization
    let L0 = L.clone();
    let d0 = d.clone();

    // Collect indices for which u[i] != 0 to avoid borrowing issues
    let indices: Vec<usize> = (0..n).filter(|&i| u[i] != 0.0).collect();

    // Update process
    for &k in &indices {
        let delta: f64 = d[k] + alp * u[k].powf(2.0);
        if alp < 0.0 && delta <= neps {
            // Update not definite
            p = vec![0.0; n];
            p[k] = 1.0;

            let p0K_range: Vec<usize> = (0..=k).collect(); // Range for p0K
            let mut p0K: Vec<f64> = p0K_range.iter().map(|&i| p[i]).collect();
            let L0K: Vec<Vec<f64>> = p0K_range
                .iter()
                .map(|&i| {
                    p0K_range
                        .iter()
                        .map(|&j| L[i][j])
                        .collect::<Vec<f64>>()
                })
                .collect();

            // Solve L0K * p0K = p0K using forward substitution
            p0K = solve_lower_triangular(&L0K, &p0K);

            for i in 0..n {
                if p0K_range.contains(&i) {
                    p[i] = p0K[i];
                }
            }

            // Restore original factorization
            L = L0;
            d = d0;
            return (L, d, p);
        }

        let q: f64 = d[k] / delta;
        d[k] = delta;

        let ind: Vec<usize> = ((k + 1)..n).collect();
        let LindK: Vec<f64> = ind.iter().map(|&i| L[i][k]).collect();
        let uk: f64 = u[k];

        let c: Vec<f64> = LindK
            .iter()
            .map(|&lk| lk * uk)
            .collect();

        for (i, &index) in ind.iter().enumerate() {
            L[index][k] = LindK[i] * q + (alp * u[k] / delta) * u[index];
        }

        for (i, &index) in ind.iter().enumerate() {
            u[index] -= c[i];
        }

        alp *= q;
        if alp == 0.0 {
            break;
        }
    }

    return (L, d, p);
}

// Helper function for solving lower-triangular systems using forward substitution
fn solve_lower_triangular(L: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += L[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / L[i][i];
    }

    x
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;

    #[test]
    fn test_2x2_positive_alp() {
        let L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];
        let d = vec![2.0, 3.0];
        let alp = 1.0;
        let u = vec![1.0, 1.0];

        let (L_new, d_new, p) = ldlrk1(L, d, alp, u);

        assert!(relative_eq!(L_new[0][0], 1.0, epsilon = 1e-6));
        assert!(relative_eq!(L_new[0][1], 0.0, epsilon = 1e-6));
        assert!(relative_eq!(L_new[1][0], 0.66666667, epsilon = 1e-6));
        assert!(relative_eq!(L_new[1][1], 1.0, epsilon = 1e-6));

        assert!(relative_eq!(d_new[0], 3.0, epsilon = 1e-6));
        assert!(relative_eq!(d_new[1], 3.16666667, epsilon = 1e-6));
        assert!(p.is_empty()); // Assert that p is empty or []
    }

    #[test]
    fn test_zero_alp() {
        let L = vec![
            vec![1.0],
        ];
        let d = vec![2.0];
        let alp = 0.0;
        let u = vec![1.0];

        let (L_new, d_new, p) = ldlrk1(L, d, alp, u);


        assert!(relative_eq!(L_new[0][0], 1.0, epsilon = 1e-6));
        assert!(relative_eq!(d_new[0], 2.0, epsilon = 1e-6));

        // Check p
        assert!(p.is_empty()); // p should be empty
    }

    #[test]
    fn test_negative_alp_update_not_definite() {
        let L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];
        let d = vec![1.0, 1.0];
        let alp = -0.5;
        let u = vec![1.0, 1.0];

        let (L_new, d_new, p) = ldlrk1(L, d, alp, u);

        assert!(relative_eq!(L_new[0][0], 1.0, epsilon = 1e-6));
        assert!(relative_eq!(L_new[0][1], 0.0, epsilon = 1e-6));
        assert!(relative_eq!(L_new[1][0], 0.0, epsilon = 1e-6));
        assert!(relative_eq!(L_new[1][1], 1.0, epsilon = 1e-6));

        assert!(relative_eq!(d_new[0], 0.5, epsilon = 1e-6));
        assert!(relative_eq!(d_new[1], 0.75, epsilon = 1e-6));

        assert!(p.is_empty());
    }

    #[test]
    fn test_single_dimension_positive_alp() {
        let L = vec![
            vec![1.0],
        ];
        let d = vec![2.0];
        let alp = 1.0;
        let u = vec![2.0];

        let (L_new, d_new, p) = ldlrk1(L, d, alp, u);


        assert!(relative_eq!(L_new[0][0], 1.0, epsilon = 1e-6));
        assert!(relative_eq!(d_new[0], 6.0, epsilon = 1e-6));
        assert!(p.is_empty());
    }

    #[test]
    fn test_single_dimension_negative_alp() {
        let L = vec![
            vec![1.0],
        ];
        let d = vec![2.0];
        let alp = -1.0;
        let u = vec![2.0];

        let (L_new, d_new, p) = ldlrk1(L, d, alp, u);

        assert!(relative_eq!(L_new[0][0], 1.0, epsilon = 1e-6));
        assert!(relative_eq!(d_new[0], 2.0, epsilon = 1e-6));
        assert_eq!(p.len(), 1);
        assert!(relative_eq!(p[0], 1.0, epsilon = 1e-6));
    }
}