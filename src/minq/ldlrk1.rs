use nalgebra::{DMatrix, DVector};

pub fn ldlrk1(
    L: &mut Vec<Vec<f64>>,
    d: &mut Vec<f64>,
    mut alp: f64,
    u: &mut Vec<f64>,
) ->
    Vec<f64> //p
{
    let mut p: Vec<f64> = vec![];
    if alp == 0.0 {
        return p;
    }

    let eps: f64 = 2.2204e-16;
    let n: usize = u.len();
    let neps: f64 = n as f64 * eps;

    // Collect indices for which u[i] != 0 to avoid borrowing issues
    let indices: Vec<usize> = (0..n).filter(|&i| u[i] != 0.0).collect();

    // Update process
    for &k in &indices {
        let delta: f64 = d[k] + alp * u[k].powi(2);
        if alp < 0.0 && delta <= neps {
            // Update not definite
            p = vec![0.0; n];
            p[k] = 1.0;

            let flattened_L0K: Vec<f64> = {
                let mut vec = Vec::with_capacity((k + 1) * (k + 1));
                for i in 0..=k {
                    vec.extend_from_slice(&L[i][..=k]); // Extend the vector with a slice of the current row
                }
                vec
            };

            let p0K: Vec<f64> = p.iter().take(k + 1).cloned().collect();

            let matrix = DMatrix::from_vec(k + 1, k + 1, flattened_L0K);
            let vector = DVector::from_vec(p0K);
            let result_vector = matrix.lu().solve(&vector).unwrap();
            let p0K: Vec<f64> = result_vector.data.into();


            for i in 0..n {
                if i <= k {
                    p[i] = p0K[i];
                }
            }
            return p;
        }

        let q = d[k] / delta;
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
    p
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real() {
        let mut L = vec![
            vec![1.0, 0.0, -1.45],
            vec![-0.6, -1.0, 4.0],
            vec![-1.6, 1.0, 0.01]
        ];
        let mut d = vec![0.01, 0.002, 0.03];
        let alp = -10.0;
        let mut u = vec![1.23, -10.0, 5.0];

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

        assert_eq!(L, vec![
            vec![1.0, 0.0, -1.45],
            vec![-0.6, -1.0, 4.0],
            vec![-1.6, 1.0, 0.01]
        ]);

        assert_eq!(d, vec![0.01, 0.002, 0.03]);
        assert_eq!(p, vec![1.0, 0.0, 0.0]);

        assert_eq!(alp, -10.0);
    }


    #[test]
    fn test_0() {
        let mut L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];
        let mut d = vec![2.0, 3.0];
        let alp = 1.0;
        let mut u = vec![1.0, 1.0];

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

        assert_eq!(L, vec![
            vec![1.0, 0.0],
            vec![0.6666666666666666, 1.0],
        ]);

        assert_eq!(d, vec![3.0, 3.1666666666666666]);
        assert_eq!(p, vec![]);
        assert_eq!(alp, 1.0);
    }

    #[test]
    fn test_1() {
        let mut L = vec![vec![1.0]];
        let mut d = vec![2.0];
        let alp = 0.0;
        let mut u = vec![1.0];

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

        assert_eq!(L, vec![vec![1.0]]);
        assert_eq!(d, vec![2.0]);
        assert_eq!(p, vec![]);
    }

    #[test]
    fn test_2() {
        let mut L = vec![
            vec![1.0, 0.0],
            vec![0.5, 1.0],
        ];
        let mut d = vec![1.0, 1.0];
        let alp = -0.5;
        let mut u = vec![1.0, 1.0];

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

        assert_eq!(L, vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]);

        assert_eq!(d, vec![0.5, 0.75]);
        assert_eq!(p, vec![]);
    }

    #[test]
    fn test_3() {
        let mut L = vec![vec![1.0]];
        let mut d = vec![2.0];
        let alp = 1.0;
        let mut u = vec![2.0];

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);


        assert_eq!(L, vec![vec![1.0]]);
        assert_eq!(d, vec![6.0]);
        assert_eq!(p, vec![]);
    }

    #[test]
    fn test_4() {
        let mut L = vec![vec![1.0]];
        let mut d = vec![2.0];
        let alp = -1.0;
        let mut u = vec![2.0];

        let p = ldlrk1(&mut L, &mut d, alp, &mut u);

        assert_eq!(L, vec![vec![1.0]]);
        assert_eq!(d, vec![2.0]);
        assert_eq!(p, vec![1.0]);
    }
}