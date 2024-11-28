use crate::feval::feval;
use crate::hessian::hessian;
use crate::polint::polint1;


pub fn triple(
    x: &mut Vec<f64>,
    mut f: f64,
    x1: &mut Vec<f64>,
    x2: &mut Vec<f64>,
    u: &[f64],
    v: &[f64],
    hess: &Vec<Vec<f64>>,
    G: &mut Vec<Vec<f64>>,
    setG: bool,
) -> (
    Vec<f64>, //xtrip
    f64, //ftrip
    Vec<f64>, //g
    usize //nf
) {
    let mut nf: usize = 0;
    let n = x.len();
    let mut g: Vec<f64> = vec![0.0; n];
    let mut nargin_lower_10 = false;

    if setG {
        nargin_lower_10 = true;
        *G = vec![vec![0.0; n]; n];
    }

    // Determine indices where u[i] < x[i] < v[i]
    let ind = x
        .iter()
        .enumerate()
        .filter(|&(i, xi)| (u[i] < *xi) && (*xi < v[i]))
        .map(|(i, _)| i)
        .collect::<Vec<usize>>();

    let ind1 = x
        .iter()
        .enumerate()
        .filter(|&(i, xi)| (*xi <= u[i]) || (*xi >= v[i]))
        .map(|(i, _)| i)
        .collect::<Vec<usize>>();

    for &ind1_j in &ind1 {
        g[ind1_j] = 0.0;
        for k in 0..n {
            G[ind1_j][k] = 0.0;
            G[k][ind1_j] = 0.0;
        }
    }

    let mut xtrip = x.clone();
    let mut ftrip = f;
    let mut xtripnew = x.clone();
    let mut ftripnew = f;

    if ind.len() <= 1 {
        if !ind.is_empty() {
            for &i in &ind {
                g[i] = 1.0;
                G[i][i] = 1.0;
            }
        }
        return (xtrip, ftrip, g, nf);
    }

    if setG {
        *G = vec![vec![0.0; n]; n];
    }

    let mut k1: Option<usize> = None;

    for (_, &i) in ind.iter().enumerate() {
        *x = xtrip.clone();
        f = ftrip;

        x[i] = x1[i];
        let f1 = feval(&x);

        x[i] = x2[i];
        let f2 = feval(&x);
        nf += 2;

        (g[i], G[i][i]) = polint1(
            &[xtrip[i], x1[i], x2[i]],
            &[f, f1, f2]);

        if f1 <= f2 {
            if f1 < ftrip {
                ftripnew = f1;
                xtripnew[i] = x1[i];
            }
        } else {
            if f2 < ftrip {
                ftripnew = f2;
                xtripnew[i] = x2[i];
            }
        }

        if nargin_lower_10 {
            if f1 <= f2 {
                x[i] = x1[i];
            } else {
                x[i] = x2[i];
            }

            for k in 0..i {
                if hess[i][k] != 0.0 {
                    if xtrip[k] > u[k] && xtrip[k] < v[k] && ind.contains(&k) {
                        let q1 = ftrip
                            + g[k] * (x1[k] - xtrip[k])
                            + 0.5 * G[k][k] * (x1[k] - xtrip[k]).powi(2);
                        let q2 = ftrip
                            + g[k] * (x2[k] - xtrip[k])
                            + 0.5 * G[k][k] * (x2[k] - xtrip[k]).powi(2);


                        if q1 <= q2 {
                            x[k] = x1[k];
                        } else {
                            x[k] = x2[k];
                        }

                        let f12 = feval(&x);
                        nf += 1;

                        G[i][k] = hessian(i, k, &x, &xtrip, f12, ftrip, &g, G);
                        G[k][i] = G[i][k];
                        if f12 < ftripnew {
                            ftripnew = f12;
                            xtripnew = x.clone();
                            k1 = Some(k);
                        }
                        x[k] = xtrip[k];
                    }
                } else {
                    G[i][k] = 0.0;
                    G[k][i] = 0.0;
                }
            }
        }

        if ftripnew < ftrip {
            if x1[i] == xtripnew[i] {
                x1[i] = xtrip[i];
            } else {
                x2[i] = xtrip[i];
            }

            if let (true, Some(k1_val)) = (nargin_lower_10, k1) {
                if xtripnew[k1_val] == x1[k1_val] {
                    x1[k1_val] = xtrip[k1_val];
                } else {
                    x2[k1_val] = xtrip[k1_val];
                }
            }

            for k in 0..=i {
                if ind.contains(&k) {
                    g[k] += G[i][k] * (xtripnew[i] - xtrip[i]);

                    if let (true, Some(k1_val)) = (nargin_lower_10, k1) {
                        g[k] += G[k1_val][k] * (xtripnew[k1_val] - xtrip[k1_val]);
                    }
                }
            }
            xtrip = xtripnew.clone();
            ftrip = ftripnew;
        }
    }

    (xtrip, ftrip, g, nf)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;


    #[test]
    fn test_cover_0() {
        let mut x = vec![1.2, -1.15, 0.01, 2.27, -1.31, 0.3];
        let f = 12.32;
        let mut x1 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut x2 = vec![0.2, 0.1, 0.4, 0.2, 0.3, 1.2];
        let u = vec![-10., -10., -10., -10., -10., -10.];
        let v = vec![3., 3., 3., 3., 3., 3.];
        let hess = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; 6];
        let mut G = vec![vec![-1.0; 6]; 6];
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&mut x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = vec![0.1, 0.2, 0.4, 0.4, 0.3, 0.6];
        let expected_ftrip = -2.630060151483446;
        let expected_g = vec![-2.8999999999999755, -0.4300000000000163, -0.0400000000443656, -1.9205128274057135, 1.2540564661233267, -2.3627224189529548];

        let expected_G = vec![
            vec![2.2399999999999984e+01, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00],
            vec![-1.0000000000000000e+00, 4.1641022738722085e-15, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00],
            vec![-1.0000000000000000e+00, -1.0000000000000000e+00, -1.5576891494122202e-10, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00],
            vec![-1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, 1.3360993045537445e-02, -1.0000000000000000e+00, -1.0000000000000000e+00],
            vec![-1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, 2.6824336175403096e+00, -1.0000000000000000e+00],
            vec![-1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, 2.0955649004349532e+01]
        ];
        let expected_x1 = vec![1.2, -1.15, 0.3, 2.27, 0.5, 0.3];
        let expected_x2 = vec![0.2, 0.1, 0.01, 0.2, -1.31, 1.2];
        let expected_nf = 12;

        assert_eq!(xtrip, expected_xtrip);
        assert_relative_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1, expected_x1);
        assert_eq!(x2, expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_cover_1() {
        let mut x = vec![-0.2, 0.15, 1.023, 0.0, 3.0, -10.];
        let f = -0.4;
        let mut x1 = vec![10.0; 6];
        let mut x2 = vec![-10.0; 6];
        let u = vec![-5.0; 6];
        let v = vec![6.0; 6];
        let hess = vec![vec![-1.0, 2.0, -3.0, -4.0, -5.0, 0.0]; 6];
        let mut G = vec![vec![2.5; 6]; 6];
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&mut x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = vec![-0.2, 0.15, 1.023, 0.0, 3.0, -10.0];
        let expected_ftrip = -0.4;
        let expected_g = vec![-0.0016006402561024313, 0.0012002700607636752, 0.00827055374338511, 0., 0.026373626373626377, 0.];

        let expected_G = vec![
            vec![0.008003201280512205, 2.5, 2.5, 2.5, 2.5, 0.],
            vec![2.5, 0.008001800405091146, 2.5, 2.5, 2.5, 0., ],
            vec![2.5, 2.5, 0.008084607764794829, 2.5, 2.5, 0.],
            vec![2.5, 2.5, 2.5, 0.008, 2.5, 0.],
            vec![2.5, 2.5, 2.5, 2.5, 0.008791208791208791, 0.],
            vec![0., 0., 0., 0., 0., 0.]
        ];
        let expected_x1 = vec![10.0; 6];
        let expected_x2 = vec![-10.0; 6];
        let expected_nf = 10;

        assert_eq!(xtrip, expected_xtrip);
        assert_relative_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1, expected_x1);
        assert_eq!(x2, expected_x2);
        assert_eq!(nf, expected_nf);
    }


    #[test]
    fn test_0() {
        let mut x = vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.66];
        let f = -3.32;
        let mut x1 = vec![0.201, 0.148, 0.475, 0.273, 0.307, 0.656];
        let mut x2 = vec![0.202, 0.146, 0.476, 0.271, 0.308, 0.654];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let hess = vec![vec![1.0; 6]; 6];
        let mut G = vec![vec![10.0; 6]; 6];
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&mut x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = vec![0.202, 0.15, 0.476, 0.273, 0.31, 0.656];
        let expected_ftrip = -3.3220115127770096;
        let expected_g = vec![0.3562133567590662, 0.05335723233435563, -0.03543761332326843, -0.1858482395075104, -0.20143664923136856, -0.06765510147636408];

        let expected_G = vec![
            vec![641.7724055589591, 10.0, 10.0, 10.0, 10.0, 10.0],
            vec![10.0, 22.690208792974747, 10.0, 10.0, 10.0, 10.0],
            vec![10.0, 10.0, 18.865775185676124, 10.0, 10.0, 10.0],
            vec![10.0, 10.0, 10.0, 59.241589659938036, 10.0, 10.0],
            vec![10.0, 10.0, 10.0, 10.0, 98.88912715464784, 10.0],
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 51.83419750517014]
        ];
        let expected_x1 = vec![0.201, 0.148, 0.475, 0.27, 0.307, 0.66];
        let expected_x2 = vec![0.2, 0.146, 0.47, 0.271, 0.308, 0.654];
        let expected_nf = 12;

        assert_eq!(xtrip, expected_xtrip);
        assert_relative_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1, expected_x1);
        assert_eq!(x2, expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_1() {
        let mut x = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let f = -3.32;
        let mut x1 = vec![0.001, 0.999, 0.001, 0.999, 0.001, 0.999];
        let mut x2 = vec![0.002, 0.998, 0.002, 0.998, 0.002, 0.998];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let hess = vec![vec![1.0; 6]; 6];
        let mut G = vec![vec![0.0; 6]; 6];
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&mut x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let expected_ftrip = -3.32;
        let expected_g = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let expected_G = vec![vec![0.0; 6]; 6];
        let expected_x1 = vec![0.001, 0.999, 0.001, 0.999, 0.001, 0.999];
        let expected_x2 = vec![0.002, 0.998, 0.002, 0.998, 0.002, 0.998];
        let expected_nf = 0;

        assert_eq!(xtrip, expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1, expected_x1);
        assert_eq!(x2, expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_2() {
        let mut x = vec![-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let f = -3.32;
        let mut x1 = vec![-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let mut x2 = vec![-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let hess = vec![vec![1.0; 6]; 6];
        let mut G = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ];
        let setG = true;

        let (xtrip, ftrip, g, nf) = triple(&mut x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = vec![-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let expected_ftrip = -3.32;
        let expected_g = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let expected_G = vec![vec![0.0; 6]; 6];
        let expected_x1 = vec![-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let expected_x2 = vec![-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let expected_nf = 0;

        assert_eq!(xtrip, expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1, expected_x1);
        assert_eq!(x2, expected_x2);
        assert_eq!(nf, expected_nf);
    }
}