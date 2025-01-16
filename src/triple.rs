use crate::feval::feval;
use crate::hessian::hessian;
use crate::polint::polint1;
use nalgebra::{SMatrix, SVector};

pub fn triple<const N: usize>(
    x: &SVector<f64, N>,
    mut f: f64,
    x1: &mut SVector<f64, N>,
    x2: &mut SVector<f64, N>,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    hess: &SMatrix<f64, N, N>,
    G: &mut SMatrix<f64, N, N>,
    setG: bool,
) -> (
    SVector<f64, N>,   // xtrip
    f64,               // ftrip
    SVector<f64, N>,   // g
    usize              // nf
) {
    let (mut nf, mut nargin_lower_10) = (0_usize, false);
    let mut g = SVector::<f64, N>::zeros();

    if setG {
        nargin_lower_10 = true;
        *G = SMatrix::<f64, N, N>::zeros();
    }

    let mut ind: Vec<usize> = Vec::with_capacity(x.len());
    for (i, &xi) in x.iter().enumerate() {
        if (u[i] < xi) && (xi < v[i]) {
            ind.push(i);
        } else {
            g[i] = 0.0;
            G.fill_row(i, 0.0);
            G.fill_column(i, 0.0)
        }
    }

    let mut xtrip = x.clone();
    let mut ftrip = f;
    let mut xtripnew = x.clone();
    let mut ftripnew = f;

    if ind.len() <= 1 {
        for i in ind {
            g[i] = 1.0;
            G[(i, i)] = 1.0;
        }
        return (xtrip, ftrip, g, nf);
    }

    if setG {
        *G = SMatrix::<f64, N, N>::zeros();
    }

    let mut k1_option: Option<usize> = None;

    for &i in ind.iter() {
        let mut x = xtrip.clone();
        f = ftrip;

        x[i] = x1[i];
        let f1 = feval(&x);

        x[i] = x2[i];
        let f2 = feval(&x);
        nf += 2;

        (g[i], G[(i, i)]) = polint1(
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
            k1_option = None;
            if f1 <= f2 {
                x[i] = x1[i];
            } else {
                x[i] = x2[i];
            }

            for k in 0..i {
                if hess[(i, k)] != 0.0 {
                    if xtrip[k] > u[k] && xtrip[k] < v[k] && ind.contains(&k) {
                        let q1 = ftrip
                            + g[k] * (x1[k] - xtrip[k])
                            + 0.5 * G[(k, k)] * (x1[k] - xtrip[k]).powi(2);
                        let q2 = ftrip
                            + g[k] * (x2[k] - xtrip[k])
                            + 0.5 * G[(k, k)] * (x2[k] - xtrip[k]).powi(2);


                        if q1 <= q2 {
                            x[k] = x1[k];
                        } else {
                            x[k] = x2[k];
                        }

                        let f12 = feval(&x);
                        nf += 1;

                        // println!("{}\n{i} {k}, {x:?}, {xtrip:?}, {f12}, {ftrip}, {g}, {G:?}", G[(1, 4)]);
                        G[(i, k)] = hessian(i, k, &x, &xtrip, f12, ftrip, &g, G);
                        G[(k, i)] = G[(i, k)];
                        // println!("{}\n", G[(i, k)]);
                        if f12 < ftripnew {
                            ftripnew = f12;
                            xtripnew = x.clone();
                            k1_option = Some(k);
                        }
                        x[k] = xtrip[k];
                    }
                } else {
                    G[(i, k)] = 0.0;
                    G[(k, i)] = 0.0;
                }
            }
        }
        // println!("{g:?}, {G:.15}");
        if ftripnew < ftrip {
            if x1[i] == xtripnew[i] {
                x1[i] = xtrip[i];
            } else {
                x2[i] = xtrip[i];
            }

            if let (true, Some(k1_val)) = (nargin_lower_10, k1_option) {
                if xtripnew[k1_val] == x1[k1_val] {
                    x1[k1_val] = xtrip[k1_val];
                } else {
                    x2[k1_val] = xtrip[k1_val];
                }
            }

            for k in 0..=i {
                if ind.contains(&k) {
                    g[k] += G[(i, k)] * (xtripnew[i] - xtrip[i]);

                    if let (true, Some(k1_val)) = (nargin_lower_10, k1_option) {
                        g[k] += G[(k1_val, k)] * (xtripnew[k1_val] - xtrip[k1_val]);
                    }
                }
            }
            xtrip = xtripnew.clone();
            ftrip = ftripnew;
        }
        // println!("end {g:?}");
    }

    (xtrip, ftrip, g, nf)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cover() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.43365618894869346, 0.8420868988145562, 0.7176872417127306, 0.4916610939334169, 0.18103825707549334, 0.028904638786252483]);
        let f = -2.9081562404231485;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[0.4336535629633933, 0.8420817995956955, 0.7176828957903273, 0.49165811670205656, 0.18103716080657348, 0.02890446375552885]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[0.4336588149339936, 0.8420919980334169, 0.7176915876351339, 0.4916640711647772, 0.1810393533444132, 0.028904813816976117]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.; 6]);
        let hess = SMatrix::<f64, 6, 6>::repeat(0.);
        let mut G = SMatrix::<f64, 6, 6>::repeat(5.);
        let setG = true;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.4336535629633933, 0.8420919980334169, 0.7176915876351339, 0.4916640711647772, 0.18103716080657348, 0.028904813816976117];
        let expected_ftrip = -2.908187629699183;
        let expected_g = SVector::<f64, 6>::from_row_slice(&
            [2.870509283439505, -1.8456380962311199, -0.028925098717332225, -4.756025709825903, 0.006436256029900503, -0.8361689370862301]
        );
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            95.77846342296617, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.01426462103387, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30851282407215563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.08631581627649, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.705041024447086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 80.21965674487866
        ]);
        let expected_x1 = [0.43365618894869346, 0.8420817995956955, 0.7176828957903273, 0.49165811670205656, 0.18103825707549334, 0.02890446375552885];
        let expected_x2 = [0.4336588149339936, 0.8420868988145562, 0.7176872417127306, 0.4916610939334169, 0.1810393533444132, 0.028904638786252483];
        let expected_nf = 12;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_real_mistake_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.4336561889486934, 0.8420868988145562, 0.7176872417127305, 0.4916610939334169, 0.18103825707549337, 0.0289046387862526]);
        let f = -2.9081562404231485;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[0.43365356296339325, 0.8420817995956955, 0.7176828957903272, 0.49165811670205656, 0.1810371608065735, 0.028904463755528968]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[0.43365881493399355, 0.8420919980334169, 0.7176915876351337, 0.4916640711647772, 0.18103935334441323, 0.028904813816976235]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.; 6]);
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);
        let mut G = SMatrix::<f64, 6, 6>::repeat(154765865.);
        let setG = true;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.43365356296339325, 0.8420919980334169, 0.7176915876351337, 0.4916640711647772, 0.1810371608065735, 0.028904813816976235];
        let expected_ftrip = -2.908187629699183;
        let expected_g = SVector::<f64, 6>::from_row_slice(&
            [2.870532358366225, -1.8456485678912016, -0.0289255380117641, -4.756026093992137, 0.006436196278905773, -0.8361689332804193]
        );
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            9.5778399023074854e+01, 1.7848172824950423e+00, -6.4984897069037626e-03, 4.6554535346548853e+00, 1.3883639904063093e-02, 8.9469421366481816e-01, 1.7848172824950423e+00, 4.5014298779076967e+01, -1.0272193762911967e-01, -3.2914230575725387e+00, 1.9701584144597964e-01, -5.7717866690791111e-02, -6.4984897069037626e-03, -1.0272193762911967e-01, 3.0851282407215563e-01, -1.3564150640439418e-01, 5.5554265772868178e-02, 1.4536950519987182e-01, 4.6554535346548853e+00, -3.2914230575725387e+00, -1.3564150640439418e-01, 5.0086265715510400e+01, 2.1048969995198918e-01, -8.7606526656835382e-01, 1.3883639904063093e-02, 1.9701584144597964e-01, 5.5554265772868178e-02, 2.1048969995198918e-01, 7.0467150609108908e-01, -3.4484592595973723e-01, 8.9469421366481816e-01, -5.7717866690791111e-02, 1.4536950519987182e-01, -8.7606526656835382e-01, -3.4484592595973723e-01, 8.0234152526930103e+01]);
        let expected_x1 = [0.4336561889486934, 0.8420817995956955, 0.7176828957903272, 0.49165811670205656, 0.18103825707549337, 0.028904463755528968];
        let expected_x2 = [0.43365881493399355, 0.8420868988145562, 0.7176872417127305, 0.4916610939334169, 0.18103935334441323, 0.0289046387862526];
        let expected_nf = 27;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }


    #[test]
    fn test_real_mistake() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.43365618894869346, 0.8420868988145562, 0.7176872417127306, 0.4916610939334169, 0.18103825707549334, 0.028904638786252483]);
        let f = -2.9081562404231485;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[0.4336535629633933, 0.8420817995956955, 0.7176828957903273, 0.49165811670205656, 0.18103716080657348, 0.02890446375552885]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[0.4336588149339936, 0.8420919980334169, 0.7176915876351339, 0.4916640711647772, 0.1810393533444132, 0.028904813816976117]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.; 6]);
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);
        let mut G = SMatrix::<f64, 6, 6>::repeat(154765865.);
        let setG = true;


        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.4336535629633933, 0.8420919980334169, 0.7176915876351339, 0.4916640711647772, 0.18103716080657348, 0.028904813816976117];
        let expected_ftrip = -2.908187629699183;
        let expected_g = SVector::<f64, 6>::from_row_slice(&
            [2.8705323582816686, -1.845648568065381, -0.02892553811394936, -4.756026093917556, 0.006436196481451522, -0.8361689370862301]
        );
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            95.77846342296617, 1.7848504468928648, -0.006459576572370291, 4.655453534841051, 0.014037902478796585, 0.894694212391099, 1.7848504468928648, 45.01426462103387, -0.10270189816114665, -3.2914523096054302, 0.19693639958736797, -0.0582154345898392, -0.006459576572370291, -0.10270189816114665, 0.30851282407215563, -0.13564150643144463, 0.05546105384553863, 0.14420187864794054, 4.655453534841051, -3.2914523096054302, -0.13564150643144463, 50.08631581627649, 0.21035363638609691, -0.8769174708134662, 0.014037902478796585, 0.19693639958736797, 0.05546105384553863, 0.21035363638609691, 0.705041024447086, -0.34021712157041645, 0.894694212391099, -0.0582154345898392, 0.14420187864794054, -0.8769174708134662, -0.34021712157041645, 80.21965674487866
        ]);
        let expected_x1 = [0.43365618894869346, 0.8420817995956955, 0.7176828957903273, 0.49165811670205656, 0.18103825707549334, 0.02890446375552885];
        let expected_x2 = [0.4336588149339936, 0.8420868988145562, 0.7176872417127306, 0.4916610939334169, 0.1810393533444132, 0.028904638786252483];
        let expected_nf = 27;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }


    #[test]
    fn test_cover_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.2, -1.15, 0.01, 2.27, -1.31, 0.3]);
        let f = 12.32;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[0.2, 0.1, 0.4, 0.2, 0.3, 1.2]);
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0, -10.0, -10.0, -10.0, -10.0, -10.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ]);
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.1, 0.2, 0.4, 0.4, 0.3, 0.6];
        let expected_ftrip = -2.630060151483446;
        let expected_g = SVector::<f64, 6>::from_row_slice(&[
            -2.8999999999999755,
            -0.4300000000000163,
            -0.0400000000443656,
            -1.9205128274057135,
            1.2540564661233267,
            -2.3627224189529548,
        ]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            2.2399999999999984e+01, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, 4.1641022738722085e-15, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.5576891494122202e-10, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, 1.3360993045537445e-02, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, 2.6824336175403096e+00, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, 2.0955649004349532e+01,
        ]);
        let expected_x1 = [1.2, -1.15, 0.3, 2.27, 0.5, 0.3];
        let expected_x2 = [0.2, 0.1, 0.01, 0.2, -1.31, 1.2];
        let expected_nf = 12;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_cover_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0.15, 1.023, 0.0, 3.0, -10.0]);
        let f = -0.4;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[10.0; 6]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[-10.0; 6]);
        let u = SVector::<f64, 6>::from_row_slice(&[-5.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[6.0; 6]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            -1.0, 2.0, -3.0, -4.0, -5.0, 0.0,
            -1.0, 2.0, -3.0, -4.0, -5.0, 0.0,
            -1.0, 2.0, -3.0, -4.0, -5.0, 0.0,
            -1.0, 2.0, -3.0, -4.0, -5.0, 0.0,
            -1.0, 2.0, -3.0, -4.0, -5.0, 0.0,
            -1.0, 2.0, -3.0, -4.0, -5.0, 0.0,
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
        ]);
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [-0.2, 0.15, 1.023, 0.0, 3.0, -10.0];
        let expected_ftrip = -0.4;
        let expected_g = SVector::<f64, 6>::from_row_slice(&[
            -0.0016006402561024313,
            0.0012002700607636752,
            0.00827055374338511,
            0.0,
            0.026373626373626377,
            0.0,
        ]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.008003201280512205, 2.5, 2.5, 2.5, 2.5, 0.,
            2.5, 0.008001800405091146, 2.5, 2.5, 2.5, 0.,
            2.5, 2.5, 0.008084607764794829, 2.5, 2.5, 0.,
            2.5, 2.5, 2.5, 0.008, 2.5, 0.,
            2.5, 2.5, 2.5, 2.5, 0.008791208791208791, 0.,
            0., 0., 0., 0., 0., 0.,
        ]);
        let expected_x1 = [10.0; 6];
        let expected_x2 = [-10.0; 6];
        let expected_nf = 10;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_cover_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.15, -0.02, 0.1, 0.0, -0.4]);
        let f = -1.2;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[11.0; 6]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[-4.0; 6]);
        let u = SVector::<f64, 6>::from_row_slice(&[-20.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[60.0; 6]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            12.0, 0.1, 0.3, -0.4, 0.6, 0.12,
            12.0, 0.1, 0.3, -0.4, 0.6, 0.12,
            12.0, 0.1, 0.3, -0.4, 0.6, 0.12,
            12.0, 0.1, 0.3, -0.4, 0.6, 0.12,
            12.0, 0.1, 0.3, -0.4, 0.6, 0.12,
            12.0, 0.1, 0.3, -0.4, 0.6, 0.12,
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
            5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
            5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
            5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
            5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
            5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
        ]);
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.1, 0.15, -0.02, 0.1, 0.0, -0.4];
        let expected_ftrip = -1.2;
        let expected_g = SVector::<f64, 6>::from_row_slice(&[
            -0.18259118371000227,
            -0.17855754816501027,
            -0.19261099390434097,
            -0.18259118371000227,
            -0.19090698885134444,
            -0.22807017543859648,
        ]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.053703289326471254, 5.6, 5.6, 5.6, 5.6, 5.6,
            5.6, 0.053300760646271726, 5.6, 5.6, 5.6, 5.6,
            5.6, 5.6, 0.05471939766897285, 5.6, 5.6, 5.6,
            5.6, 5.6, 5.6, 0.053703289326471254, 5.6, 5.6,
            5.6, 5.6, 5.6, 5.6, 0.0545450723458976, 5.6,
            5.6, 5.6, 5.6, 5.6, 5.6, 0.05847953216374269,
        ]);
        let expected_x1 = [11.0; 6];
        let expected_x2 = [-4.0; 6];
        let expected_nf = 12;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_cover_3() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.112, -0.1, -0.32, 0.0, 10.0, -0.4]);
        let f = -4.5;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[11.0; 6]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[-1.0; 6]);
        let u = SVector::<f64, 6>::from_row_slice(&[-20.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[60.0; 6]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.0, 0.1, 0.4, 0.4, 0.4, 0.5,
            0.0, 0.1, 0.4, 0.4, 0.4, 0.5,
            0.0, 0.1, 0.4, 0.4, 0.4, 0.5,
            0.0, 0.1, 0.4, 0.4, 0.4, 0.5,
            0.0, 0.1, 0.4, 0.4, 0.4, 0.5,
            0.0, 0.1, 0.4, 0.4, 0.4, 0.5,
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
        ]);
        let setG = true;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.112, -0.1, -0.32, 0.0, 10.0, -0.4];
        let expected_ftrip = -4.5;
        let expected_g = SVector::<f64, 6>::from_row_slice(&[
            -3.6334635451080715,
            -4.594594594594595,
            -6.220120557002464,
            -4.090909090909091,
            4.090909096062172,
            -7.105263157894737,
        ]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.7433436057913404, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.9009009009009009, 0.5961844197086474, 0.40540540540540543, 0.03685503127875094, 0.6756756756756757,
            0.0, 0.5961844197086474, 1.1691955934221243, -6.617647058765987, -0.6016042690572945, -11.029411764609979,
            0.0, 0.40540540540540543, -6.617647058765987, 0.8181818181818182, -0.4090908471941409, -7.5,
            0.0, 0.03685503127875094, -0.6016042690572945, -0.4090908471941409, 0.8181818078647831, -0.6818180786573167,
            0.0, 0.6756756756756757, -11.029411764609979, -7.5, -0.6818180786573167, 1.3157894736842104,
        ]);
        let expected_x1 = [11.0; 6];
        let expected_x2 = [-1.0; 6];
        let expected_nf = 22;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.66]);
        let f = -3.32;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[0.201, 0.148, 0.475, 0.273, 0.307, 0.656]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[0.202, 0.146, 0.476, 0.271, 0.308, 0.654]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        ]);
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.202, 0.15, 0.476, 0.273, 0.31, 0.656];
        let expected_ftrip = -3.3220115127770096;
        let expected_g = SVector::<f64, 6>::from_row_slice(&[
            0.3562133567590662,
            0.05335723233435563,
            -0.03543761332326843,
            -0.1858482395075104,
            -0.20143664923136856,
            -0.06765510147636408,
        ]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            641.7724055589591, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 22.690208792974747, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 18.865775185676124, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 59.241589659938036, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 98.88912715464784, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 51.83419750517014,
        ]);
        let expected_x1 = [0.201, 0.148, 0.475, 0.27, 0.307, 0.66];
        let expected_x2 = [0.2, 0.146, 0.47, 0.271, 0.308, 0.654];
        let expected_nf = 12;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let f = -3.32;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[0.001, 0.999, 0.001, 0.999, 0.001, 0.999]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[0.002, 0.998, 0.002, 0.998, 0.002, 0.998]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let setG = false;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let expected_ftrip = -3.32;
        let expected_g = SVector::<f64, 6>::from_row_slice(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let expected_x1 = [0.001, 0.999, 0.001, 0.999, 0.001, 0.999];
        let expected_x2 = [0.002, 0.998, 0.002, 0.998, 0.002, 0.998];
        let expected_nf = 0;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[-0.1, 1.1, -0.1, 1.1, -0.1, 1.1]);
        let f = -3.32;
        let mut x1 = SVector::<f64, 6>::from_row_slice(&[-0.1, 1.1, -0.1, 1.1, -0.1, 1.1]);
        let mut x2 = SVector::<f64, 6>::from_row_slice(&[-0.1, 1.1, -0.1, 1.1, -0.1, 1.1]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let setG = true;

        let (xtrip, ftrip, g, nf) = triple(&x, f, &mut x1, &mut x2, &u, &v, &hess, &mut G, setG);

        let expected_xtrip = [-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let expected_ftrip = -3.32;
        let expected_g = SVector::<f64, 6>::from_row_slice(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let expected_x1 = [-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let expected_x2 = [-0.1, 1.1, -0.1, 1.1, -0.1, 1.1];
        let expected_nf = 0;

        assert_eq!(xtrip.as_slice(), expected_xtrip);
        assert_eq!(ftrip, expected_ftrip);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(x1.as_slice(), expected_x1);
        assert_eq!(x2.as_slice(), expected_x2);
        assert_eq!(nf, expected_nf);
    }
}