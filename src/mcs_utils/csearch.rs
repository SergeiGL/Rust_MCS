use crate::gls::gls;
use crate::mcs_utils::{helper_funcs::clamp_SVector, hessian::hessian, polint::polint1};
use nalgebra::{SMatrix, SVector};

const EPS_POW_1_3: f64 = 0.000006055454452393343;

pub(super) fn csearch<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    f: f64,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
) -> (
    SVector<f64, N>,     // xmin
    f64,                 // fmi
    SVector<f64, N>,     // g
    SMatrix<f64, N, N>,  // G
    usize                // nfcsearch
) {
    // TODO: hess is always 1 as nargin always < 7 < 8? Moreover hess is not returned ?

    let mut x = clamp_SVector(x, u, v);

    let (mut nfcsearch, smaxls, small) = (0_usize, 6_usize, 0.1_f64);

    let (mut xmin, mut xminnew) = (x.clone(), x.clone());
    let (mut fmi, mut fminew) = (f, f);
    let (mut x1, mut x2): (SVector<f64, N>, SVector<f64, N>) = (SVector::zeros(), SVector::zeros());
    let (mut g, mut G): (SVector<f64, N>, SMatrix::<f64, N, N>) = (SVector::zeros(), SMatrix::<f64, N, N>::zeros());
    let (mut f1, mut f2) = (f64::INFINITY, f64::INFINITY);
    let (mut alist, mut flist): (Vec<f64>, Vec<f64>) = (Vec::with_capacity(1000), Vec::with_capacity(1000));
    let mut p: SVector<f64, N> = SVector::zeros();

    for i in 0..N {
        if i != 0 { p[i - 1] = 0.0; }
        p[i] = 1.0;

        let delta = if xmin[i] != 0.0 {
            EPS_POW_1_3 * xmin[i].abs()
        } else {
            EPS_POW_1_3
        };
        let mut linesearch = true;

        if xmin[i] <= u[i] {
            f1 = func(&(xmin + delta * p));
            nfcsearch += 1;
            if f1 >= fmi {
                f2 = func(&(xmin + 2.0 * delta * p));
                nfcsearch += 1;
                x1[i] = xmin[i] + delta;
                x2[i] = xmin[i] + 2.0 * delta;
                if f2 >= fmi {
                    xminnew[i] = xmin[i];
                    fminew = fmi;
                } else {
                    xminnew[i] = x2[i];
                    fminew = f2;
                }
                linesearch = false;
            } else {
                alist = vec![0.0, delta];
                flist = vec![fmi, f1];
            }
        } else if xmin[i] >= v[i] {
            f1 = func(&(xmin - delta * p));
            nfcsearch += 1;
            if f1 >= fmi {
                f2 = func(&(xmin - 2.0 * delta * p));
                nfcsearch += 1;
                x1[i] = xmin[i] - delta;
                x2[i] = xmin[i] - 2.0 * delta;
                if f2 >= fmi {
                    xminnew[i] = xmin[i];
                    fminew = fmi;
                } else {
                    xminnew[i] = x2[i];
                    fminew = f2;
                }
                linesearch = false;
            } else {
                alist = vec![0.0, -delta];
                flist = vec![fmi, f1];
            }
        } else {
            alist = vec![0.0];
            flist = vec![fmi];
        }

        if linesearch {
            nfcsearch += gls(func, &xmin, &p, &mut alist, &mut flist, u, v, 1, small, smaxls);
            let mut j;

            (j, fminew) = flist
                .iter()
                .enumerate()
                .min_by(|(_, val_1), (_, val_2)| val_1.total_cmp(val_2))
                .map(|(j, fminew)| (j, *fminew))
                .unwrap();

            if fminew == fmi {
                j = alist.iter().position(|&alist_i| alist_i == 0.0).unwrap();
            }

            // Process both arrays in a single pass with swap-and-truncate pattern
            let alist_j = alist[j];
            let mut write_idx = 0;
            for read_idx in 0..alist.len() {
                let a_i = alist[read_idx];
                // Check if this element should be kept
                if (a_i - alist_j).abs() >= delta || read_idx == j {
                    // Keep this element by moving it to the write position
                    if read_idx != write_idx {
                        alist.swap(read_idx, write_idx);
                        flist.swap(read_idx, write_idx);
                    }
                    write_idx += 1;
                }
            }

            // Truncate both vectors to the new size
            alist.truncate(write_idx);
            flist.truncate(write_idx);

            (j, fminew) = flist.iter()
                .enumerate()
                .min_by(|(_, val_1), (_, val_2)| val_1.total_cmp(val_2))
                .map(|(j, fminew)| (j, *fminew))
                .unwrap();

            // println!("xminnew[i] ={} {} ; {i}, {j}", xmin[i], alist[4]);
            xminnew[i] = xmin[i] + alist[j];
            // println!("xminnew[i] ={xminnew:?}");

            if i == 0 || alist[j] == 0.0 {
                if j == 0 {
                    x1[i] = xmin[i] + alist[1];
                    f1 = flist[1];
                    x2[i] = xmin[i] + alist[2];
                    f2 = flist[2];
                } else if j == alist.len() - 1 {
                    x1[i] = xmin[i] + alist[j - 1];
                    f1 = flist[j - 1];
                    x2[i] = xmin[i] + alist[j - 2];
                    f2 = flist[j - 2];
                } else {
                    x1[i] = xmin[i] + alist[j - 1];
                    f1 = flist[j - 1];
                    x2[i] = xmin[i] + alist[j + 1];
                    f2 = flist[j + 1];
                }

                xmin[i] = xminnew[i];
                fmi = fminew;
            } else {
                x1[i] = xminnew[i];
                f1 = fminew;
                if xmin[i] < x1[i] && j < alist.len() - 1 {
                    x2[i] = xmin[i] + alist[j + 1];
                    f2 = flist[j + 1];
                } else if j == 0 {
                    if alist[j + 1] != 0.0 {
                        x2[i] = xmin[i] + alist[j + 1];
                        f2 = flist[j + 1];
                    } else {
                        x2[i] = xmin[i] + alist[j + 2];
                        f2 = flist[j + 2];
                    }
                } else if alist[j - 1] != 0.0 {
                    x2[i] = xmin[i] + alist[j - 1];
                    f2 = flist[j - 1];
                } else {
                    x2[i] = xmin[i] + alist[j - 2];
                    f2 = flist[j - 2];
                }
            }
        }

        // println!("polint_x: {:?}, polint_f: {:?}", polint_x, polint_f);
        (g[i], G[(i, i)]) = polint1(&[xmin[i], x1[i], x2[i]], &[fmi, f1, f2]);

        x = xmin;

        if f1 <= f2 {
            x[i] = x1[i];
        } else {
            x[i] = x2[i];
        }

        // println!("Grand before {xminnew:?}");
        let mut k1: Option<usize> = None;
        for k in 0..i { // i: -1 from Matlab
            let q1 = fmi
                + g[k] * (x1[k] - xmin[k])
                + 0.5 * G[(k, k)] * (x1[k] - xmin[k]).powi(2);
            let q2 = fmi
                + g[k] * (x2[k] - xmin[k])
                + 0.5 * G[(k, k)] * (x2[k] - xmin[k]).powi(2);
            if q1 <= q2 {
                x[k] = x1[k];
            } else {
                x[k] = x2[k];
            }
            let f12 = func(&x);
            nfcsearch += 1;
            G[(i, k)] = hessian(i, k, &x, &xmin, f12, fmi, &g, &G);
            G[(k, i)] = G[(i, k)];

            if f12 < fminew {
                fminew = f12;
                xminnew = x.clone();
                k1 = Some(k);
            }
            x[k] = xmin[k];
        }

        if fminew <= fmi {
            if x1[i] == xminnew[i] {
                x1[i] = xmin[i];
            } else if x2[i] == xminnew[i] {
                x2[i] = xmin[i];
            }
            if let Some(k1) = k1 { // k: -1 from Matlab
                if xminnew[k1] == x1[k1] {
                    x1[k1] = xmin[k1];
                } else if xminnew[k1] == x2[k1] {
                    x2[k1] = xmin[k1];
                }
            }
            for k in 0..=i { // i: -1 from Matlab
                g[k] += G[(i, k)] * (xminnew[i] - xmin[i]);
                if let Some(k1) = k1 {
                    g[k] += G[(k1, k)] * (xminnew[k1] - xmin[k1]);
                }
            }
        }
        xmin = xminnew;
        fmi = fminew;
    }

    (xmin, fmi, g, G, nfcsearch)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-12;

    #[test]
    fn test_matlab_0() {
        // Matlab Equivalent test
        //
        // clearvars; clear global;
        // fcn = "feval"; data = "hm6"; path(path,'jones'); stop = [100];
        // x = [1.2, -0.1, 0.0, -0.01, 1.0, 1.6]';
        // f = 100.;
        // u = zeros(6,1);
        // v = ones(6,1);
        // hess = ones(6,6);
        //
        // format long g;
        // [xmin,fmi,g,G,nfcsearch] = csearch(fcn,data,x,f,u,v,hess)

        let x = SVector::<f64, 6>::from_row_slice(&[1.2, -0.1, 0.0, -0.01, 1.0, 1.6]);
        let f = 100.;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        assert_relative_eq!(xmin.as_slice(), [0.,0.171229504985996,0.75728402737634,0.250004541590839,0.333331314848516,0.695335531985148].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -2.16939912649297, epsilon = TOLERANCE);
        assert_relative_eq!(g.as_slice(), [0.437359199626315,-3.22197789089883,1.75837841612486,-3.76358531848627, 0.189362876830768, 1.08053992507632].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(G.as_slice(), [
                    -0.00126223495093886 ,     0.000715238883325839 ,         0.54035017594183   ,     0.0130243025356302   ,     0.0656954295588853 ,       -0.219595714873091,
                    0.000715238883325839  ,     0.00682787168712506  ,       0.115595291020514    ,     -13.6053920417095    ,   -0.0549827859164636  ,      -0.181199475327673,
                    0.54035017594183   ,      0.115595291020514   ,        2.0032818047489     ,    0.017904718685595     ,     1.48579944105136   ,      -6.97028826467059,
                    0.0130243025356302    ,     -13.6053920417095    ,     0.017904718685595      ,    1.26511906055634      ,    3.73591628610961    ,      5.17472626468738,
                    0.0656954295588853     ,  -0.0549827859164636     ,     1.48579944105136       ,   3.73591628610961       ,   2.82375017035131     ,     2.72729224432554,
                    -0.219595714873091      ,  -0.181199475327673      ,   -6.97028826467059        ,  5.17472626468738        ,  2.72729224432554      ,    9.98582878734768,
                ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(nfcsearch, 46);
    }


    #[test]
    fn test_matlab_1() {
        // Matlab Equivalent test
        //
        // clearvars; clear global;
        // fcn = "feval"; data = "hm6"; path(path,'jones'); stop = [100];
        // x = [1.1, -0.1, 0.1, -0.01, 0.9, 1.2]';
        // f = -10.;
        // u = zeros(6,1);
        // v = ones(6,1)*1.1;
        // hess = ones(6,6);
        //
        // format long g;
        // [xmin,fmi,g,G,nfcsearch] = csearch(fcn,data,x,f,u,v,hess)

        let x = SVector::<f64, 6>::from_row_slice(&[1.1, -0.1, 0.1, -0.01, 0.9, 1.2]);
        let f = -10.;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.1; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        assert_relative_eq!(xmin.as_slice(), [1.1, 0., 0.1, 0., 0.9,1.1].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -10., epsilon = TOLERANCE);
        assert_relative_eq!(g.as_slice(), [-2251900.249836313, 2477090.2751214015, -199.99907414611474, 2477090.274831074, 16.809552839333453, -2251900.249730433            ].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(G.as_slice(), [-225381602838.39246, 61979940780.77809, 30025317.92886204, 61979940780.77958, -9997831.201668503, -56345400709.60008, 61979940780.77809, -272711739436.39044, -33027849.721561164, -68177934859.09798, 10997614.30979206, 61979940780.778145, 30025317.92886204, -33027849.721561164, 47999.673545437836, -33027849.721762765, 5327.643081804327, 30025317.928907093, 61979940780.77958, -68177934859.09798, -33027849.721762765, -272711739436.3917, 10997614.322017025, 61979940780.780174, -9997831.201668503, 10997614.30979206, 5327.643081804327, 10997614.322017025, 3995.7066291470805, -9997831.20623445, -56345400709.60008, 61979940780.778145, 30025317.928907093, 61979940780.780174, -9997831.20623445, -225381602838.3932].as_slice(), epsilon = TOLERANCE);
        assert_eq!(nfcsearch, 33);
    }


    #[test]
    fn test_5() {
        // Matlab Equivalent test
        //
        // clearvars; clear global;
        // fcn = "feval"; data = "hm6"; path(path,'jones'); stop = [100];
        // x = [0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999]';
        // f = -2.728684905407648;
        // u = zeros(6,1);
        // v = ones(6,1);
        // hess = ones(6,6);
        //
        // format long g;
        // [xmin,fmi,g,G,nfcsearch] = csearch(fcn,data,x,f,u,v,hess)

        let x = SVector::<f64, 6>::from_row_slice(&[0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999]);
        let f = -2.728684905407648;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        assert_relative_eq!(xmin.as_slice(), [0.20094711239564478, 0.1495167421889697, 0.45913871, 0.2559626362565819, 0.33160230910548794, 0.6275210838397162].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -3.2661659570240427, epsilon = TOLERANCE);
        assert_relative_eq!(g.as_slice(), [0.0114919524849414, 0.10990155244416452, -0.5975771816968102, -0.8069326056544469, 1.8713998467574906, -1.4958051414638653].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(G.as_slice(), [23.12798652584253, 0.0808647397791913, 1.75381629525258, -1.90128293323017, 1.786461227928722, -0.7406818881368897, 0.0808647397791913, 18.5767212985667, -0.5909985456367552, 0.801357349181761, -0.9992079198192292, 0.18105617066417665, 1.75381629525258, -0.5909985456367552, 24.556579083791647, 3.3716142085157075, -3.5009378170621, 0.09958957165243534, -1.90128293323017, 0.801357349181761, 3.3716142085157075, 48.678472018407895, -1.0333246379474361, 0.9233898437178975, 1.786461227928722, -0.9992079198192292, -3.5009378170621, -1.0333246379474361, 89.37343113076405, 4.016171463396038, -0.7406818881368897, 0.18105617066417665, 0.09958957165243534, 0.9233898437178975, 4.016171463396038, 48.17000841044073].as_slice(), epsilon = TOLERANCE);
        assert_eq!(nfcsearch, 50);
    }

    #[test]
    fn test_random_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[8.527395349768865, 7.0360099156585285, 9.476097788271057, 0.9596866992140249, 9.72767113420278, 11.818974094659938]);
        let f = 0.01;
        let u = SVector::<f64, 6>::from_row_slice(&[1.5692744390150857, 3.4940496081522063, 5.747100035185964, 5.58384046504999, 2.793673436034129, 1.5739936351217687]);
        let v = SVector::<f64, 6>::from_row_slice(&[2.5692744390150857, 4.494049608152206, 6.747100035185964, 6.58384046504999, 3.793673436034129, 2.5739936351217687]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        assert_relative_eq!(xmin.as_slice(), [1.5692744390150857, 3.4940496081522063, 5.747100035185964, 5.58384046504999, 2.793673436034129, 1.5739936351217687].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -4.382690949216887e-158, epsilon = TOLERANCE);
        assert_relative_eq!(g.as_slice(), [8.765518273141369e-158, 1.1474346043336644e-157, 2.018267794287133e-158, 4.383638860064236e-156, 2.2628965429130008e-158, 1.585690382258133e-157].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(G.as_slice(), SMatrix::<f64, 6, 6>::from_row_slice(&[-2.2786483959708093e-204, -4.740423592365374e-183, -3.3729722286362593e-183, -8.115150627808157e-181, -7.226103980100922e-183, -8.765518273141369e-158, -4.740423592365374e-183, -1.2410734372571352e-182, -4.415329400946724e-183, -1.062299382599952e-180, -9.459203098905338e-183, -1.1474346043336644e-157, -3.3729722286362593e-183, -4.415329400946724e-183, -5.878369166271537e-184, -1.8685201087077833e-181, -1.6638163867498336e-183, -2.018267794287133e-158, -8.115150627808157e-181, -1.062299382599952e-180, -1.8685201087077833e-181, -4.0503312664371886e-179, -3.6137772151014935e-181, -4.383638860064236e-156, -7.226103980100922e-183, -9.459203098905338e-183, -1.6638163867498336e-183, -3.6137772151014935e-181, -1.1230051052493818e-183, -2.2628965429130008e-158, -8.765518273141369e-158, -1.1474346043336644e-157, -2.018267794287133e-158, -4.383638860064236e-156, -2.2628965429130008e-158, -2.294842574672889e-157]).as_slice(), epsilon = TOLERANCE);
        assert_eq!(nfcsearch, 32);
    }

    #[test]
    fn test_cover_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[-0.5, -0.61, -0.7, -0.8, -0.09, -0.5]);
        let f = 15.0;
        let u = SVector::<f64, 6>::from_row_slice(&[10.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[20.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        assert_eq!(xmin.as_slice(), [10.000020184848175, 10., 10., 10., 10., 10.]);
        assert_eq!(fmi, 0.0);
        assert_eq!(g, SVector::<f64, 6>::from_row_slice(&[0., 0., 0., 0., 0., 0.]));
        assert_eq!(G, SMatrix::<f64, 6, 6>::from_row_slice(&[0.; 36]));
        assert_eq!(nfcsearch, 30);
    }

    #[test]
    fn test_cover_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[-0.5, -0.61, -0.7, -0.8, -0.09, -0.5]);
        let f = 10.0;
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[12.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_g = SVector::<f64, 6>::from_row_slice(&[-0.02105800978773433, 0.0049572412272171455, 0.018418190128533754, -0.016036235142429266, -0.04441832232932502, 0.010042152909065433]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1.5696814271457484e-12, -3.1061316413703195e-11, 1.2010834746673555e-09, -0.0004052472582463076, 0.0015442042714962024, -0.015439829404769616, -3.1061316413703195e-11, 5.619548494927433e-11, 4.065469329511873e-09, -8.374530892854174e-05, 0.0012885397682397793, 0.003415845189902907, 1.2010834746673555e-09, 4.065469329511873e-09, 2.1828250505657833e-09, -3.532526887736334e-06, 1.8092611470004093e-06, 0.013448592373580695, -0.0004052472582463076, -8.374530892854174e-05, -3.532526887736334e-06, 0.0001912679454339812, 2.6847930204653707e-05, -0.011670149785391744, 0.0015442042714962024, 0.0012885397682397793, 1.8092611470004093e-06, 2.6847930204653707e-05, 2.7346432586681816e-05, -0.032428401108624064, -0.015439829404769616, 0.003415845189902907, 0.013448592373580695, -0.011670149785391744, -0.032428401108624064, 0.05419611001381587
        ]);

        assert_eq!(xmin.as_slice(), [0.0, 1.0035377004597188, 0.0, 0.0, 0.1781739563644665, 0.8696995987258165]);
        assert_eq!(fmi, -0.03719287278329851);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 48);
    }

    #[test]
    fn test_cover_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.22, -0.31, 0.153, 1.24, 1.5, -0.15]);
        let f = -3.0;
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[12.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_g = SVector::<f64, 6>::from_row_slice(&[2.6151445640718367, 12.442396312277596, 25.2100840332843, 2.470830473949214, 0.8571428572094838, 25.714285713952464]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            23.419203030767363, 46.08296002919352, 93.37068495191433, -7.5988245751419665, -8.163265705981399, 95.23809902438819, 46.08296002919352, 321.09409842553345, -569.2599620479299, 46.328071399681725, 49.76958525378894, -580.6451612895457, 93.37068495191433, -569.2599620479299, 1318.1743285558548, 93.86733417802581, 100.84033613446475, -1176.4705882335106, -7.5988245751419665, 46.328071399681725, 93.86733417802581, 23.164035687965598, -8.206687525245558, 95.74468086148659, -8.163265705981399, 49.76958525378894, 100.84033613446475, -8.206687525245558, 20.57142857113299, 102.85714285923983, 95.23809902438819, -580.6451612895457, -1176.4705882335106, 95.74468086148659, 102.85714285923983, 1371.4285714297291
        ]);

        assert_eq!(xmin.as_slice(), [1.22, -0.31, 0.153, 1.24, 1.5, -0.15]);
        assert_eq!(fmi, -3.);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 57);
    }

    #[test]
    fn test_cover_3() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.12, 0.31, 125.153, -9.24, 0.5, -0.15]);
        let f = -2.0;
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[12.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_g = SVector::<f64, 6>::from_row_slice(&[2.2643593519882197, 8.294930875576037, -41285.09296299202, 0.816696914394655, 5.142857142857147, 17.14285714285715]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            16.568483063328422, 29.62475312705728, 73723.38029105718, 7.04887218045113, 18.367346938775515, 61.224489795918366, 29.62475312705728, 214.06273227293, -152203.10769766645, -14.55251030802814, -37.91968400263331, -126.39894667544438, 73723.38029105718, -152203.10769766645, -378768644.65842074, -36214.993827185994, -94365.92677255321, -314553.0892418439, 7.04887218045113, -14.55251030802814, -36214.993827185994, 9.074410162534415, -9.022556390977446, -30.07518796992481, 18.367346938775515, -37.91968400263331, -94365.92677255321, -9.022556390977446, 82.2857142857143, 137.14285714285717, 61.224489795918366, -126.39894667544438, -314553.0892418439, -30.07518796992481, 137.14285714285717, 914.2857142857144
        ]);

        assert_eq!(xmin.as_slice(), [1.12, 0.31, 12., -9.24, 0.5, -0.15]);
        assert_eq!(fmi, -2.);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 50);
    }

    #[test]
    fn test_cover_4() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.12, 0.31, -125.153, -9.24, 0.5, -0.15]);
        let f = -2.0;
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[12.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_g = SVector::<f64, 6>::from_row_slice(&[2.2643593519882197, 8.294930875576037, 49542.11155583264, 0.8166969142962373, 5.142857142857147, 17.14285714285715]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            16.568483063328422, 29.62475312705728, -88468.05634970115, 7.048872180451129, 18.367346938775515, 61.224489795918366, 29.62475312705728, 214.06273227293, 182643.7292380927, -14.552510308028138, -37.91968400263331, -126.39894667544438, -88468.05634970115, 182643.7292380927, -545426848.3134592, 43457.99259283565, 113239.1121276175, 377463.7070920582, 7.048872180451129, -14.552510308028138, 43457.99259283565, 9.07441016227542, -9.022556390977448, -30.075187969924816, 18.367346938775515, -37.91968400263331, 113239.1121276175, -9.022556390977448, 82.2857142857143, 137.14285714285717, 61.224489795918366, -126.39894667544438, 377463.7070920582, -30.075187969924816, 137.14285714285717, 914.2857142857144
        ]);

        assert_eq!(xmin.as_slice(), [1.12, 0.31, -10., -9.24, 0.5, -0.15]);
        assert_eq!(fmi, -2.);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 50);
    }

    #[test]
    fn test_cover_5() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.9, -2.8, 3., 7., -4.6, 5.5]);
        let f = -1.0;
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[12.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_g = SVector::<f64, 6>::from_row_slice(&[-0.35653650254668956, -0.6493506493506492, -1.9999999999999987, -0.45714285714285685, -0.05847953216374191, -1.575757575757576]);
        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            6.112054329371818, -1.1278195488721803, 4.736842105263156, 1.3533834586466162, -1.6620498614958463, 4.2105263157894735, -1.1278195488721803, 1.9480519480519478, 2.142857142857142, 0.6122448979591832, -0.7518796992481206, 1.9047619047619042, 4.736842105263156, 2.142857142857142, 5.999999999999997, -2.5714285714285694, 3.157894736842106, -7.999999999999995, 1.3533834586466162, 0.6122448979591832, -2.5714285714285694, 0.6857142857142855, 0.9022556390977449, -2.2857142857142847, -1.6620498614958463, -0.7518796992481206, 3.157894736842106, 0.9022556390977449, 2.339181286549708, -2.9629629629629615, 4.2105263157894735, 1.9047619047619042, -7.999999999999995, -2.2857142857142847, -2.9629629629629615, 5.818181818181816
        ]);

        assert_eq!(xmin.as_slice(), [1.9, -2.8, 3., 7., -4.6, 5.5]);
        assert_eq!(fmi, -1.);
        assert_eq!(g, expected_g);
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 51);
    }

    #[test]
    fn test_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.45, 0.56, 0.68, 0.72]);
        let f = -2.7;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            33610.72860793838, -31917.574765584413, 7276.950901772966, -5657.831243106884, -9877.962761593055, -6656.034760465145, -31917.574765584413, 33735.973427798126, 7454.000105222104, -5795.584485949894, -10120.79020418948, -6818.460877403494, 7276.950901772966, 7454.000105222104, 7139.373385105945, 1323.7423396357972, 2306.156662939039, 1555.4803515240671, -5657.831243106884, -5795.584485949894, 1323.7423396357972, 4012.286818921415, -1803.2451971949533, -1210.5123749494508, -9877.962761593055, -10120.79020418948, 2306.156662939039, -1803.2451971949533, 9625.122952165366, -2110.9690207169406, -6656.034760465145, -6818.460877403494, 1555.4803515240671, -1210.5123749494508, -2110.9690207169406, 7427.872886675865
        ]);

        assert_eq!(xmin.as_slice(), [0.2, 0.2, 0.45, 0.56, 0.68, 0.72]);
        assert_eq!(fmi, -2.7);
        assert_eq!(g, SVector::<f64, 6>::from_row_slice(&[-124.21565138002285, -133.97466510318048, -70.18409712868349, 48.39130644417342, 47.671997661330856, 93.56555490452016]));
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 57);
    }

    #[test]
    fn test_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let f = -1.35;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[2.1, 2.2, 2.3, 2.4, 2.5, 2.6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            -36815402316.54175, -9203850579.135637, -9203850579.13714, 668801.9766471039, 2668385.501245511, 2674890.8722308264, -9203850579.135637, -36815402316.542015, -9203850579.136158, 668801.9760468211, 2668385.5011991933, 2674890.871860553, -9203850579.13714, -9203850579.136158, -36815402316.55257, 668801.9825047515, 2668385.5015069894, 2674890.8759595747, 668801.9766471039, 668801.9760468211, 668801.9825047515, 64.81480786627557, -193.9018085234481, -194.37504360388937, 2668385.501245511, 2668385.5011991933, 2668385.5015069894, -193.9018085234481, 776.5949442178951, -775.5062594158967, 2674890.8722308264, 2674890.871860553, 2674890.8759595747, -194.37504360388937, -775.5062594158967, 777.5345833891469
        ]);

        assert_eq!(xmin.as_slice(), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(fmi, -1.35);
        assert_eq!(g, SVector::<f64, 6>::from_row_slice(&[334400.9877256005, 334400.9877775648, 334400.9872032684, -2.698149299274311, -16.096078870312834, -16.194566535234358]));
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 40);
    }

    #[test]
    fn test_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 0.6, 0.7, 0.8, 0.9, 1.5]);
        let f = 0.1;
        let u = SVector::<f64, 6>::from_row_slice(&[-1.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_g = SVector::<f64, 6>::from_row_slice(&[-1.553577178019332, 1.6277375974962205, 1.5623144452891058, 1.1146776795991615, -2.5576875518394755, 1.0365978929648942]);

        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.8429779496902778, 0.038527425030238774, 0.08016691619464542, 0.4198271251725716, -1.0089458453227638, 6.442359154330082, 0.038527425030238774, 0.1585317564251635, -0.08074702867269075, -0.11732762041885024, 0.5192671643626129, -5.802272526736849, 0.08016691619464542, -0.08074702867269075, 0.739027409525485, 0.1953843687434614, 1.6410915456775919, -8.6178048318381, 0.4198271251725716, -0.11732762041885024, 0.1953843687434614, 0.8648855174952936, 3.6425765888725365, -10.82478073553638, -1.0089458453227638, 0.5192671643626129, 1.6410915456775919, 3.6425765888725365, 4.9506593399538215, 5.781450885439346, 6.442359154330082, -5.802272526736849, -8.6178048318381, -10.82478073553638, 5.781450885439346, 12.50714291630882
        ]);

        assert_relative_eq!(xmin.as_slice(), [0.190983, 0.1923299222415401, 0.5490777434437788, 0.0, 0.27333123819254646, 0.6794905377752728].as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -1.7536640842012834, epsilon = TOLERANCE);
        assert_relative_eq!(g.as_slice(), expected_g.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(G.as_slice(), expected_G.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nfcsearch, 48);
    }

    #[test]
    fn test_3() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.4, 0.6, 0.8, 0.1, 0.3]);
        let f = -1.5;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1617.141765629833, -239.60638966090045, 2413.746317004152, 354.3488495459659, -1060.9751839203611, 542.3729072815653, -239.60638966090045, 364.2341178708943, 1093.9148531450705, 161.5036714685619, -480.46362460362, 249.26445612335806, 2413.746317004152, 1093.9148531450705, 10473.469754959657, -1639.8005196209724, 4930.317367115214, -2504.700458062508, 354.3488495459659, 161.5036714685619, -1639.8005196209724, 546.5174982652572, 723.530373324211, -365.5301863497862, -1060.9751839203611, -480.46362460362, 4930.317367115214, 723.530373324211, 6606.16130771491, 1099.8765899392931, 542.3729072815653, 249.26445612335806, -2504.700458062508, -365.5301863497862, 1099.8765899392931, 734.8681806295466
        ]);

        assert_eq!(xmin.as_slice(), [0.2, 0.4, 0.6, 0.8, 0.1, 0.3]);
        assert_eq!(fmi, -1.5);
        assert_eq!(g, SVector::<f64, 6>::from_row_slice(&[-15.364962350081559, -9.126032408929731, -66.72517400621334, 3.3001784827451157, -29.091529160361628, -9.393402972970774]));
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 47);
    }

    #[test]
    fn test_4() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 1.0, 1.0, 0.5, 0.5]);
        let f = -0.2;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.5, 0.5]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]);

        let (xmin, fmi, g, G, nfcsearch) = csearch(hm6, &x, f, &u, &v);

        let expected_G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            -5321411539.7519655, -1330352884.950399, 1330352884.9861374, 1330352885.0233428, -5321411539.816137, -2660705769.886557, -1330352884.950399, -5321411539.804666, 1330352885.018136, 1330352884.9994442, -5321411539.765417, -2660705770.007373, 1330352884.9861374, 1330352885.018136, -5321411539.906169, -1330352885.124455, 5321411539.7908, 2660705770.0038385, 1330352885.0233428, 1330352884.9994442, -1330352885.124455, -5321411540.356289, 5321411539.578128, 2660705770.007158, -5321411539.816137, -5321411539.765417, 5321411539.7908, 5321411539.578128, -21285646159.31742, -10642823079.480272, -2660705769.886557, -2660705770.007373, 2660705770.0038385, 2660705770.007158, -10642823079.480272, -21285646159.405983
        ]);

        assert_eq!(xmin.as_slice(), [0.0, 0.0, 1.0, 1.0, 0.5, 0.5]);
        assert_eq!(fmi, -0.2);
        assert_eq!(g, SVector::<f64, 6>::from_row_slice(&[48335.34155263062, 48335.334904356554, -48335.32412470991, -48335.299023819374, 96670.7221889008, 96670.67130089863]));
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 27);
    }
}