use crate::feval::feval;
use crate::gls::gls;
use crate::hessian::hessian;
use crate::polint::polint1;


pub fn csearch(
    x: &[f64],
    f: f64,
    u: &[f64],
    v: &[f64],
) -> (
    Vec<f64>, //xmin
    f64, //fmi
    Vec<f64>, //g
    Vec<Vec<f64>>, //G
    usize //nfcsearch
) {
    let n = x.len();

    let mut x = (0..n).map(|i| {
        x[i].max(u[i]).min(v[i])
    }).collect::<Vec<f64>>();

    let mut nfcsearch: usize = 0;
    let smaxls: usize = 6;
    let small: f64 = 0.1;
    let nloc: i32 = 1;

    // Initialize Hessian matrix as ones
    // let hess_matrix = vec![vec![1.0; n]; n];
    let mut xmin = x.clone();
    let mut fmi = f;
    let mut xminnew = xmin.clone();
    let mut fminew = fmi;
    let mut g = vec![0.0; n];
    let mut x1 = vec![0.0; n];
    let mut x2 = vec![0.0; n];
    let mut G = vec![vec![0.0; n]; n];
    let eps: f64 = 2.220446049250313e-16;

    // Initialize f1 and f2
    let mut f1 = fmi;
    let mut f2 = fmi;

    for i in 0..n {
        let mut p = vec![0.0; n];
        p[i] = 1.0;
        let delta = if xmin[i] != 0.0 {
            eps.powf(1.0 / 3.0) * xmin[i].abs()
        } else {
            eps.powf(1.0 / 3.0)
        };
        let mut linesearch = true;
        let mut alist: Vec<f64> = Vec::new();
        let mut flist: Vec<f64> = Vec::new();

        if xmin[i] <= u[i] {
            f1 = feval(&xmin.iter().zip(&p).map(|(&x, &p)| x + delta * p).collect::<Vec<_>>());
            nfcsearch += 1;
            if f1 >= fmi {
                f2 = feval(&xmin.iter().zip(&p).map(|(&x, &p)| x + 2.0 * delta * p).collect::<Vec<_>>());
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
            f1 = feval(&xmin.iter().zip(&p).map(|(&x, &p)| x - delta * p).collect::<Vec<_>>());
            nfcsearch += 1;
            if f1 >= fmi {
                f2 = feval(&xmin.iter().zip(&p).map(|(&x, &p)| x - 2.0 * delta * p).collect::<Vec<_>>());
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

        // println!("if linesearch:: {g:?}");
        if linesearch {
            // Perform line search using GLS
            // println!("csearch: {xmin:?}\n {p:?}\n {alist:?}\n {flist:?}\n {u:?}\n {v:?}\n {nloc}\n {small}\n {smaxls}");
            let nfls = gls(&xmin, &p, &mut alist, &mut flist, u, v, nloc, small, smaxls);
            // println!("gls result {alist:?}\n{flist:?}\n{nfls}");
            nfcsearch += nfls;

            // Find the index of the minimum in flist
            let (mut j, min_f) = flist
                .iter()
                .enumerate()
                .min_by(|(_, val_1), (_, val_2)| val_1.partial_cmp(val_2).unwrap())
                .unwrap().clone();
            fminew = *min_f;

            if fminew == fmi {
                j = alist.iter().position(|&x| x == 0.0).unwrap_or(0);
            }
            // println!("\n{flist:?}\n {j}\n {fminew}\n");

            // Find indices where |alist[k] - alist[j]| < delta
            let alist_j = alist[j];
            let mut ind: Vec<usize> = alist
                .iter()
                .enumerate()
                .filter_map(|(k, &x)| if (x - alist_j).abs() < delta { Some(k) } else { None })
                .collect();

            // Remove the index j from ind
            ind.retain(|&k| k != j);

            // Remove these indices from alist and flist in reverse order to avoid shifting
            for &k in ind.iter().rev() {
                alist.remove(k);
                flist.remove(k);
            }

            // Find the new index of the minimum in flist
            let (j, min_f) = flist
                .iter()
                .enumerate()
                .min_by(|(_, &val_1), (_, val_2)| val_1.partial_cmp(val_2).unwrap())
                .unwrap();
            fminew = *min_f;

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

        // Call polynomial interpolation
        let polint_x: [f64; 3] = [xmin[i], x1[i], x2[i]];
        let polint_f: [f64; 3] = [fmi, f1, f2];
        // println!("polint_x: {:?}, polint_f: {:?}", polint_x, polint_f);
        (g[i], G[i][i]) = polint1(&polint_x, &polint_f);

        x = xmin.clone();

        if f1 <= f2 {
            x[i] = x1[i];
        } else {
            x[i] = x2[i];
        }

        // println!("Grand before {xminnew:?}");
        let mut k1: Option<usize> = None;
        for k in 0..i {
            if true {
                let q1 = fmi
                    + g[k] * (x1[k] - xmin[k])
                    + 0.5 * G[k][k] * (x1[k] - xmin[k]).powi(2);
                let q2 = fmi
                    + g[k] * (x2[k] - xmin[k])
                    + 0.5 * G[k][k] * (x2[k] - xmin[k]).powi(2);
                if q1 <= q2 {
                    x[k] = x1[k];
                } else {
                    x[k] = x2[k];
                }
                let f12 = feval(&x);
                nfcsearch += 1;
                let hess_val =
                    hessian(i, k, &x, &xmin, f12, fmi, &g, &G);
                G[i][k] = hess_val;
                G[k][i] = hess_val;
                if f12 < fminew {
                    fminew = f12;
                    // println!("f12 < fminew: {x:?}");
                    xminnew = x.clone();
                    k1 = Some(k);
                }
                x[k] = xmin[k];
            } else {
                G[i][k] = 0.0;
                G[k][i] = 0.0;
            }
        }
        // println!("bef{g:?}");

        if fminew <= fmi {
            if x1[i] == xminnew[i] {
                x1[i] = xmin[i];
            } else if x2[i] == xminnew[i] {
                x2[i] = xmin[i];
            }
            if let Some(k1) = k1 {
                if xminnew[k1] == x1[k1] {
                    x1[k1] = xmin[k1];
                } else if xminnew[k1] == x2[k1] {
                    x2[k1] = xmin[k1];
                }
            }
            for k in 0..=i {
                g[k] += G[i][k] * (xminnew[i] - xmin[i]);
                if let Some(k1) = k1 {
                    g[k] += G[k1][k] * (xminnew[k1] - xmin[k1]);
                }
            }
        }

        // println!("aft{g:?}");

        xmin = xminnew.clone();
        // println!("after {xmin:?}");
        fmi = fminew;
    }

    (xmin, fmi, g, G, nfcsearch)
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_cover_0() {
        let x = vec![-0.5, -0.61, -0.7, -0.8, -0.09, -0.5];
        let f = 15.0;
        let u = vec![10.0; 6];
        let v = vec![20.0; 6];

        let (xmin, fmi, g, G, nfcsearch) = csearch(&x, f, &u, &v);
        assert_eq!(xmin, vec![10.000020184848175, 10., 10., 10., 10., 10.]);
        assert_eq!(fmi, 0.0);
        assert_eq!(g, vec![0., 0., 0., 0., 0., 0.]);
        assert_eq!(G, vec![vec![0.; 6]; 6]);
        assert_eq!(nfcsearch, 30);
    }

    #[test]
    fn test_cover_1() {
        let x = vec![-0.5, -0.61, -0.7, -0.8, -0.09, -0.5];
        let f = 10.0;
        let u = vec![-10.0; 6];
        let v = vec![12.0; 6];

        let (xmin, fmi, g, G, nfcsearch) = csearch(&x, f, &u, &v);
        assert_eq!(xmin, vec![0.0, 1.0035377004597188, 0.0, 0.0, 0.1781739563644665, 0.8696995987258165]);
        assert_eq!(fmi, -0.03719287278329851);
        // assert_eq!(g, vec![-0.02105800978773433, 0.0049572412272171455, 0.018418190128533754, -0.016036235142429266, -0.04441832232932502, 0.010042152909065433]);
        // assert_eq!(G,);
        assert_eq!(nfcsearch, 48);
    }


    #[test]
    fn test_0() {
        let x = vec![0.2, 0.2, 0.45, 0.56, 0.68, 0.72];
        let f = -2.7;
        let u = vec![0.0; 6];
        let v = vec![1.0; 6];

        let (xmin, fmi, g, G, nfcsearch) = csearch(&x, f, &u, &v);

        assert_eq!(xmin, vec![0.2, 0.2, 0.45, 0.56, 0.68, 0.72]);
        assert_eq!(fmi, -2.7);
        assert_eq!(g, vec![-124.21565138002285, -133.97466510318048, -70.18409712868349, 48.39130644417342, 47.671997661330856, 93.56555490452016]);

        let expected_G = vec![
            vec![33610.72860793838, -31917.574765584413, 7276.950901772964, -5657.831243106882, -9877.962761593053, -6656.034760465144],
            vec![-31917.574765584413, 33735.973427798126, 7454.000105222104, -5795.584485949892, -10120.790204189478, -6818.460877403494],
            vec![7276.950901772964, 7454.000105222104, 7139.373385105945, 1323.7423396357967, 2306.1566629390386, 1555.4803515240671],
            vec![-5657.831243106882, -5795.584485949892, 1323.7423396357967, 4012.286818921415, -1803.2451971949522, -1210.5123749494503],
            vec![-9877.962761593053, -10120.790204189478, 2306.1566629390386, -1803.2451971949522, 9625.122952165366, -2110.96902071694],
            vec![-6656.034760465144, -6818.460877403494, 1555.4803515240671, -1210.5123749494503, -2110.96902071694, 7427.872886675865]
        ];

        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 57);
    }

    #[test]
    fn test_1() {
        let x = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let f = -1.35;
        let u = vec![0.0; 6];
        let v = vec![2.1, 2.2, 2.3, 2.4, 2.5, 2.6];

        let (xmin, fmi, g, G, nfcsearch) = csearch(&x, f, &u, &v);

        assert_eq!(xmin, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(fmi, -1.35);
        assert_eq!(g, vec![334400.9877256005, 334400.9877775648, 334400.9872032684, -2.698149299274311, -16.096078870312834, -16.194566535234358]);

        let expected_G = vec![
            vec![-36815402316.54175, -9203850579.135637, -9203850579.13714, 668801.9766471038, 2668385.5012455117, 2674890.8722308264],
            vec![-9203850579.135637, -36815402316.542015, -9203850579.136158, 668801.976046821, 2668385.5011991933, 2674890.871860553],
            vec![-9203850579.13714, -9203850579.136158, -36815402316.55257, 668801.9825047514, 2668385.50150699, 2674890.8759595747],
            vec![668801.9766471038, 668801.976046821, 668801.9825047514, 64.81480786627557, -193.9018085234481, -194.37504360388937],
            vec![2668385.5012455117, 2668385.5011991933, 2668385.50150699, -193.9018085234481, 776.5949442178951, -775.5062594158968],
            vec![2674890.8722308264, 2674890.871860553, 2674890.8759595747, -194.37504360388937, -775.5062594158968, 777.5345833891469]
        ];
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 40);
    }

    #[test]
    fn test_2() {
        let x = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.5];
        let f = 0.1;
        let u = vec![-1.0; 6];
        let v = vec![1.0; 6];

        let (xmin, fmi, g, G, nfcsearch) = csearch(&x, f, &u, &v);

        assert_eq!(xmin, vec![0.190983, 0.1923299222415401, 0.5490777434437786, 0.0, 0.27333123819254623, 0.6794905377752729]);
        assert_eq!(fmi, -1.753664084201283);

        assert_eq!(g, vec![-1.5535771780193317, 1.6277375974962203, 1.5623144452891058, 1.1146776795991618, -2.557687551839474, 1.036597892964894]);

        let expected_G = vec![
            vec![0.8429779496902778, 0.038527425030238836, 0.08016691619464511, 0.41982712517257176, -1.0089458453227644, 6.442359154330083],
            vec![0.038527425030238836, 0.15853175642516312, -0.08074702867269061, -0.11732762041885096, 0.5192671643626103, -5.802272526736841],
            vec![0.08016691619464511, -0.08074702867269061, 0.7390274095254871, 0.19538436874345963, 1.6410915456775932, -8.617804831838098],
            vec![0.41982712517257176, -0.11732762041885096, 0.19538436874345963, 0.8648855174952936, 3.6425765888725383, -10.824780735536386],
            vec![-1.0089458453227644, 0.5192671643626103, 1.6410915456775932, 3.6425765888725383, 4.9506593399538215, 5.78145088543934],
            vec![6.442359154330083, -5.802272526736841, -8.617804831838098, -10.824780735536386, 5.78145088543934, 12.50714291630882]
        ];
        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 48);
    }

    #[test]
    fn test_3() {
        let x = vec![0.2, 0.4, 0.6, 0.8, 0.1, 0.3];
        let f = -1.5;
        let u = vec![0.0; 6];
        let v = vec![1.0; 6];

        let (xmin, fmi, g, G, nfcsearch) = csearch(&x, f, &u, &v);

        assert_eq!(xmin, vec![0.2, 0.4, 0.6, 0.8, 0.1, 0.3]);
        assert_eq!(fmi, -1.5);
        assert_eq!(g, vec![-15.364962350081559, -9.126032408929731, -66.72517400621334, 3.3001784827451157, -29.091529160361628, -9.393402972970774]);

        let expected_G = vec![
            vec![1617.141765629833, -239.6063896609004, 2413.7463170041506, 354.34884954596583, -1060.9751839203607, 542.3729072815652],
            vec![-239.6063896609004, 364.2341178708943, 1093.9148531450705, 161.50367146856198, -480.46362460362013, 249.26445612335812],
            vec![2413.7463170041506, 1093.9148531450705, 10473.469754959657, -1639.8005196209726, 4930.3173671152135, -2504.700458062508],
            vec![354.34884954596583, 161.50367146856198, -1639.8005196209726, 546.5174982652572, 723.5303733242112, -365.5301863497862],
            vec![-1060.9751839203607, -480.46362460362013, 4930.3173671152135, 723.5303733242112, 6606.16130771491, 1099.8765899392931],
            vec![542.3729072815652, 249.26445612335812, -2504.700458062508, -365.5301863497862, 1099.8765899392931, 734.8681806295466]
        ];

        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 47);
    }

    #[test]
    fn test_4() {
        let x = vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.5];
        let f = -0.2;
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5];
        let v = vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5];

        let (xmin, fmi, g, G, nfcsearch) = csearch(&x, f, &u, &v);

        assert_eq!(xmin, vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.5]);
        assert_eq!(fmi, -0.2);
        assert_eq!(g, vec![48335.34155263062, 48335.334904356554, -48335.32412470991, -48335.299023819374, 96670.7221889008, 96670.67130089863]);

        let expected_G = vec![
            vec![-5321411539.7519655, -1330352884.950399, 1330352884.986138, 1330352885.0233428, -5321411539.816137, -2660705769.886557],
            vec![-1330352884.950399, -5321411539.804666, 1330352885.018136, 1330352884.9994442, -5321411539.76542, -2660705770.0073743],
            vec![1330352884.986138, 1330352885.018136, -5321411539.906169, -1330352885.1244552, 5321411539.790804, 2660705770.003839],
            vec![1330352885.0233428, 1330352884.9994442, -1330352885.1244552, -5321411540.356289, 5321411539.578127, 2660705770.0071573],
            vec![-5321411539.816137, -5321411539.76542, 5321411539.790804, 5321411539.578127, -21285646159.31742, -10642823079.480274],
            vec![-2660705769.886557, -2660705770.0073743, 2660705770.003839, 2660705770.0071573, -10642823079.480274, -21285646159.405983]
        ];

        assert_eq!(G, expected_G);
        assert_eq!(nfcsearch, 27);
    }
}