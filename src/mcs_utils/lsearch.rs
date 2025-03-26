use crate::gls::gls;
use crate::mcs_utils::{csearch::csearch, helper_funcs::*, neighbor::neighbor, triple::triple};
use crate::minq::minq;
use crate::StopStruct;
use nalgebra::{SMatrix, SVector};

pub fn lsearch<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    mut f: f64,
    f0: f64,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    nf_left: Option<usize>,
    stop_struct: &StopStruct,
    maxstep: usize,
    gamma: f64,
    hess: &SMatrix<f64, N, N>,
) -> (
    SVector<f64, N>, // xmin
    f64,             // fmi
    usize,           // ncall
    bool,            // flag
) {
    let EPS_POW_1_3 = f64::EPSILON.powf(1.0 / 3.0);

    let (mut eps0, mut ncall, nloc, small, smaxls, mut flag) = (0.001, 0_usize, 1, 0.1, 15, true);

    let (mut xmin, mut fmi, mut g, mut G, nfcsearch) = csearch(func, x, f, u, v);
    clamp_SVector_mut(&mut xmin, u, v);

    ncall += nfcsearch;

    let mut xold = xmin.clone();
    let mut fold = fmi.clone();

    update_flag(&mut flag, fmi, stop_struct);

    if !flag { return (xmin, fmi, ncall, flag); }

    let mut x0: SVector<f64, N> = SVector::zeros();
    clamp_SVector_mut(&mut x0, u, v);

    let mut d: SVector<f64, N> = (xmin - u) // Component-wise subtraction
        .zip_map(&(v - xmin), f64::min)  // Parallel minimum operation
        .zip_map(
            &((x - x0).abs().add_scalar(1.0).scale(0.25)),  // Compute 0.25 * (1.0 + |x - x0|)
            f64::min,
        );

    let (mut p, _, _) = minq(fmi, &g, &mut G, &-d, &d);

    let mut x_new = xmin + p;
    clamp_SVector_mut(&mut x_new, u, v);

    p = x_new - xmin;

    let norm = p.norm();

    let mut alist = Vec::<f64>::with_capacity(10_000);
    let mut flist = Vec::<f64>::with_capacity(10_000);

    let (mut gain, mut r) = if norm != 0.0 {
        let f1 = func(&x_new);
        ncall += 1;

        alist.clear();
        alist.extend([0.0, 1.0]);

        flist.clear();
        flist.extend([fmi, f1]);

        let fpred = fmi + g.dot(&p) + 0.5 * (p.dot(&(G * p)));

        ncall += gls(func, &xmin, &p, &mut alist, &mut flist, u, v, nloc, small, smaxls);

        // Find the minimum
        let (mut i, &fminew) = flist
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        if fminew == fmi {
            i = alist.iter().position(|&a| a == 0.0).unwrap();
        } else {
            fmi = fminew;
        }

        // Update xmin
        for (x_ref, &p_scaled) in xmin.iter_mut().zip(&p.scale(alist[i])) {
            *x_ref += p_scaled;
        };

        clamp_SVector_mut(&mut xmin, u, v);

        update_flag(&mut flag, fmi, stop_struct);

        if !flag {
            return (xmin, fmi, ncall, flag);
        }

        (f - fmi, if fold == fmi { 0.0 } else if fold == fpred { 0.5 } else { (fold - fmi) / (fold - fpred) })
    } else {
        (f - fmi, 0.0)
    };

    let mut diag = false;

    // Compute ind
    let mut ind_len = (0..N)
        .filter(|&i| u[i] < xmin[i] && xmin[i] < v[i])
        .count();

    let dot_right: SVector<f64, N> = SVector::<f64, N>::from_fn(|i, _| xmin[i].abs().max(xold[i].abs()));

    let mut b = g.abs().dot(&dot_right);
    let mut nstep = 0;

    // Compute minusd and mind
    let mut minusd: SVector<f64, N>;
    let mut mind: SVector<f64, N>;

    while (nf_left.is_some() && ncall < nf_left.unwrap())
        && (nstep < maxstep)
        && (
        (diag || ind_len < N)
            || (stop_struct.nsweeps == 0 && fmi - gain <= stop_struct.freach)
            || (b >= gamma * (f0 - f) && gain > 0.0)
    )
    {
        nstep += 1;
        let mut delta = xmin.abs().scale(EPS_POW_1_3);
        for delta_i in delta.iter_mut() {
            if *delta_i == 0.0 {
                *delta_i = EPS_POW_1_3;
            }
        }

        let (mut x1, mut x2) = neighbor(&xmin, &delta, u, v);
        f = fmi;

        if ind_len < N && (b < gamma * (f0 - f) || gain == 0.0) {
            let mut ind1: Vec<Option<usize>> = u.iter()
                .zip(v.iter())
                .enumerate()
                .filter_map(|(i, (&u_i, &v_i))| {
                    if xmin[i] == u_i || xmin[i] == v_i {
                        Some(Some(i))
                    } else {
                        None
                    }
                })
                .collect();

            p = SVector::<f64, N>::zeros();
            for k in 0..ind1.len() {
                if let Some(i) = ind1[k] {
                    let mut x = xmin.clone();
                    x[i] = if xmin[i] == u[i] { x2[i] } else { x1[i] };
                    let f1 = func(&x);
                    ncall += 1;

                    if f1 < fmi {
                        alist.clear();
                        alist.extend([0.0, x[i], -xmin[i]]);

                        flist.clear();
                        flist.extend([fmi, f1]);

                        p[i] = 1.0;
                        ncall += gls(func, &xmin, &p, &mut alist, &mut flist, u, v, nloc, small, 6);
                        p[i] = 0.0;

                        let (mut j, &fminew) = flist
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.total_cmp(b))
                            .unwrap();

                        if fminew == fmi {
                            j = alist.iter().position(|&a| a == 0.0).unwrap();
                        } else {
                            fmi = fminew;
                        }

                        xmin[i] += alist[j];
                    } else {
                        ind1[k] = None;
                    }
                }
            }

            clamp_SVector_mut(&mut xmin, u, v);

            // if not sum(ind1):
            if ind1.iter().fold(ind1.len(), |acc, x| {
                match x {
                    Some(x_usize) => acc + x_usize,
                    None => acc - 1,
                }
            }) == ind1.len() {
                break;
            }

            delta = xmin.abs().scale(EPS_POW_1_3);

            for delta_i in delta.iter_mut() {
                if *delta_i == 0.0 {
                    *delta_i = EPS_POW_1_3;
                }
            }
            (x1, x2) = neighbor(&xmin, &delta, u, v);
        }
        let nftriple;
        // println!("before triple {xmin:?}\n{fmi}\n{x1:?}\n{x2:?}\n{hess:.15}, {G:.15}");
        diag = if (r - 1.0).abs() > 0.25 || gain == 0.0 || b < gamma * (f0 - f) {
            (xmin, fmi, g, nftriple) = triple(func, &xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, true);
            ncall += nftriple;
            false
        } else {
            (xmin, fmi, g, nftriple) = triple(func, &xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, false);
            ncall += nftriple;
            true
        };

        // println!("after triple {g:?}");

        xold = xmin.clone();
        fold = fmi;

        update_flag(&mut flag, fmi, stop_struct);

        if !flag {
            return (xmin, fmi, ncall, flag);
        }

        // Adjust d based on r
        if r < 0.25 {
            for d_i in &mut d {
                *d_i *= 0.5;
            }
        } else if r > 0.75 {
            for d_i in &mut d {
                *d_i *= 2.0;
            }
        }

        // Compute minusd and mind
        minusd = (u - xmin).zip_map(&d, |diff, d_val| diff.max(-d_val));
        mind = (v - xmin).zip_map(&d, |diff, d_val| diff.min(d_val));

        p = minq(fmi, &g, &mut G, &minusd, &mind).0;
        let norm = p.norm();

        if norm == 0.0 && !diag && ind_len == N {
            break;
        }

        if norm != 0.0 {
            let fpred = fmi + g.dot(&p) + 0.5 * p.dot(&(G * p));
            let f1 = func(&(xmin + p));
            ncall += 1;

            alist.clear();
            alist.extend([0.0, 1.0]);

            flist.clear();
            flist.extend([fmi, f1]);

            ncall += gls(func, &xmin, &p, &mut alist, &mut flist, u, v, nloc, small, smaxls);

            let (argmin, &fmi_new) = flist
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();
            fmi = fmi_new;

            xmin = xmin + alist[argmin] * p;
            clamp_SVector_mut(&mut xmin, u, v);

            update_flag(&mut flag, fmi, stop_struct);

            if !flag {
                return (xmin, fmi, ncall, flag);
            }

            // Compute gain and r
            gain = f - fmi;
            r = if fold == fmi {
                0.0
            } else if fold == fpred {
                0.5
            } else {
                (fold - fmi) / (fold - fpred)
            };

            if fmi < fold {
                eps0 = ((1.0 - 1.0 / r).abs() * eps0).min(0.001).max(f64::EPSILON);
            } else {
                eps0 = 0.001;
            }
        } else {
            gain = f - fmi;
            if gain == 0.0 {
                eps0 = 0.001;
                r = 0.0;
            }
        }

        ind_len = (0..N).filter(|&inx| u[inx] < xmin[inx] && xmin[inx] < v[inx]).count();

        b = g.abs().dot(&SVector::<f64, N>::from_fn(|i, _| xmin[i].abs().max(xold[i].abs())));
    }

    (xmin, fmi, ncall, flag)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-5;

    #[test]
    fn test_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999]);
        let f = -2.728684905407648;
        let f0 = -0.9883412202327723;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = None;
        let stop = StopStruct::new(vec![18.0, f64::NEG_INFINITY, 0.0]);
        let maxstep = 50;
        let gamma = f64::EPSILON;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.20290601266983127, 0.14223984340198792, 0.4775778674570614, 0.2700662542458104, 0.3104183680708858, 0.6594579515964624];

        assert_eq!(ncall, 56);
        assert_eq!(flag, true);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -3.320610466393837, epsilon = TOLERANCE);
    }

    #[test]
    fn test_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999]);
        let f = -2.728684905407648;
        let f0 = -0.9883412202327723;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = Some(95);
        let stop = StopStruct::new(vec![18.0, f64::NEG_INFINITY, 0.0]);
        let maxstep = 50;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.20169016858295652, 0.1500100239040133, 0.4768726575742668, 0.2753321620932197, 0.31165307086540384, 0.6572993388248786];

        assert_eq!(ncall, 107);
        assert_eq!(flag, true);
        assert_relative_eq!(fmi, -3.322368011243928, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }

    #[test]
    fn test_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0.0, 0.4, 0.5, 1.0, 1.7]);
        let f = -2.7;
        let f0 = -0.9;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = Some(95);
        let stop = StopStruct::new(vec![18.0, f64::NEG_INFINITY, 0.0]);
        let maxstep = 50;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.0, 0.0, 0.4, 0.5, 1.0, 1.0];

        assert_eq!(ncall, 98);
        assert_eq!(flag, true);
        assert_relative_eq!(fmi, -2.7, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }


    #[test]
    fn test_3() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 1.8, -0.4, -0.5, -1.0, -1.7]);
        let f = 2.1;
        let f0 = 0.8;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = Some(90);
        let stop = StopStruct::new(vec![20.0, f64::NEG_INFINITY, 0.0]);
        let maxstep = 100;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.40466193295801317, 0.8824636278527813, 0.8464090444977177, 0.5740213137874376, 0.13816034855820664, 0.0384741244365488];

        assert_eq!(ncall, 101);
        assert_eq!(flag, true);
        assert_relative_eq!(fmi, -3.2031616661534357, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }
}