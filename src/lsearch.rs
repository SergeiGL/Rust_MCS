use crate::chk_flag::*;
use crate::csearch::csearch;
use crate::feval::feval;
use crate::gls::gls;
use crate::minq::minq;
use crate::neighbor::neighbor;
use crate::triple::triple;
use nalgebra::{Const, DVector, DimMin, SMatrix, SVector};


fn clamp_xmin<const N: usize>(xmin: &mut [f64; N], u: &[f64; N], v: &[f64; N]) {
    xmin.iter_mut().zip(u.iter()).zip(v.iter()).for_each(|((x_i, &u_i), &v_i)| {
        *x_i = x_i.clamp(u_i, v_i);
    });
}

pub fn lsearch<const N: usize>(
    x: &[f64; N],
    mut f: f64,
    f0: f64,
    u: &[f64; N],
    v: &[f64; N],
    nf: isize,
    stop: &[f64],
    maxstep: usize,
    gamma: f64,
    hess: &SMatrix<f64, N, N>,
) -> (
    [f64; N], // xmin
    f64,      // fmi
    usize,    // ncall
    bool,     // flag
) where
    Const<N>: DimMin<Const<N>, Output=Const<N>>,
{
    let mut ncall = 0_usize;

    let x0: Vec<f64> = u.iter()
        .zip(v.iter())
        .map(|(&u_i, &v_i)| u_i.clamp(0.0, v_i))
        .collect();

    let mut eps0 = 0.001;
    let nloc = 1;
    let small = 0.1;
    let smaxls = 15;

    let (mut xmin, mut fmi, mut g, mut G, nfcsearch) = csearch(x, f, u, v);
    // println!("{g:?}");
    clamp_xmin(&mut xmin, u, v);

    ncall += nfcsearch;

    let mut xold = xmin.clone();
    let mut fold = fmi.clone();
    let eps: f64 = 2.220446049250313e-16;

    let mut flag = if stop[0] > 0.0 && stop[0] < 1.0 {
        chrelerr(fmi, stop)
    } else if stop[0] == 0.0 {
        chvtr(fmi, stop[1])
    } else {
        true // Default to true
    };

    if !flag {
        return (xmin, fmi, ncall, flag);
    }

    // Compute d
    let mut d: [f64; N] = std::array::from_fn(|i|
        f64::min(f64::min(xmin[i] - u[i], v[i] - xmin[i]), 0.25 * (1.0 + (x[i] - x0[i]).abs()))
    );

    let (mut p, _, _) = minq(
        fmi,
        &g,
        &G,
        &std::array::from_fn(|i| -d[i]),
        &std::array::from_fn(|i| d[i]),
    );


    let x_new = std::array::from_fn::<f64, N, _>(|i| (xmin[i] + p[i]).clamp(u[i], v[i]));

    p = SVector::<f64, N>::from_iterator(
        x_new.iter()
            .zip(&xmin)
            .map(|(&x_i, &xmin_i)| x_i - xmin_i)
    );
    let norm = p.norm();

    let mut alist;
    let mut flist;
    let (mut gain, mut r) = if norm != 0.0 {
        let f1 = feval(&x_new);
        ncall += 1;
        alist = vec![0., 1.];
        flist = vec![fmi, f1];

        let fpred = fmi + (g.transpose() * p)[(0, 0)] + (0.5 * (p.transpose() * (G * p)))[(0, 0)];

        ncall += gls(&xmin, &p, &mut alist, &mut flist, u, v, nloc, small, smaxls);

        // Find the minimum
        let (mut i, &fminew) = flist
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
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

        clamp_xmin(&mut xmin, u, v);

        // Update flag
        if stop[0] > 0.0 && stop[0] < 1.0 {
            flag = chrelerr(fmi, stop)
        } else if stop[0] == 0.0 {
            flag = chvtr(fmi, stop[1])
        }

        if !flag {
            return (xmin, fmi, ncall, flag);
        }

        (f - fmi, if fold == fmi { 0.0 } else if fold == fpred { 0.5 } else { (fold - fmi) / (fold - fpred) })
    } else {
        (f - fmi, 0.0)
    };

    let mut diag = false;

    // Compute ind
    let mut ind: Vec<usize> = (0..N)
        .filter(|&i| u[i] < xmin[i] && xmin[i] < v[i])
        .collect();

    let dot_right: SVector<f64, N> = SVector::from_fn(|i, _|
        xmin[i].abs().max(xold[i].abs())
    );
    let mut b = (g.transpose().abs() * dot_right)[(0, 0)];
    let mut nstep = 0;

    // println!("{xmin:?}"); // correct
    // Begin while loop
    while ((ncall as isize) < nf)
        && (nstep < maxstep)
        && (
        (diag || ind.len() < N)
            || (stop[0] == 0.0 && fmi - gain <= stop[1])
            || (b >= gamma * (f0 - f) && gain > 0.0)
    )
    {
        nstep += 1;
        let mut delta: [f64; N] = std::array::from_fn(|i| xmin[i].abs() * eps.powf(1.0 / 3.0));

        for delta_i in delta.iter_mut() {
            if *delta_i == 0.0 {
                *delta_i = eps.powf(1.0 / 3.0);
            }
        }

        let (mut x1, mut x2) = neighbor(&xmin, &delta, u, v);
        f = fmi;

        if ind.len() < N && (b < gamma * (f0 - f) || gain == 0.0) {
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

            for k in 0..ind1.len() {
                if let Some(i) = ind1[k] {
                    let mut x = xmin.clone();
                    x[i] = if xmin[i] == u[i] { x2[i] } else { x1[i] };
                    let f1 = feval(&x);
                    ncall += 1;

                    if f1 < fmi {
                        alist = vec![0.0, x[i], -xmin[i]];
                        flist = vec![fmi, f1];
                        p = SVector::<f64, N>::repeat(0.0);
                        p[i] = 1.0;
                        ncall += gls(&xmin, &p, &mut alist, &mut flist, u, v, nloc, small, 6);

                        let (mut j, &fminew) = flist
                            .iter()
                            .enumerate()
                            .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
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

            clamp_xmin(&mut xmin, u, v);

            // if not sum(ind1):
            if ind1.iter().fold(ind1.len(), |acc, x| {
                match x {
                    Some(x_usize) => acc + x_usize,
                    None => acc - 1,
                }
            }) == ind1.len() {
                break;
            }

            delta = std::array::from_fn(|i| xmin[i].abs() * eps.powf(1.0 / 3.0));

            for delta_i in delta.iter_mut() {
                if *delta_i == 0.0 {
                    *delta_i = eps.powf(1.0 / 3.0);
                }
            }
            (x1, x2) = neighbor(&xmin, &delta, u, v);
        }
        let nftriple;
        // println!("before triple {xmin:?}\n{fmi}\n{x1:?}\n{x2:?}\n{hess:.15}, {G:.15}");
        diag = if (r - 1.0).abs() > 0.25 || gain == 0.0 || b < gamma * (f0 - f) {
            (xmin, fmi, g, nftriple) = triple(&xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, true);
            ncall += nftriple;
            false
        } else {
            (xmin, fmi, g, nftriple) = triple(&xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, false);
            ncall += nftriple;
            true
        };

        // println!("after triple {g:?}");

        xold = xmin.clone();
        fold = fmi;

        // Update flag again
        if stop[0] > 0.0 && stop[0] < 1.0 {
            flag = chrelerr(fmi, stop)
        } else if stop[0] == 0.0 {
            flag = chvtr(fmi, stop[1])
        };

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
        let minusd = std::array::from_fn(|jnx| (u[jnx] - xmin[jnx]).max(-d[jnx]));
        let mind = std::array::from_fn(|jnx| (v[jnx] - xmin[jnx]).min(d[jnx]));

        // Recompute p using minq
        // println!("MINQ:\nfmi={fmi}\ng={g:?}\nG={G:?}\nminusd={minusd:?}\nmind={mind:?}");
        p = minq(fmi, &g, &G, &minusd, &mind).0;
        let norm = p.norm();
        // println!("{p}");

        if norm == 0.0 && !diag && ind.len() == N {
            break;
        }

        if norm != 0.0 {
            let fpred = fmi + (g.transpose() * p)[(0, 0)] + (0.5 * (p.transpose() * (G * p)))[(0, 0)];
            let x_pred = std::array::from_fn::<f64, N, _>(|i| xmin[i] + p[i]);
            let f1 = feval(&x_pred);
            ncall += 1;

            alist = vec![0.0, 1.0];
            flist = vec![fmi, f1];
            ncall += gls(&xmin, &p, &mut alist, &mut flist, u, v, nloc, small, smaxls);

            let (argmin, &fmi_new) = flist
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            fmi = fmi_new;

            xmin = std::array::from_fn(|i| {
                let new_xmin_i = xmin[i] + alist[argmin] * p[i];
                new_xmin_i.clamp(u[i], v[i])
            });

            if stop[0] > 0.0 && stop[0] < 1.0 {
                flag = chrelerr(fmi, stop)
            } else if stop[0] == 0.0 {
                flag = chvtr(fmi, stop[1])
            };

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
                eps0 = ((1.0 - 1.0 / r).abs() * eps0).min(0.001).max(eps);
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

        ind = (0..N)
            .filter(|&inx| u[inx] < xmin[inx] && xmin[inx] < v[inx])
            .collect();

        let dot_right = DVector::from_vec(
            xmin.iter()
                .zip(&xold)
                .map(|(&xmin_i, &xold_i)| xmin_i.abs().max(xold_i.abs()))
                .collect::<Vec<f64>>()
        );
        b = g.abs().dot(&dot_right);
    }

    (xmin, fmi, ncall, flag)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-5;

    #[test]
    fn test_0() {
        let x = [0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999];
        let f = -2.728684905407648;
        let f0 = -0.9883412202327723;
        let u = [0.0; 6];
        let v = [1.0; 6];
        let nf = -95;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let maxstep = 50;
        let gamma = 2.220446049250313e-16;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(&x, f, f0, &u, &v, nf, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.20290601266983127, 0.14223984340198792, 0.4775778674570614, 0.2700662542458104, 0.3104183680708858, 0.6594579515964624];

        assert_eq!(ncall, 56);
        assert_eq!(flag, true);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -3.320610466393837, epsilon = TOLERANCE);
    }

    #[test]
    fn test_1() {
        let x = [0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999];
        let f = -2.728684905407648;
        let f0 = -0.9883412202327723;
        let u = [0.0; 6];
        let v = [1.0; 6];
        let nf = 95;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let maxstep = 50;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(&x, f, f0, &u, &v, nf, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.20169016858295652, 0.1500100239040133, 0.4768726575742668, 0.2753321620932197, 0.31165307086540384, 0.6572993388248786];

        assert_eq!(ncall, 107);
        assert_eq!(flag, true);
        assert_relative_eq!(fmi, -3.322368011243928, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }

    #[test]
    fn test_2() {
        let x = [-0.2, 0.0, 0.4, 0.5, 1.0, 1.7];
        let f = -2.7;
        let f0 = -0.9;
        let u = [0.0; 6];
        let v = [1.0; 6];
        let nf = 95;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let maxstep = 50;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(&x, f, f0, &u, &v, nf, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.0, 0.0, 0.4, 0.5, 1.0, 1.0];

        assert_eq!(ncall, 98);
        assert_eq!(flag, true);
        assert_relative_eq!(fmi, -2.7, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }


    #[test]
    fn test_3() {
        let x = [0.0, 1.8, -0.4, -0.5, -1.0, -1.7];
        let f = 2.1;
        let f0 = 0.8;
        let u = [0.0; 6];
        let v = [1.0; 6];
        let nf = 90;
        let stop = vec![20.0, f64::NEG_INFINITY];
        let maxstep = 100;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall, flag) = lsearch(&x, f, f0, &u, &v, nf, &stop, maxstep, gamma, &hess);

        let expected_xmin = [0.40466193295801317, 0.8824636278527813, 0.8464090444977177, 0.5740213137874376, 0.13816034855820664, 0.0384741244365488];

        assert_eq!(ncall, 101);
        assert_eq!(flag, true);
        assert_relative_eq!(fmi, -3.2031616661534357, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }
}