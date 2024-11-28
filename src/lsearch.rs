use crate::chk_flag::*;
use crate::csearch::csearch;
use crate::feval::feval;
use crate::gls::gls;
use crate::minq::minq;
use crate::neighbor::neighbor;
use crate::triple::triple;
use nalgebra::{DMatrix, DVector};

pub fn lsearch(x: &[f64], mut f: f64, f0: f64, u: &[f64], v: &[f64],
               nf: isize, stop: &[f64], maxstep: usize, gamma: f64, hess: &Vec<Vec<f64>>,
) -> (
    Vec<f64>, //xmin
    f64, //fmi
    usize, //ncall
    bool //flag
) {
    let mut ncall = 0_usize;
    let n = x.len();
    let x0: Vec<f64> = u.iter()
        .zip(v.iter())
        .map(|(&u_i, &v_i)| u_i.clamp(0.0, v_i))  // Use clamp() to restrict the value
        .collect();


    let mut alist = vec![1.0, 2.0];
    let mut flist = vec![1.0, 2.0];

    let mut eps0 = 0.001;
    let nloc = 1;
    let small = 0.1;
    let smaxls = 15;

    let (mut xmin, mut fmi, mut g, mut G, mut nfcsearch) = csearch(x, f, u, v);
    let g_na = DVector::from_vec(g.clone());
    // println!("xmin={xmin:?}\ng={g:?}\nG={G:?}");

    xmin = xmin
        .iter()
        .zip(u)
        .zip(v)
        .map(|((&x_i, &u_i), &v_i)| x_i.clamp(u_i, v_i))
        .collect::<Vec<f64>>();

    ncall += nfcsearch;
    let mut xold = xmin.clone();
    let mut fold = fmi.clone();


    let eps: f64 = 2.220446049250313e-16;

    let mut flag = if stop[0] > 0.0 && stop[0] < 1.0 { chrelerr(fmi, stop) } else { chvtr(fmi, stop[1]) };

    if !flag {
        return (xmin, fmi, ncall, flag);
    }
    let mut d = (0..n)
        .map(|i| {
            let min1 = xmin[i] - u[i];
            let min2 = v[i] - xmin[i];
            let abs_diff = (x[i] - x0[i]).abs();
            let cond = 0.25 * (1.0 + abs_diff);

            f64::min(f64::min(min1, min2), cond)
        }).collect::<Vec<f64>>();


    println!("{fmi}\ng={g:?}\nG={G:?}\nd{d:?}");
    let (mut p, _, _) = minq(fmi, &g, &mut G, &d.iter().map(|&x| -x).collect::<Vec<f64>>(), &d);

    println!("p={p:?}");

    let mut x = (0..n).map(|i|
        {
            (xmin[i] + p[i]).min(v[i]).max(u[i])
        }
    ).collect::<Vec<f64>>();


    p = x.iter().zip(&xmin).map(|(&x_i, &xmin_i)| x_i - xmin_i).collect::<Vec<f64>>();

    let p_na = DVector::from_vec(p.clone());
    let norm = p_na.norm();

    let (mut gain, mut r) = if norm != 0.0 {
        let f1 = feval(&x);
        ncall += 1;
        alist = vec![0.0, 1.0];
        flist = vec![fmi, f1];

        //fpred = fmi + np.dot(g.T,p) + np.dot(0.5, np.dot(p.T,np.dot(G,p)))
        let G_na = DMatrix::from_vec(G.len(), G[0].len(), G.iter().flat_map(|row| row.iter().cloned()).collect());
        let fpred = fmi + (g_na.transpose() * &p_na)[(0, 0)] + 0.5 * (p_na.transpose() * &(G_na * &g_na))[(0, 0)];

        let nfls = gls(&xmin, &p, &mut alist, &mut flist, u, v, nloc, small, smaxls);
        ncall += nfls;

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

        let alist_na = DVector::from_vec(alist.clone());
        let p_na = DVector::from_vec(p.clone());

        let alist_dot_p = alist_na.dot(&p_na);
        xmin.iter_mut().for_each(|xmin_i| *xmin_i += alist_dot_p);

        for i in 0..n {
            xmin[i] = (xmin[i] + v[i]).min(v[i]).max(u[i]);
        }


        flag = if stop[0] > 0.0 && stop[0] < 1.0 { chrelerr(fmi, stop) } else { chvtr(fmi, stop[1]) };
        if !flag {
            return (xmin, fmi, ncall, flag);
        }

        (f - fmi, if fold == fmi { 0.0 } else if fold == fpred { 0.5 } else { (fold - fmi) / (fold - fpred) })
    } else {
        (f - fmi, 0.0)
    };

    let mut diag = false;
    let mut ind = vec![];
    for i in 0..n {
        if u[i] < xmin[i] && xmin[i] < v[i] {
            ind.push(i);
        }
    };


    //     b = np.dot(np.abs(g).T,[max(abs(xmin[i]),abs(xold[i])) for i in range(len(xmin))])
    let dot_right = xmin.iter().zip(&xold).map(|(&xmin_i, &xold_i)| {
        // max(abs(xmin[i]),abs(xold[i]))
        xmin_i.abs().max(xold_i.abs())
    }).collect::<Vec<f64>>();

    let dot_right_na = DVector::from_vec(dot_right.clone());
    let mut b = g_na.abs().transpose().dot(&dot_right_na);


    let mut nstep = 0;
    while ((ncall as isize) < nf) && (nstep < maxstep) &&
        ((diag || ind.len() < n) || (stop[0] == 0.0 && fmi - gain <= stop[1]) || (b >= gamma * (f0 - f) && gain > 0.0)) {
        nstep = nstep + 1;
        let mut delta = xmin.iter().map(|&xmin_i| xmin_i.abs() * eps.powf(1.0 / 3.0)).collect::<Vec<f64>>();

        let mut j = vec![];
        for (inx, &el) in delta.iter().enumerate() {
            if el == 0.0 { j.push(inx) }
        };

        if j.len() != 0 {
            for inx in j {
                delta[inx] = eps.powf(1.0 / 3.0)
            }
        }

        let (mut x1, mut x2) = neighbor(&xmin, &delta, u, v);

        f = fmi;

        if ind.len() < n && (b < gamma * (f0 - f) || gain == 0.0) {
            let mut ind1 = vec![];
            for i in 0..u.len() {
                if xmin[i] == u[i] || xmin[i] == v[i] {
                    ind1.push(Some(i));
                }
            };

            for k in 0..ind1.len() {
                let i = ind1[k].unwrap();
                x = xmin.clone();
                x[i] = if xmin[i] == u[i] { x2[i] } else { x1[i] };
                let f1 = feval(&x);
                ncall += 1;
                if f1 < fmi {
                    alist = vec![0.0, x[i], -xmin[i]];
                    flist = vec![fmi, f1];
                    p = vec![0.0; n];
                    p[i] = 1.0;
                    let nfls = gls(&xmin, &p, &mut alist, &mut flist, u, v, nloc, small, 6);
                    ncall += nfls;
                    let (mut j, &fminew) = flist
                        .iter()
                        .enumerate()
                        .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();

                    if fminew == fmi {
                        j = alist.iter().position(|&a| a == 0.0).unwrap();
                    } else {
                        fmi = fminew
                    }

                    xmin[i] += alist[j];
                } else {
                    ind1[k] = None;
                }
            }
            xmin = (0..xmin.len()).map(|inx| {
                xmin[inx].min(v[inx]).max(u[inx])
            }).collect::<Vec<f64>>();

            let sum_ind1 = ind1.iter().fold(0_isize, |acc, &new| {
                match new {
                    Some(k) => acc + k as isize,
                    None => acc - 1
                }
            });
            if sum_ind1 == 0 {
                break;
            }

            for inx in 0..delta.len() {
                delta[inx] = xmin[inx].abs() * eps.powf(1.0 / 3.0);
            }
            let mut j = vec![];
            for (inx, &el) in delta.iter().enumerate() {
                if el == 0.0 { j.push(inx) }
            };


            if j.len() != 0 {
                for inx in j {
                    delta[inx] = eps.powf(1.0 / 3.0)
                }
            }
            (x1, x2) = neighbor(&xmin, &delta, u, v)
        }

        if (r - 1.0).abs() > 0.25 || gain == 0.0 || b < gamma * (f0 - f) {
            //     Vec<f64>, //xtrip
            //     f64, //ftrip
            //     Vec<f64>, //g
            //     usize //nf
            let (xmin_new, fmi_new, g_new, nftriple) = triple(&mut xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, true);
            xmin = xmin_new;
            fmi = fmi_new;
            g = g_new;
            ncall += nftriple;
            diag = false;
        } else {
            let (xmin_new, fmi_new, g_new, nftriple) = triple(&mut xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, false);
            xmin = xmin_new;
            fmi = fmi_new;
            g = g_new;
            ncall += nftriple;
            diag = true;
        }

        xold = xmin.clone();
        fold = fmi;
        flag = if stop[0] > 0.0 && stop[0] < 1.0 { chrelerr(fmi, stop) } else { chvtr(fmi, stop[1]) };

        if !flag {
            return (xmin, fmi, ncall, flag);
        }

        if r < 0.25 {
            d.iter_mut().for_each(|x| *x *= 0.5);
        } else if r > 0.75 {
            d.iter_mut().for_each(|x| *x *= 2.0);
        }

        let minusd = (0..xmin.len()).map(|jnx| {
            // max(-d[jnx],u[jnx]-xmin[jnx])
            (u[jnx] - xmin[jnx]).max(-d[jnx])
        }).collect::<Vec<f64>>();

        let mind = (0..xmin.len()).map(|jnx| {
            // [min(d[jnx],v[jnx]-xmin[jnx])
            (v[jnx] - xmin[jnx]).min(d[jnx])
        }).collect::<Vec<f64>>();

        (p, _, _) = minq(fmi, &g, &mut G, &minusd, &mind);
        let p_na = DVector::from_vec(p.clone());
        let norm = p_na.norm();

        if norm == 0.0 && !diag && ind.len() == n {
            break;
        }

        if norm != 0.0 {
            let g_na = DVector::from_vec(g.clone());
            let G_na = DMatrix::from_vec(G.len(), G[0].len(), G.iter().flat_map(|row| row.iter().cloned()).collect());
            let fpred = fmi + (g_na.transpose() * &p_na)[(0, 0)] + 0.5 * (p_na.transpose() * &(G_na * &g_na))[(0, 0)];
            let f1 = feval(&xmin.iter().zip(&p).map(|(&xmin_i, &p_i)| xmin_i + p_i).collect::<Vec<f64>>());
            ncall += 1;
            alist = vec![0.0, 1.0];
            flist = vec![fmi, f1];
            let nfls = gls(&xmin, &p, &mut alist, &mut flist, u, v, nloc, small, smaxls);
            ncall += nfls;

            let (argmin, &fmi_new) = flist
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            fmi = fmi_new;


            // xmin = [xmin[jnx] + alist[argmin]*p[jnx] for jnx in range(len(xmin))]
            // xmin = np.asarray([max(u[jnx],min(xmin[jnx],v[jnx])) for jnx in range(len(xmin))])
            xmin = (0..xmin.len()).map(|jnx| {
                let new_xmin_i = xmin[jnx] + alist[argmin] * p[jnx];
                new_xmin_i.min(v[jnx]).max(u[jnx])
            }).collect::<Vec<f64>>();

            flag = if stop[0] > 0.0 && stop[0] < 1.0 { chrelerr(fmi, stop) } else { chvtr(fmi, stop[1]) };

            if !flag {
                return (xmin, fmi, ncall, flag);
            }

            gain = f - fmi;
            r = if fold == fmi { 0.0 } else if fold == fpred { 0.5 } else { (fold - fmi) / (fold - fpred) };

            if fmi < fold {
                eps0 = ((1.0 - 1.0 / r).abs() * eps0).min(0.001).max(eps);
            } else { eps0 = 0.001 }
        } else {
            gain = f - fmi;
            if gain == 0.0 {
                eps0 = 0.01;
                r = 0.0;
            }
        }

        let mut ind_new: Vec<usize> = vec![];
        for inx in 0..u.len() {
            if u[inx] < xmin[inx] && xmin[inx] < v[inx] {
                ind_new.push(inx);
            }
        };
        ind = ind_new;

        let dot_right: Vec<f64> = xmin.iter().zip(&xold).map(|(&xmin_i, &xold_i)| {
            xmin_i.abs().max(xold_i.abs())
        }).collect();

        let dot_right_na = DVector::from_vec(dot_right.clone());
        b = g_na.abs().transpose().dot(&dot_right_na);
    }

    (xmin, fmi, ncall, flag)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let x = vec![0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999];
        let f = -2.728684905407648;
        let f0 = -0.9883412202327723;
        let u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let nf = -95;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let maxstep = 50;
        let gamma = 2.220446049250313e-16;
        let hess = vec![vec![1.; 6]; 6];
        let (xmin, fmi, ncall, flag) = lsearch(&x, f, f0, &u, &v, nf, &stop, maxstep, gamma, &hess);

        assert_eq!(xmin, vec![0.20290601, 0.14223984, 0.47757787, 0.27006625, 0.31041837, 0.65945795]);
        assert_eq!(fmi, -3.320610466393837);
        assert_eq!(ncall, 56);
        assert_eq!(flag, true);
    }
}