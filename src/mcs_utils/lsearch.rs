use crate::gls::gls;
use crate::mcs_utils::{csearch::csearch, helper_funcs::*, neighbor::neighbor, triple::triple};
use crate::minq::minq;
use nalgebra::{SMatrix, SVector};

const EPS_POW_1_3: f64 = 0.000006055454452393343;

pub(crate) fn lsearch<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    mut f: f64,
    f0: f64,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    nf: usize,
    maxstep: usize,
    gamma: f64,
    hess: &SMatrix<f64, N, N>,
) -> (
    SVector<f64, N>, // xmin
    f64,             // fmi
    usize,           // ncall
) {

    // flag will always be true as nsweeps != 0 => no need
    // ncloc: === 1;
    // smaxls: === 15;
    // small: === 0.1;
    let mut ncall = 0_usize;

    let mut x0: SVector<f64, N> = SVector::zeros();
    clamp_SVector_mut(&mut x0, u, v);

    let (mut xmin, mut fmi, mut g, mut G, nfcsearch) = csearch(func, x, f, u, v);
    clamp_SVector_mut(&mut xmin, u, v);

    ncall += nfcsearch;

    let mut xold = xmin.clone();
    let mut fold = fmi;

    let mut d: SVector<f64, N> = (xmin - u) // Component-wise subtraction
        .zip_map(&(v - xmin), f64::min)  // Parallel minimum operation
        .zip_map(
            &((x - x0).abs().add_scalar(1.0).scale(0.25)),  // Compute 0.25 * (1.0 + |x - x0|)
            f64::min,
        );

    let (mut p, _, _) = minq(fmi, &g, &G, &-d, &d);

    let mut x = xmin + p;
    clamp_SVector_mut(&mut x, u, v);

    p = x - xmin;

    let mut alist;
    let mut flist;

    let mut r = if p.norm() != 0.0 {
        let f1 = func(&x);
        ncall += 1;

        alist = Vec::from([0.0, 1.0]);
        flist = Vec::from([fmi, f1]);

        let fpred = fmi + g.dot(&p) + 0.5 * (p.dot(&(G * p)));

        ncall += gls(func, &xmin, &p, &mut alist, &mut flist, u, v, 1, 0.1, 15);

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

        xmin += p.scale(alist[i]);
        clamp_SVector_mut(&mut xmin, u, v);

        if fold == fmi {
            0.0
        } else if fold == fpred {
            0.5
        } else {
            (fold - fmi) / (fold - fpred)
        }
    } else {
        0.0
    };
    let mut gain = f - fmi;

    let mut diag = false;

    // Compute ind
    let mut ind_len = (0..N).filter(|&i| u[i] < xmin[i] && xmin[i] < v[i]).count();

    let max_vector = xmin.abs().zip_map(&xold.abs(), f64::max);
    let mut b = g.abs().dot(&max_vector);
    let mut nstep = 0;

    while (ncall < nf)
        && (nstep < maxstep)
        && (diag || ind_len < N || (b >= gamma * (f0 - f) && gain > 0.0))  // || (nsweeps == 0 && fmi - gain <= freach) //nsweeps cannot be 0
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
            let mut ind1 = u.iter().zip(v)
                .enumerate()
                .filter_map(|(i, (&u_i, &v_i))| {
                    if xmin[i] == u_i || xmin[i] == v_i {
                        Some(i)
                    } else {
                        None
                    }
                }).collect::<Vec<usize>>();

            for ind1_mut_ref in ind1.iter_mut() {
                let i = *ind1_mut_ref;
                x = xmin.clone();
                x[i] = if xmin[i] == u[i] { x2[i] } else { x1[i] };
                let f1 = func(&x);
                ncall += 1;

                if f1 < fmi {
                    alist = Vec::from([0.0, x[i] - xmin[i]]);
                    flist = Vec::from([fmi, f1]);
                    p = SVector::<f64, N>::zeros();
                    p[i] = 1.0;
                    ncall += gls(func, &xmin, &p, &mut alist, &mut flist, u, v, 1, 0.1, 6);

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
                    *ind1_mut_ref = 0;
                }
            }

            clamp_SVector_mut(&mut xmin, u, v);

            // if not sum(ind1):
            if ind1.iter().max().map_or(true, |&max| max == 0) {
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

        // println!("before triple {xmin:?}\n{fmi}\n{x1:?}\n{x2:?}\n{hess:.15}, {G:.15}");
        diag = if (r - 1.0).abs() > 0.25 || gain == 0.0 || b < gamma * (f0 - f) {
            (xmin, fmi, g) = triple(func, &xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, true, &mut ncall);
            false
        } else {
            (xmin, fmi, g) = triple(func, &xmin, fmi, &mut x1, &mut x2, u, v, hess, &mut G, false, &mut ncall);
            true
        };

        xold = xmin.clone();
        fold = fmi;

        if r < 0.25 {
            d.scale_mut(0.5)
        } else if r > 0.75 {
            d.scale_mut(2.0)
        }

        (p, _, _) = minq(fmi, &g, &G,
                         &(u - xmin).zip_map(&-d, f64::max),
                         &(v - xmin).zip_map(&d, f64::min));

        let norm = p.norm();
        if norm.abs() == 0.0 && !diag && ind_len == N {
            break;
        }

        if norm != 0.0 {
            let fpred = fmi + g.dot(&p) + 0.5 * p.dot(&(G * p));
            x = xmin + p;
            let f1 = func(&x);
            ncall += 1;

            alist = Vec::from([0.0, 1.0]);
            flist = Vec::from([fmi, f1]);
            ncall += gls(func, &xmin, &p, &mut alist, &mut flist, u, v, 1, 0.1, 15);
            let (i, &fmi_new) = flist
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();
            fmi = fmi_new;

            xmin = xmin + alist[i] * p;
            clamp_SVector_mut(&mut xmin, u, v);

            // Compute gain and r
            gain = f - fmi;
            r = if fold == fmi {
                0.0
            } else if fold == fpred {
                0.5
            } else {
                (fold - fmi) / (fold - fpred)
            };
        } else {
            gain = f - fmi;
            if gain.abs() == 0.0 {
                r = 0.0;
            }
        }

        ind_len = (0..N).filter(|&inx| u[inx] < xmin[inx] && xmin[inx] < v[inx]).count();

        let max_vector = xmin.abs().zip_map(&xold.abs(), f64::max);
        b = g.abs().dot(&max_vector);
    }

    (xmin, fmi, ncall)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-5;


    #[test]
    fn test_1() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // fcn = "feval";  % do not change
        // data = "hm6";  % do not change
        // stop = [1000]; % do not change
        // x = [0.20601133; 0.20601133; 0.45913871; 0.15954294; 0.37887001; 0.62112999];
        // f = -2.728684905407648;
        // f0 = -0.9883412202327723;
        // u =[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; ];
        // v =[1.0; 1.0; 1.0; 1.0; 1.0; 1.0; ];
        // nf_left = 95;
        // local = 50;
        // gamma = 2e-6;
        // hess = ones(6,6);
        //
        // format long g;
        // [xmin,fmi,ncall,flag] = lsearch(fcn,data,x,f,f0,u,v,nf_left,stop,local,gamma,hess)

        let x = SVector::<f64, 6>::from_row_slice(&[0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999]);
        let f = -2.728684905407648;
        let f0 = -0.9883412202327723;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = 95;
        let local = 50;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, local, gamma, &hess);

        let expected_xmin = [0.201690168374353, 0.150010022862184, 0.476872657786236, 0.275332161789012, 0.311653070783851, 0.657299338893286];

        assert_eq!(ncall, 107);
        assert_relative_eq!(fmi, -3.32236801124393, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }

    #[test]
    fn test_2() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // fcn = "feval";  % do not change
        // data = "hm6";  % do not change
        // stop = [1000]; % do not change
        // x = [-0.2; 0.0; 0.4; 0.5; 1.0; 1.7];
        // f = -2.7;
        // f0 = -0.9;
        // u =[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; ];
        // v =[1.0; 1.0; 1.0; 1.0; 1.0; 1.0; ];
        // nf_left = 95;
        // local = 50;
        // gamma = 2e-6;
        // hess = ones(6,6);
        //
        // format long g;
        // [xmin,fmi,ncall,flag] = lsearch(fcn,data,x,f,f0,u,v,nf_left,stop,local,gamma,hess)

        let x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0.0, 0.4, 0.5, 1.0, 1.7]);
        let f = -2.7;
        let f0 = -0.9;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = 95;
        let local = 50;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, local, gamma, &hess);

        let expected_xmin = [0.0, 0.0, 0.4, 0.5, 1.0, 1.0];

        assert_eq!(ncall, 54);
        assert_relative_eq!(fmi, -2.7, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }


    #[test]
    fn test_3() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // fcn = "feval";  % do not change
        // data = "hm6";  % do not change
        // stop = [1000]; % do not change
        // x = [0.0; 1.8; -0.4; -0.5; -1.0; -1.7];
        // f =  2.1;
        // f0 = 0.8;
        // u =[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; ];
        // v =[1.0; 1.0; 1.0; 1.0; 1.0; 1.0; ];
        // nf_left = 90;
        // local = 100;
        // gamma = 2e-6;
        // hess = ones(6,6);
        //
        // format long g;
        // [xmin,fmi,ncall,flag] = lsearch(fcn,data,x,f,f0,u,v,nf_left,stop,local,gamma,hess)

        let x = SVector::<f64, 6>::from_row_slice(&[0.0, 1.8, -0.4, -0.5, -1.0, -1.7]);
        let f = 2.1;
        let f0 = 0.8;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = 90;
        let local = 100;
        let gamma = 2e-6;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, local, gamma, &hess);

        let expected_xmin = [0.404661968210107, 0.882463280195061, 0.846412514807172, 0.574020996351391, 0.138156075633913, 0.0384746993988985, ];

        assert_eq!(ncall, 101);
        assert_relative_eq!(fmi, -3.20316166578478, epsilon = TOLERANCE);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
    }


    #[test]
    fn test_0() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // fcn = "feval";  % do not change
        // data = "hm6";  % do not change
        // stop = [1000]; % do not change
        // x = [0.20601133; 0.20601133; 0.45913871; 0.15954294; 0.37887001; 0.62112999];
        // f = -2.728684905407648;
        // f0 = -0.9883412202327723;
        // u =[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; ];
        // v =[1.0; 1.0; 1.0; 1.0; 1.0; 1.0; ];
        // nf_left = 1;
        // local = 100;
        // gamma = 2e-8;
        // hess = ones(6,6);
        //
        // format long g;
        // [xmin,fmi,ncall,flag] = lsearch(fcn,data,x,f,f0,u,v,nf_left,stop,local,gamma,hess)

        let x = SVector::<f64, 6>::from_row_slice(&[0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.37887001, 0.62112999]);
        let f = -2.728684905407648;
        let f0 = -0.9883412202327723;
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nf_left = 1;
        let local = 100;
        let gamma = 2e-8;
        let hess: SMatrix<f64, 6, 6> = SMatrix::repeat(1.0);

        let (xmin, fmi, ncall) = lsearch(hm6, &x, f, f0, &u, &v, nf_left, local, gamma, &hess);

        let expected_xmin = [0.202906012669831, 0.142239843401988, 0.477577867457062, 0.270066254245811, 0.310418368070886, 0.659457951596462, ];

        assert_eq!(ncall, 56);
        assert_relative_eq!(xmin.as_slice(), expected_xmin.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fmi, -3.32061046639384, epsilon = TOLERANCE);
    }
}