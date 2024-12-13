#![allow(non_snake_case)]

pub mod minq;
pub mod gls;
mod feval;
mod init_func;
mod sign;
mod polint;
mod quadratic_func;
mod chk_flag;
mod strtsw;
mod updtf;
mod split_func;
mod chk_locks;
mod splrnk;
mod updtrec;
mod exgain;
mod basket_func;
mod vertex_func;
mod lsearch;
mod neighbor;
mod hessian;
mod triple;
mod csearch;

use crate::basket_func::basket;
use crate::basket_func::basket1;
use crate::chk_flag::chrelerr;
use crate::chk_flag::chvtr;
use crate::chk_locks::chkloc;
use crate::chk_locks::fbestloc;
use crate::exgain::exgain;
use crate::init_func::init;
use crate::init_func::initbox;
use crate::init_func::subint;
use crate::lsearch::lsearch;
use crate::split_func::splinit;
use crate::split_func::split;
use crate::splrnk::splrnk;
use crate::strtsw::strtsw;
use crate::updtrec::updtrec;
use crate::vertex_func::vertex;

use nalgebra::{Const, DimMin, Matrix2xX, Matrix3xX, SMatrix};

#[derive(PartialEq)]
pub enum IinitEnum {
    Zero, // Simple initialization list
    One,
    Two,
    Three, // TODO: add support of this variant (WTF it is doing?)
}


const STEP1: usize = 10_000_usize;
const STEP: usize = 40_000_usize;


pub fn mcs<const SMAX: usize, const N: usize>(
    u: [f64; N],
    v: [f64; N],
    nf: usize,
    stop: Vec<f64>,
    iinit: IinitEnum,
    local: usize,
    gamma: f64,
    hess: SMatrix<f64, N, N>,
) -> (
    [f64; N],       // xbest
    f64,            // fbest
    Vec<[f64; N]>,  // xmin
    Vec<f64>,       // fmi
    usize,          // ncall
    usize,          // ncloc
    bool,           // flag
) where
    Const<N>: DimMin<Const<N>, Output=Const<N>>,
{
    if u >= v {
        panic!("Error MCS main: v should be > u.\nu = {u:?}\nv = {v:?}");
    }


    let l = [1_usize; N]; // l indicates the mid-point
    let L = [2_usize; N]; // L indicates the end point

    // Definition of the initialization list
    let mut x0 = SMatrix::<f64, N, 3>::zeros();

    match iinit {
        IinitEnum::Zero => {
            for i in 0..N {
                x0[(i, 0)] = u[i];
                x0[(i, 1)] = (u[i] + v[i]) / 2.0;
                x0[(i, 2)] = v[i];
            }
        }
        IinitEnum::One => {
            for i in 0..N {
                if u[i] >= 0.0 {
                    x0[(i, 0)] = u[i];
                    (x0[(i, 1)], x0[(i, 2)]) = subint(u[i], v[i]);
                    x0[(i, 1)] = 0.5 * (x0[(i, 0)] + x0[(i, 2)]);
                } else if v[i] <= 0.0 {
                    x0[(i, 2)] = v[i];
                    (x0[(i, 1)], x0[(i, 0)]) = subint(v[i], u[i]);
                    x0[(i, 1)] = 0.5 * (x0[(i, 0)] + x0[(i, 2)]);
                } else {
                    x0[(i, 1)] = 0.0;
                    (_, x0[(i, 0)]) = subint(0.0, u[i]);
                    (_, x0[(i, 2)]) = subint(0.0, v[i]);
                }
            }
        }
        IinitEnum::Two => {
            for i in 0..N {
                x0[(i, 0)] = (u[i] * 5.0 + v[i]) / 6.0;
                x0[(i, 1)] = 0.5 * (u[i] + v[i]);
                x0[(i, 2)] = (u[i] + v[i] * 5.0) / 6.0;
            }
        }
        _ => {}
    }


    // Check whether there are infinities in the initialization list
    if x0.iter().any(|&value| value.is_infinite()) {
        panic!("Error- MCS main: infinities in initialization list");
    }

    let mut ncloc = 0_usize;
    let (f0, istar, mut ncall) =
        init(&x0, &l, &L);

    // Computing B[x,y] in this case y = v
    let mut x = [0.0; N];
    for i in 0..N {
        x[i] = x0[(i, l[i])];
    }
    // 2 opposite vertex
    let mut v1 = [0.0; N];
    for i in 0..N {
        if (x[i] - u[i]).abs() > (x[i] - v[i]).abs() {
            // corener at the lower bound side (left of mid point)
            v1[i] = u[i]; // go left
        } else {
            // corener of the upper bound side
            v1[i] = v[i]; // go right of mid point
        }
    }

    // Some parameters needed for initializing large arrays
    let mut dim = STEP1;

    // Initialization of some large arrays
    let mut isplit = vec![0_isize; STEP1]; // can be negative
    let mut level = vec![0_usize; STEP1];
    let mut ipar = vec![0_usize; STEP1];
    let mut ichild = vec![0_isize; STEP1]; // can be negative
    let mut nogain = vec![0_usize; STEP1];

    let mut f: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; STEP1]);
    let mut z: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; STEP1]);

    // Initialization of the record list, the counters nboxes, nbasket, m
    // and nloc, xloc and the output flag
    let mut nboxes = 0_usize;
    let mut nbasket_option: Option<usize> = None; // -1
    let mut nbasket0_option: Option<usize> = None; // -1
    let mut nsweepbest = 0_usize;
    let mut nsweep = 0_usize;
    let mut m = N;
    let mut nloc = 0_usize;
    let mut xloc: Vec<[f64; N]> = Vec::with_capacity(200);
    let mut flag = true;

    // Initialize the boxes
    let (p, mut xbest, mut fbest) =
        initbox(
            &x0,
            &f0,
            &l,
            &L,
            &istar,
            &u,
            &v,
            &mut isplit,
            &mut level,
            &mut ipar,
            &mut ichild,
            &mut f,
            &mut nboxes,
        );

    let f0min = fbest;
    let mut xmin: Vec<[f64; N]> = Vec::with_capacity(STEP1);
    let mut fmi: Vec<f64> = Vec::with_capacity(STEP1);

    // Check for convergence
    if stop[0] > 0.0 && stop[0] < 1.0 {
        flag = chrelerr(fbest, &stop);
    } else if stop[0] == 0.0 {
        flag = chvtr(fbest, stop[1]);
    }
    if !flag {
        return (xbest, fbest, xmin, fmi, ncall, ncloc, flag);
    }

    // The vector record is updated, and the minimal level s containing non-split boxes is computed
    let (mut s, mut record) = strtsw::<SMAX>(&level, &f[0], nboxes);
    nsweep += 1;

    let mut loc: bool;
    let mut f0: [Vec<f64>; 3] = [
        f0[0].to_vec(),
        f0[1].to_vec(),
        f0[2].to_vec(),
    ];

    while s < SMAX && ncall + 1 <= nf {
        let par = record[s];
        // Compute the base vertex x, the opposite vertex y, the 'neighboring'
        // vertices and their function values needed for quadratic
        // interpolation and the vector n0 indicating that the ith coordinate
        // has been split n0(i) times in the history of the box
        let (n0, mut x, mut y, x1, x2, f1, f2) =
            vertex(
                par,
                &u,
                &v,
                &v1,
                &x0,
                &Matrix3xX::<f64>::from_row_iterator(f0[0].len(), f0.iter().flatten().cloned()),
                &ipar,
                &isplit,
                &ichild,
                &Matrix2xX::<f64>::from_row_iterator(z[0].len(), z.iter().flatten().cloned()),
                &Matrix2xX::<f64>::from_row_iterator(f[0].len(), f.iter().flatten().cloned()),
                &l,
                &L,
            );

        // s 'large'
        let splt = if s > 2 * N * (n0.iter().min().unwrap() + 1) {
            // Splitting index and splitting value z[1][par] for splitting by rank
            (isplit[par], z[1][par]) = splrnk(&n0, &p, &x, &y);
            true
        } else {
            // Box has already been marked as not eligible for splitting by expected gain
            if nogain[par] != 0 {
                false
            } else {
                // Splitting by expected gain
                let e;
                (e, isplit[par], z[1][par]) =
                    exgain(
                        &n0,
                        &l,
                        &L,
                        &x,
                        &y,
                        &x1,
                        &x2,
                        f[0][par],
                        &f0,
                        &f1,
                        &f2,
                    );
                let fexp = f[0][par] + e.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                if fexp < fbest {
                    true
                } else {
                    nogain[par] = 1;  // The box is marked as not eligible for splitting by expected gain
                    false             // The box is not split since we expect no improvement
                }
            }
        };

        if splt {
            let i = isplit[par];
            if i < 0 { panic!() }
            let i = i as usize;

            level[par] = 0;
            if z[1][par] == f64::INFINITY {
                m += 1;
                z[1][par] = m as f64;
                let (f01, ncall_add);
                (f01, flag, ncall_add) =
                    splinit(
                        i,
                        s,
                        SMAX,
                        par,
                        &x0,
                        &u,
                        &v,
                        &mut x,
                        &L,
                        &l,
                        &mut xmin,
                        &mut fmi,
                        &mut ipar,
                        &mut level,
                        &mut ichild,
                        &mut f,
                        &mut xbest,
                        &mut fbest,
                        &stop,
                        &mut record,
                        &mut nboxes,
                        &mut nbasket_option,
                        &mut nsweepbest,
                        &mut nsweep,
                    );
                // f01 = f01.reshape(len(f01),1)
                // f0 = np.concatenate((f0,f01),axis=1)
                for (i, &el_to_push) in f01.iter().enumerate() {
                    f0[i].push(el_to_push);
                }
                ncall += ncall_add;
            } else {
                z[0][par] = x[i];
                let ncall1;
                (flag, ncall1) = split(
                    i,
                    s,
                    SMAX,
                    par,
                    &mut x,
                    &mut y,
                    &z.iter().map(|row| row[par]).collect::<Vec<f64>>(),
                    &mut xmin,
                    &mut fmi,
                    &mut ipar,
                    &mut level,
                    &mut ichild,
                    &mut f,
                    &mut xbest,
                    &mut fbest,
                    &stop,
                    &mut record,
                    &mut nboxes,
                    &mut nbasket_option,
                    &mut nsweepbest,
                    &mut nsweep,
                );
                ncall += ncall1;
            }
            if nboxes > dim {
                level.resize(level.len() + STEP, 0_usize);
                ipar.resize(ipar.len() + STEP, 0_usize);
                isplit.resize(isplit.len() + STEP, 0isize);
                ichild.resize(ichild.len() + STEP, 0_isize);
                nogain.resize(nogain.len() + STEP, 0_usize);
                let old_len = f[0].len();
                f[0].resize(old_len + STEP, 1.0);
                let old_len = f[1].len();
                f[1].resize(old_len + STEP, 1.0);
                let old_len = z[0].len();
                z[0].resize(old_len + STEP, 1.0);
                let old_len = z[1].len();
                z[1].resize(old_len + STEP, 1.0);
                dim = nboxes + STEP;
            }
            if !flag {
                break;
            }
        } else {
            if s + 1 < SMAX {
                level[par] = s + 1;
                updtrec(par, s + 1, &f[0], &mut record);
            } else {
                level[par] = 0;
                let nbasket = match nbasket_option {
                    Some(n) => {
                        nbasket_option = Some(n + 1);
                        n + 1
                    }
                    None => {
                        nbasket_option = Some(0);
                        0
                    }
                };
                if xmin.len() == nbasket {
                    xmin.push(x.clone());
                    fmi.push(f[0][par]);
                } else {
                    xmin[nbasket] = x.clone();
                    fmi[nbasket] = f[0][par];
                }
            }
        }

        // Update s to split boxes
        s += 1;
        while s < SMAX {
            if record[s] == 0 {
                s += 1;
            } else {
                break;
            }
        }

        // If smax is reached, a new sweep is started
        if s == SMAX {
            if local != 0 {
                let nbasket = nbasket_option.unwrap();
                let nbasket0_plus_1 = match nbasket0_option {
                    Some(n) => n + 1,
                    None => 0,
                };

                let mut fmi_temp = fmi[nbasket0_plus_1..=nbasket].to_vec();
                let mut xmin_temp = xmin[nbasket0_plus_1..=nbasket].to_vec();

                let mut indices: Vec<usize> = (0..fmi_temp.len()).collect();

                indices.sort_by(|&i, &j| fmi_temp[i].partial_cmp(&fmi_temp[j]).unwrap());

                fmi_temp.sort_by(|a, b| a.partial_cmp(b).unwrap());

                xmin_temp = indices.iter().map(|&i| xmin_temp[i]).collect();

                fmi[nbasket0_plus_1..(nbasket + 1)].copy_from_slice(&fmi_temp);
                xmin[nbasket0_plus_1..(nbasket + 1)].copy_from_slice(&xmin_temp);

                for j in (nbasket0_plus_1)..(nbasket + 1) {
                    let mut x = xmin[j].clone();
                    let mut f1 = fmi[j];
                    if chkloc(nloc, &xloc, &x) {
                        nloc += 1;
                        xloc.push(x.clone());
                        let ncall_add;
                        (loc, flag, ncall_add) =
                            basket(&mut x, &mut f1, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket0_option, nsweep, &mut nsweepbest);
                        ncall += ncall_add;
                        if !flag {
                            break;
                        }
                        if loc {
                            let (mut xmin1, fmi1, nc, flag_new) =
                                lsearch(&x, f1, f0min, &u, &v, nf as isize - ncall as isize, &stop, local, gamma,
                                        &hess);
                            ncall = ncall + nc;
                            ncloc = ncloc + nc;
                            flag = flag_new;
                            if fmi1 < fbest {
                                xbest = xmin1.clone();
                                fbest = fmi1;
                                nsweepbest = nsweep;
                                if !flag {
                                    let nbasket = match nbasket0_option {
                                        Some(n) => {
                                            nbasket0_option = Some(n + 1);
                                            n + 1
                                        }
                                        None => {
                                            nbasket0_option = Some(0);
                                            0
                                        }
                                    };
                                    nbasket_option = nbasket0_option;
                                    if xmin.len() == nbasket {
                                        xmin.push(xmin1.clone());
                                        fmi.push(fmi1);
                                    } else {
                                        xmin[nbasket] = xmin1.clone();
                                        fmi[nbasket] = fmi1;
                                    }
                                    break;
                                }

                                if stop[0] > 0. && stop[0] < 1. {
                                    flag = chrelerr(fbest, &stop);
                                } else {
                                    flag = chvtr(fbest, stop[1]);
                                }
                                if !flag {
                                    return (xbest, fbest, xmin, fmi, ncall, ncloc, flag);
                                }
                            }
                            let ncall_add;
                            (loc, flag, ncall_add) =
                                basket1(&mut xmin1, fmi1, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop,
                                        &nbasket0_option, nsweep, &mut nsweepbest);
                            ncall += ncall_add;

                            if !flag {
                                break;
                            }
                            if loc {
                                let nbasket0 = match nbasket0_option {
                                    Some(n) => {
                                        nbasket0_option = Some(n + 1);
                                        n + 1
                                    }
                                    None => {
                                        nbasket0_option = Some(0);
                                        0
                                    }
                                };

                                if xmin.len() == nbasket0 {
                                    xmin.push(xmin1.clone());
                                    fmi.push(fmi1);
                                } else {
                                    xmin[nbasket0] = xmin1.clone();
                                    fmi[nbasket0] = fmi1;
                                }
                                fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);

                                if !flag {
                                    nbasket_option = nbasket0_option;
                                    break;
                                }
                            }
                        }
                    }
                }
                nbasket_option = nbasket0_option;
                if !flag {
                    break;
                }
            }
            (s, record) = strtsw::<SMAX>(&level, &f[0], nboxes);
            if stop[0] > 1.0 {
                if (nsweep - nsweepbest) as f64 >= stop[0] {
                    flag = true;
                    println!("flag 3: (nsweep - nsweepbest) as f64 >= stop[0]");
                    return (xbest, fbest, xmin, fmi, ncall, ncloc, flag);
                }
            }
            nsweep += 1;
        }
    }

    if ncall >= nf {
        flag = true;
        println!("flag 2: ncall >= nf");
    }

    (xbest, fbest, xmin, fmi, ncall, ncloc, flag)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        const SMAX: usize = 20;
        let nf = 1000;
        let stop = vec![18., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 50;
        let gamma = 2e-6;
        let u = [0.; 6];
        let v = [1.0; 6];
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);


        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 1031);
        assert_eq!(ncloc, 968);
        assert_eq!(flag, true);
        assert_eq!(xbest, [0.20168951100810487, 0.15001069181869445, 0.47687397421890854, 0.27533243049404754, 0.3116516166017299, 0.6573005340667769]);
        assert_eq!(fbest, -3.3223680114155156);
        assert_eq!(xmin, vec!(
            [0.20168951100810487, 0.15001069181869445, 0.47687397421890854, 0.27533243049404754, 0.3116516166017299, 0.6573005340667769],
            [0.2016895108708528, 0.1500106919377769, 0.4768739750906546, 0.27533243052642653, 0.31165161660939505, 0.6573005340921458],
            [0., 0., 0.5, 0., 0.5, 0.5]));
        assert_eq!(fmi, vec![-3.3223680114155156, -3.322368011415515, -0.9883412202327723]);
    }

    #[test]
    fn test_1() {
        const SMAX: usize = 30;
        let nf = 1000;
        let stop = vec![100., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 0;
        let gamma = 2e-9;
        let u = [-1., -2., -3., -4., -5., -6.];
        let v = [1., 2., 3., 4., 5., 6.];
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 1001);
        assert_eq!(ncloc, 0);
        assert_eq!(flag, true);
        assert_eq!(xbest, [0.2510430974668892, 0.0, 0.0, 0.24721359549995797, 0.0, 0.0]);
        assert_eq!(fbest, -0.014725947982821272);
    }
}