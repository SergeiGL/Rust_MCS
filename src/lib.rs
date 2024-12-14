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
mod add_basket;

use crate::add_basket::add_basket;
use crate::basket_func::basket;
use crate::basket_func::basket1;
use crate::chk_flag::update_flag;
use crate::chk_locks::chkloc;
use crate::chk_locks::fbestloc;
use crate::exgain::exgain;
use crate::init_func::init;
use crate::init_func::initbox;
use crate::lsearch::lsearch;
use crate::split_func::splinit;
use crate::split_func::split;
use crate::splrnk::splrnk;
use crate::strtsw::strtsw;
use crate::updtrec::updtrec;
use crate::vertex_func::vertex;
use nalgebra::{Const, DimMin, Matrix2xX, Matrix3x1, SMatrix};

#[derive(PartialEq)]
pub enum IinitEnum {
    Zero, // Simple initialization list
    One,
    Two,
    Three, // (WTF it is doing?)
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

    // l is always 1; L is always 2


    let mut x0 = SMatrix::<f64, N, 3>::zeros();
    match iinit {
        IinitEnum::Zero => {
            for i in 0..N {
                x0[(i, 0)] = u[i];
                x0[(i, 1)] = (u[i] + v[i]) / 2.0;
                x0[(i, 2)] = v[i];
            }
        }
        _ => { panic!("iinit != 0 is not implemented"); }
    }


    // Check whether there are infinities in the initialization list
    if x0.iter().any(|&value| value.is_infinite()) {
        panic!("Error- MCS main: infinities in initialization list");
    }

    let (mut f0, istar, mut ncall) = init(&x0);

    // Computing B[x,y] in this case y = v
    let mut x: [f64; N] = std::array::from_fn(|ind| x0[(ind, 1)]);

    // 2 opposite vertex
    let v1: [f64; N] = std::array::from_fn(|ind| {
        if (x[ind] - u[ind]).abs() > (x[ind] - v[ind]).abs() {
            // corener at the lower bound side (left of mid point)
            u[ind] // go left
        } else {
            // corener of the upper bound side
            v[ind] // go right of mid point
        }
    });


    let mut ncloc = 0_usize;
    let mut dim = STEP1;
    let mut isplit = vec![0_isize; STEP1]; // can be negative
    let mut level = vec![0_usize; STEP1];
    let mut ipar = vec![Some(0_usize); STEP1]; //can be >=0 or -1 (None)
    let mut ichild = vec![0_isize; STEP1]; // can be negative
    let mut nogain = vec![0_usize; STEP1];

    let mut f: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; STEP1]);
    let mut z: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; STEP1]);

    let mut nboxes = 0_usize;
    let mut nbasket_option: Option<usize> = None; // -1
    let mut nbasket0_option: Option<usize> = None; // -1
    let mut nsweepbest = 0_usize;
    let mut m = N;
    let mut nloc = 0_usize;
    let mut xloc: Vec<[f64; N]> = Vec::with_capacity(200);
    let mut flag = true;

    // Initialize the boxes
    let (p, mut xbest, mut fbest) = initbox(&x0, &f0, &istar, &u, &v, &mut isplit, &mut level, &mut ipar, &mut ichild, &mut f, &mut nboxes);
    let f0min = fbest;

    update_flag(&mut flag, &stop, fbest, 1);

    let (mut s, mut record) = strtsw::<SMAX>(&level, &f[0], nboxes);
    let mut nsweep = 1_usize;

    let mut loc: bool;
    let mut xmin: Vec<[f64; N]> = Vec::with_capacity(STEP1);
    let mut fmi: Vec<f64> = Vec::with_capacity(STEP1);

    while s < SMAX && ncall + 1 <= nf {
        let par = record[s];
        let (n0, mut x, mut y,
            x1, x2, f1, f2) = vertex(par, &u, &v, &v1, &x0,
                                     &f0,
                                     &ipar, &isplit, &ichild,
                                     &Matrix2xX::<f64>::from_row_iterator(z[0].len(), z.iter().flatten().cloned()), &Matrix2xX::<f64>::from_row_iterator(f[0].len(), f.iter().flatten().cloned()));

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
                (e, isplit[par], z[1][par]) = exgain(&n0, &x, &y, &x1, &x2, f[0][par], &f0, &f1, &f2);
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
            let i = match isplit[par] {
                n if n >= 0 => n as usize,
                _ => panic!("lib.rs: isplit[par] <0"),
            };

            level[par] = 0;
            if z[1][par] == f64::INFINITY {
                m += 1;
                z[1][par] = m as f64;
                let (f01, ncall_add);
                (f01, flag, ncall_add) = splinit(i, s, SMAX, par, &x0, &u, &v, &mut x, &mut xmin, &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f,
                                                 &mut xbest, &mut fbest, &stop, &mut record, &mut nboxes, &mut nbasket_option, &mut nsweepbest, &mut nsweep);

                // f01 = f01.reshape(len(f01),1)
                // f0 = np.concatenate((f0,f01),axis=1)
                f0.resize_horizontally_mut(f0.ncols() + 1, 0.0);
                f0.set_column(f0.ncols() - 1, &Matrix3x1::from(f01));
                ncall += ncall_add;
            } else {
                z[0][par] = x[i];
                let ncall1;
                (flag, ncall1) = split(i, s, SMAX, par, &mut x, &mut y, &z.iter().map(|row| row[par]).collect::<Vec<f64>>(),
                                       &mut xmin, &mut fmi, &mut ipar, &mut level, &mut ichild, &mut f, &mut xbest, &mut fbest, &stop,
                                       &mut record, &mut nboxes, &mut nbasket_option, &mut nsweepbest, &mut nsweep);
                ncall += ncall1;
            }
            if nboxes > dim {
                level.resize(level.len() + STEP, 0_usize);
                ipar.resize(ipar.len() + STEP, Some(0_usize));
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
                add_basket(&mut nbasket_option, &mut xmin, &mut fmi, x.clone(), f[0][par]);
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

                for j in nbasket0_plus_1..(nbasket + 1) {
                    x = xmin[j].clone();
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
                                    add_basket(&mut nbasket0_option, &mut xmin, &mut fmi, xmin1.clone(), fmi1);
                                    break;
                                }
                                update_flag(&mut flag, &stop, fbest, 1);
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
                                add_basket(&mut nbasket0_option, &mut xmin, &mut fmi, xmin1.clone(), fmi1);
                                fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0_option.unwrap());

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
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-14;

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


    #[test]
    fn test_2() {
        const SMAX: usize = 50;
        let nf = 100;
        let stop = vec![11., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 0;
        let gamma = 2e-10;
        let u = [-3.; 6];
        let v = [3.; 6];
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 100);
        assert_eq!(ncloc, 0);
        assert_eq!(flag, true);
        assert_relative_eq!(xbest.as_slice(), [0.5092880150001403, 0.5092880150001403, 0.5599033356467378, 0.5092880150001403, 0.5092880150001403, 0.].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -0.8165894352179089);
        assert_eq!(xmin, vec![
            [0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.],
            [0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.],
            [0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.],
            [0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.],
            [0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.],
            [0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.],
            [0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.],
            [0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]]
        );
        assert_eq!(fmi, vec![-0.8154675055618968, -0.18369566296880474, -0.8154675055618968, -0.18369566296880474, -0.8154675055618968, -0.18369566296880474, -0.8154675055618968, -0.18369566296880474]);
    }

    #[test]
    fn test_3() {
        const SMAX: usize = 100;
        let nf = 100;
        let stop = vec![11., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 50;
        let gamma = 2e-10;
        let u = [-3.; 6];
        let v = [3.; 6];
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 246);
        assert_eq!(ncloc, 188);
        assert_eq!(flag, true);
        assert_relative_eq!(xbest.as_slice(), [0.3840375197582117  , 1.0086370591370057  , 0.83694910437547    ,       0.5292791936678723  , 0.10626783160256341 , 0.013378574118123088].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -2.720677123715828);
        assert_eq!(xmin, vec![
            [0.3840375197582117, 1.0086370591370057, 0.83694910437547, 0.5292791936678723, 0.10626783160256341, 0.013378574118123088],
            [0.40280505110773396, 1.0185760300002806, 0.8542187590882532, 0.5092880150001403, 0.12196569488839512, 0.03842934900132638],
            [0.40280505110773396, 0.5032406113241115, 0.5092880150001403, 0.5092880150001403, 0.299449812778105, 0.]
        ]);
        assert_eq!(fmi, vec![-2.720677123715828, -2.648749360565575, -0.975122854985175]);
    }


    #[test]
    fn test_random_0() {
        const SMAX: usize = 100;
        let nf = 100;
        let stop = vec![512., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 10;
        let gamma = 2e-12;
        let u = [-0.8309704077049194, -2.4272947296816656, -0.48741498147393303, -0.0055484063449791066, -0.09406660821187174, -2.6969517198501585];
        let v = [0.16902959229508063, -1.4272947296816656, 0.512585018526067, 0.9944515936550209, 0.9059333917881283, -1.6969517198501585];
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 207);
        assert_eq!(ncloc, 149);
        assert_eq!(flag, true);
        assert_relative_eq!(xbest.as_slice(), [ 0.13121886125314583 , -1.4272947296816656  ,        0.512585018526067   ,  0.012536302119667286,        0.8263007034754487  , -1.6969517198501585  ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -3.266015278673272e-22);
        assert_relative_eq!(xmin.concat().as_slice(), vec![
            [0.13121886125314583, -1.4272947296816656, 0.512585018526067, 0.012536302119667286, 0.8263007034754487, -1.6969517198501585],
            [0.13125949450448435, -1.4272947296816656, 0.512585018526067, 0.012556045555902345, 0.8261449970542363, -1.6969517198501585],
            [0.13195110521652437, -1.4272947296816656, 0.512585018526067, 0.010005585926404648, 0.8262980110103224, -1.6969517198501585]].concat().as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-3.266015278673272e-22, -3.266014188552043e-22, -3.2659191367691953e-22]);
    }

    #[test]
    fn test_random_1() {
        const SMAX: usize = 101;
        let nf = 101;
        let stop = vec![14., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 14;
        let gamma = 2e-12;
        let u = [4.727314049468511, 0.6729453410064368, 0.3696063529368182, 0.011224068637479823, 3.5672862570948705, 4.051833711629832];
        let v = [7.727314049468511, 3.6729453410064368, 3.369606352936818, 3.01122406863748, 6.567286257094871, 7.051833711629832];
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.07459779468772165, 0.18847789193008613, 0.8094721251488728, 0.29159437541655264, 0.6237880657270168, 0.9685287338392282,
            0.6766394151615276, 0.41742372614225665, 0.6263689467222898, 0.8996234339899951, 0.45459801042237435, 0.039404723770840366,
            0.22125824689814066, 0.7135066372005419, 0.021872674550278526, 0.8502631362975381, 0.21093823698900427, 0.8234176333796507,
            0.4774003947631743, 0.010827623037043765, 0.9301261005699737, 0.6373221578763562, 0.4477966704247218, 0.10011848143887458,
            0.5379347361044786, 0.20326717081553425, 0.24670504237607038, 0.06824635059355644, 0.07277606505750234, 0.22214215734209453,
            0.8954440435903359, 0.15526584218928685, 0.9137102875926272, 0.7949626220636087, 0.2212099433145306, 0.2424741572631993
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 205);
        assert_eq!(ncloc, 146);
        assert_eq!(flag, true);
        assert_relative_eq!(xbest.as_slice(), [4.727314049468511 , 0.6729453410064368, 0.8307181913359948,       0.3738546506877803, 3.5672862570948705, 4.051833711629832 ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -8.454999787537318e-100);
        assert_relative_eq!(xmin.concat().as_slice(), vec![
            [4.727314049468511 , 0.6729453410064368, 0.8307181913359948,        0.3738546506877803, 3.5672862570948705, 4.051833711629832 ],
            [4.727314049468511  , 0.6729453410064368 , 0.8307232233520431 ,        0.37379694824296633, 3.5672862570948705 , 4.051833711629832  ],
            [4.727314049468511 , 0.6729453410064368, 0.7813364364685025,        0.3746140498873046, 3.5672862570948705, 4.051833711629832 ]].concat().as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-8.454999787537318e-100, -8.454999779615405e-100, -8.111906516040064e-100]);
    }

    #[test]
    fn test_random_2() {
        const SMAX: usize = 101;
        let nf = 101;
        let stop = vec![14., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 14;
        let gamma = 2e-12;
        let u = [-4.727314049468511, -0.6729453410064368, -0.3696063529368182, -0.011224068637479823, -3.5672862570948705, -4.051833711629832];
        let v = [-3.727314049468511, 0.32705465899356323, 0.6303936470631818, 0.9887759313625202, -2.5672862570948705, -3.0518337116298317];
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.07459779468772165, 0.18847789193008613, 0.8094721251488728, 0.29159437541655264, 0.6237880657270168, 0.9685287338392282,
            0.6766394151615276, 0.41742372614225665, 0.6263689467222898, 0.8996234339899951, 0.45459801042237435, 0.039404723770840366,
            0.22125824689814066, 0.7135066372005419, 0.021872674550278526, 0.8502631362975381, 0.21093823698900427, 0.8234176333796507,
            0.4774003947631743, 0.010827623037043765, 0.9301261005699737, 0.6373221578763562, 0.4477966704247218, 0.10011848143887458,
            0.5379347361044786, 0.20326717081553425, 0.24670504237607038, 0.06824635059355644, 0.07277606505750234, 0.22214215734209453,
            0.8954440435903359, 0.15526584218928685, 0.9137102875926272, 0.7949626220636087, 0.2212099433145306, 0.2424741572631993
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 206);
        assert_eq!(ncloc, 147);
        assert_eq!(flag, true);
        assert_relative_eq!(xbest.as_slice(), [-3.727314049468511   ,  0.1706764288762547  ,        0.5569723444858972  ,  0.012261741062775881,       -2.5672862570948705  , -3.0518337116298317  ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -6.085020790782224e-120);
        assert_relative_eq!(xmin.concat().as_slice(), vec![
            [-3.727314049468511   ,  0.1706764288762547  ,         0.5569723444858972  ,  0.012261741062775881,        -2.5672862570948705  , -3.0518337116298317  ],
            [-3.727314049468511   ,  0.16903108328269634 ,         0.5577318081166199  ,  0.012399931291049738,        -2.5672862570948705  , -3.0518337116298317  ],
            [-3.727314049468511   ,  0.17032057929123579 ,         0.553447248647957   ,  0.009173805411918867,        -2.5672862570948705  , -3.0518337116298317  ]].concat().as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-6.085020790782224e-120, -6.084965406311038e-120, -6.0835785981210616e-120]);
    }


    #[test]
    fn test_random_3() {
        const SMAX: usize = 101;
        let nf = 101;
        let stop = vec![21., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 21;
        let gamma = 2e-12;
        let u = [-1.5963680834773746, -1.702096253354359, -0.3129868586761164, -0.42277410386119807, -1.2353044063604557, -1.3252710096724756];
        let v = [-0.5963680834773746, -0.7020962533543591, 0.6870131413238836, 0.5772258961388019, -0.2353044063604557, -0.3252710096724756];
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.7610862893603182, 0.15561139671229107, 0.0414571725711701, 0.768214621760824, 0.3623003529256186, 0.7237493292892264,
            0.9802764801645983, 0.9208355845976742, 0.7710372979193391, 0.7250798823142288, 0.9243176233916595, 0.5582646569502009,
            0.4151029180599519, 0.5954064369303929, 0.8140252048030944, 0.6038218205816934, 0.1537579535064416, 0.6949865078666053,
            0.19907195322465754, 0.7311388550777824, 0.7019720767106651, 0.652460347712853, 0.4159602166483142, 0.854563385975831,
            0.5436556123230359, 0.48981216802677363, 0.7613861056820711, 0.04525373095163232, 0.9855285117090369, 0.8925047290734425,
            0.07396847834524134, 0.9249142825555874, 0.39542437348302395, 0.7255942609674336, 0.3014258849016729, 0.5500039335110584
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 186);
        assert_eq!(ncloc, 127);
        assert_eq!(flag, true);
        assert_relative_eq!(xbest.as_slice(), [-0.5963680834773746 , -0.7020962533543591 ,  0.5396573533314668 ,        0.21225449313799843, -0.2353044063604557 , -0.3252710096724756 ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -1.5651035236609192e-07);
        assert_relative_eq!(xmin.concat().as_slice(), vec![
            [-0.5963680834773746 , -0.7020962533543591 ,  0.5396573533314668 ,         0.21225449313799843, -0.2353044063604557 , -0.3252710096724756 ],
            [-0.5963680834773746 , -0.7020962533543591 ,  0.5400041203184601 ,         0.21184439001334524, -0.2353044063604557 , -0.3252710096724756 ],
            [-0.5963680834773746 , -0.7020962533543591 ,  0.547289700125349  ,         0.07702204416582413, -0.2353044063604557 , -0.3252710096724756 ]
        ].concat().as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-1.5651035236609192e-07, -1.565100473659719e-07, -1.4330918719490945e-07]);
    }

    #[test]
    fn test_random_4() {
        const SMAX: usize = 101;
        let nf = 101;
        let stop = vec![7., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 7;
        let gamma = 2e-7;
        let u = [-6.506834377244, -0.5547628574185793, -0.4896101151981129, -4.167584856725679, -6.389642504060847, -5.528716818248636];
        let v = [0.6136260223676221, 3.3116327823744762, 1.815553122672147, 0.06874148889830267, 0.7052383406994288, 0.93288192217477];
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.8277209419820275, 0.35275501307855395, 0.252012633495165, 0.5667951361102919, 0.19630620226079598, 0.0648101272618129,
            0.5081006457816327, 0.2660878681097819, 0.09782770288876363, 0.43830363933100314, 0.4746456902322366, 0.4661411009402323,
            0.19980055789123086, 0.4986248326438728, 0.012620127489665345, 0.19089710870186494, 0.4362731501809838, 0.6063090941013247,
            0.7310040262066118, 0.4204623417897273, 0.8664287267092771, 0.9742278318360923, 0.6386093993614557, 0.27981042978028847,
            0.6800547697745852, 0.5742073425616279, 0.8821852581714857, 0.13408110711794174, 0.04935188705985705, 0.9987572981515097,
            0.6187202250393025, 0.1377423026724791, 0.8070825819627165, 0.2817037864244687, 0.5842187774516107, 0.09751501025007547
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(u, v, nf, stop, iinit, local, gamma, hess);

        assert_eq!(ncall, 222);
        assert_eq!(ncloc, 162);
        assert_eq!(flag, true);
        assert_relative_eq!(xbest.as_slice(), [0.17893289851322705, 0.15050297148806233, 0.5093505507161508 ,       0.06874148889830267, 0.3277860135659822 , 0.6461894522528354 ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -2.3417974980696994);
        assert_relative_eq!(xmin.concat().as_slice(), vec![
            [0.17893289851322705, 0.15050297148806233, 0.5093505507161508 ,        0.06874148889830267, 0.3277860135659822 , 0.6461894522528354 ],
            [0.17891587003871542, 0.150482894697167  , 0.5093410639955795 ,        0.06874148889830267, 0.3278075419276147 , 0.6461749608224651 ],
            [0.17828933536282762, 0.16875439312523205, 0.5823529521265135 ,        0.06874148889830267, 0.33828521303632625, 0.7069002116463341 ]
        ].concat().as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-2.3417974980696994, -2.341784320434968, -2.2316087735356884]);
    }
}