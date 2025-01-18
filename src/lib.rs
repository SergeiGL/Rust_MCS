#![allow(non_snake_case)]

pub mod minq;
pub mod gls;
pub mod mcs_utils;
mod feval;

use feval::feval;
use mcs_utils::add_basket::add_basket;
use mcs_utils::basket_func::basket;
use mcs_utils::basket_func::basket1;
use mcs_utils::chk_locks::chkloc;
use mcs_utils::chk_locks::fbestloc;
use mcs_utils::exgain::exgain;
use mcs_utils::helper_funcs::update_flag;
use mcs_utils::init_func::init;
use mcs_utils::init_func::initbox;
use mcs_utils::lsearch::lsearch;
use mcs_utils::split_func::splinit;
use mcs_utils::split_func::split;
use mcs_utils::splrnk::splrnk;
use mcs_utils::strtsw::strtsw;
use mcs_utils::updtrec::updtrec;
use mcs_utils::vertex::vertex;

use nalgebra::{Const, DimMin, Matrix2xX, SMatrix, SVector};

#[derive(PartialEq)]
pub enum IinitEnum {
    Zero,   // Simple initialization list
    One,    // Not implemented
    Two,    // Not implemented
    Three,  // (WTF it is doing?)
}

pub enum ExitFlagEnum {
    NormalShutdown,         // flag 1: True
    StopNfExceeded,         // flag 2: ncall >= nf
    StopNsweepsExceeded,    // flag 3: (nsweep - nsweepbest) >= stop[0]
}
pub struct StopStruct {
    nsweeps: usize,
    freach: f64,
    nf: usize,
}

impl StopStruct {
    pub fn new<T>(data: T) -> StopStruct
    where
        T: std::ops::Index<usize> + AsRef<[<T as std::ops::Index<usize>>::Output]>,
        <T as std::ops::Index<usize>>::Output: Into<f64> + Copy,
    {
        let data_ref = data.as_ref();
        debug_assert!(data_ref.len() == 3);
        StopStruct {
            nsweeps: data[0].into() as usize,
            freach: data[1].into(),
            nf: data[2].into() as usize,
        }
    }
}


const INIT_VEC_CAPACITY: usize = 10_000;


pub fn mcs<const SMAX: usize, const N: usize>(
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    stop_struct: &StopStruct,
    iinit: &IinitEnum,
    local: usize,
    gamma: f64,
    hess: &SMatrix<f64, N, N>,
) -> (
    SVector<f64, N>,       // xbest
    f64,                   // fbest
    Vec<SVector<f64, N>>,  // xmin
    Vec<f64>,              // fmi
    usize,                 // ncall
    usize,                 // ncloc
    ExitFlagEnum,          // flag
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
        panic!("Error MCS main: infinities in initialization list");
    }

    let (mut f0, istar, mut ncall) = init(&x0);

    // Computing B[x,y] in this case y = v
    let x: SVector<f64, N> = x0.column(1).into_owned();

    // 2 opposite vertex
    let v1: SVector<f64, N> = SVector::from_fn(|ind, _| {
        if (x[ind] - u[ind]).abs() > (x[ind] - v[ind]).abs() {
            // corener at the lower bound side (left of mid point)
            u[ind] // go left
        } else {
            // corener of the upper bound side
            v[ind] // go right of mid point
        }
    });


    let mut isplit = vec![0_isize; INIT_VEC_CAPACITY]; // can be any negative or positive integer number
    let mut ichild = vec![0_isize; INIT_VEC_CAPACITY]; // can be negative
    let mut ipar = vec![Some(0_usize); INIT_VEC_CAPACITY]; // can be >=0 or -1 (None)
    let mut level = vec![0_usize; INIT_VEC_CAPACITY];
    let mut nogain = vec![0_usize; INIT_VEC_CAPACITY];

    let mut f = Matrix2xX::<f64>::zeros(INIT_VEC_CAPACITY);
    let mut z = Matrix2xX::<f64>::zeros(INIT_VEC_CAPACITY);

    let (mut ncloc, mut nboxes, mut m) = (0_usize, 0_usize, N);
    let (mut nbasket_option, mut nbasket0_option): (Option<usize>, Option<usize>) = (None, None); // -1
    let (mut nsweep, mut nsweepbest) = (1_usize, 0_usize);
    let mut xloc: Vec<SVector<f64, N>> = Vec::with_capacity(INIT_VEC_CAPACITY);
    let mut flag = true;

    // Initialize the boxes
    let (p, mut xbest, mut fbest) = initbox(&x0, &f0, &istar, u, v, &mut isplit, &mut level, &mut ipar, &mut ichild, &mut f, &mut nboxes);
    let f0min = fbest;

    update_flag(&mut flag, fbest, stop_struct);

    let (mut s, mut record) = strtsw::<SMAX>(&level, f.row(0), nboxes);

    let mut loc: bool;
    let mut nf_left: Option<usize>;
    let (mut fmi, mut xmin): (Vec<f64>, Vec<SVector<f64, N>>) = (Vec::with_capacity(INIT_VEC_CAPACITY), Vec::with_capacity(INIT_VEC_CAPACITY));
    let (mut e, mut xmin1, mut ncall_add): (SVector<f64, N>, SVector<f64, N>, usize);

    let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) =
        (SVector::<usize, N>::zeros(), SVector::<f64, N>::repeat(f64::INFINITY), SVector::<f64, N>::repeat(f64::INFINITY), SVector::<f64, N>::repeat(f64::INFINITY), SVector::<f64, N>::repeat(f64::INFINITY), SVector::<f64, N>::zeros(), SVector::<f64, N>::zeros());

    while s < SMAX && ncall + 1 <= stop_struct.nf {
        let par = record[s];
        vertex(par, u, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        let splt = if s > 2 * N * (n0.min() + 1) {
            // Splitting index and splitting value z[1][par] for splitting by rank
            (isplit[par], z[(1, par)]) = splrnk(&n0, &p, &x, &y);
            true
        } else {
            // Box has already been marked as not eligible for splitting by expected gain
            if nogain[par] != 0 {
                false
            } else {
                // Splitting by expected gain
                (e, isplit[par], z[(1, par)]) = exgain(&n0, &x, &y, &x1, &x2, f[(0, par)], &f0, &f1, &f2);
                let fexp = f[(0, par)] + e.min();
                if fexp < fbest {
                    true
                } else {
                    nogain[par] = 1;  // The box is marked as not eligible for splitting by expected gain
                    false             // The box is not split since we expect no improvement
                }
            }
        };

        if splt {
            debug_assert!(isplit[par] >= 0);
            let i = isplit[par] as usize;

            level[par] = 0;
            if z[(1, par)] == f64::INFINITY {
                m += 1;
                z[(1, par)] = m as f64;
                splinit(i, s, par, &x0, u, v, &mut x, &mut xmin, &mut fmi,
                        &mut ipar, &mut level, &mut ichild, &mut isplit, &mut nogain, &mut f, &mut z,
                        &mut xbest, &mut fbest, &mut record, &mut nboxes, &mut nbasket_option, &mut nsweepbest, &mut nsweep, &mut f0);
                ncall += 2; // splinit does exactly 2 calls
            } else {
                z[(0, par)] = x[i];
                split(i, s, par, &mut x, &mut y, z[(0, par)], z[(1, par)], &mut xmin, &mut fmi, &mut ipar, &mut level,
                      &mut ichild, &mut isplit, &mut nogain, &mut f, &mut z, &mut xbest, &mut fbest, &mut record, &mut nboxes,
                      &mut nbasket_option, &mut nsweepbest, &mut nsweep);
                ncall += 1; // split does exactly 1 call
            }
        } else {
            if s + 1 < SMAX {
                level[par] = s + 1;
                updtrec(par, s + 1, f.row(0), &mut record);
            } else {
                level[par] = 0;
                add_basket(&mut nbasket_option, &mut xmin, &mut fmi, &x, f[(0, par)]);
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

                fmi[nbasket0_plus_1..=nbasket].sort_by(|a, b| a.total_cmp(b));
                xmin[nbasket0_plus_1..=nbasket].sort_by(|a, b| feval(a).total_cmp(&feval(b))); // nbasket0_plus_1..=nbasket is actually very small range; easier to evaluate function one more time

                for j in nbasket0_plus_1..(nbasket + 1) {
                    x = xmin[j].clone();
                    let mut f1 = fmi[j];
                    if chkloc(&xloc, &x) {
                        xloc.push(x);
                        (loc, flag, ncall_add) = basket(&mut x, &mut f1, &mut xmin, &mut fmi, &mut xbest, &mut fbest, stop_struct, &nbasket0_option, nsweep, &mut nsweepbest);
                        ncall += ncall_add;
                        if !flag {
                            break;
                        }
                        if loc {
                            nf_left = match stop_struct.nf >= ncall {
                                true => { Some(stop_struct.nf - ncall) }
                                false => { None }
                            };
                            let (fmi1, nc);
                            (xmin1, fmi1, nc, flag) = lsearch(&mut x, f1, f0min, u, v, nf_left, stop_struct, local, gamma, hess);
                            ncall = ncall + nc;
                            ncloc = ncloc + nc;
                            if fmi1 < fbest {
                                xbest = xmin1.clone();
                                fbest = fmi1;
                                nsweepbest = nsweep;
                                if !flag {
                                    add_basket(&mut nbasket0_option, &mut xmin, &mut fmi, &xmin1, fmi1);
                                    break;
                                }
                                update_flag(&mut flag, fbest, stop_struct);
                                if !flag {
                                    return (xbest, fbest, xmin, fmi, ncall, ncloc, ExitFlagEnum::NormalShutdown);
                                }
                            }
                            (loc, flag, ncall_add) = basket1(&mut xmin1, fmi1, &mut xmin, &mut fmi, &mut xbest, &mut fbest, stop_struct,
                                                             &nbasket0_option, nsweep, &mut nsweepbest);
                            ncall += ncall_add;

                            if !flag { break; }
                            if loc {
                                add_basket(&mut nbasket0_option, &mut xmin, &mut fmi, &xmin1, fmi1);
                                fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0_option.unwrap());

                                if !flag { break; }
                            }
                        }
                    }
                }
                nbasket_option = nbasket0_option;
                if !flag {
                    break;
                }
            }
            (s, record) = strtsw::<SMAX>(&level, f.row(0), nboxes);
            if stop_struct.nsweeps > 1 && nsweep - nsweepbest >= stop_struct.nsweeps {
                return (xbest, fbest, xmin, fmi, ncall, ncloc, ExitFlagEnum::StopNsweepsExceeded);
            }
            nsweep += 1;
        }
    }

    let exit_flag = if ncall >= stop_struct.nf { ExitFlagEnum::StopNfExceeded } else { ExitFlagEnum::NormalShutdown };
    (xbest, fbest, xmin, fmi, ncall, ncloc, exit_flag)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-14;


    // cargo flamegraph --unit-test -- tests::test_for_flamegraph_0
    #[test]
    fn test_for_flamegraph_0() {
        const SMAX: usize = 1_000;
        let stop = StopStruct::new(vec![70., f64::NEG_INFINITY, 1_000_000.]);
        let iinit = IinitEnum::Zero;
        let local = 20;
        let gamma = 2e-7;
        let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);
    }

    #[test]
    fn test_0() {
        const SMAX: usize = 20;
        let stop = StopStruct::new(vec![18., f64::NEG_INFINITY, 1000.]);
        let iinit = IinitEnum::Zero;
        let local = 50;
        let gamma = 2e-6;
        let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 1031);
        assert_eq!(ncloc, 968);
        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.20168951100810487, 0.15001069181869445, 0.47687397421890854, 0.27533243049404754, 0.3116516166017299, 0.6573005340667769]));
        assert_eq!(fbest, -3.3223680114155156);
        assert_eq!(xmin, vec!(
            SVector::<f64, 6>::from_row_slice(&[0.20168951100810487, 0.15001069181869445, 0.47687397421890854, 0.27533243049404754, 0.3116516166017299, 0.6573005340667769]),
            SVector::<f64, 6>::from_row_slice(&[0.2016895108708528, 0.1500106919377769, 0.4768739750906546, 0.27533243052642653, 0.31165161660939505, 0.6573005340921458]),
            SVector::<f64, 6>::from_row_slice(&[0., 0., 0.5, 0., 0.5, 0.5]))
        );
        assert_eq!(fmi, vec![-3.3223680114155156, -3.322368011415515, -0.9883412202327723]);
    }

    #[test]
    fn test_1() {
        const SMAX: usize = 30;
        let stop = StopStruct::new(vec![100., f64::NEG_INFINITY, 1000.]);
        let iinit = IinitEnum::Zero;
        let local = 0;
        let gamma = 2e-9;
        let u = SVector::<f64, 6>::from_row_slice(&[-1., -2., -3., -4., -5., -6.]);
        let v = SVector::<f64, 6>::from_row_slice(&[1., 2., 3., 4., 5., 6.]);
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 1001);
        assert_eq!(ncloc, 0);
        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.2510430974668892, 0.0, 0.0, 0.24721359549995797, 0.0, 0.0]));
        assert_eq!(fbest, -0.014725947982821272);
    }


    #[test]
    fn test_2() {
        const SMAX: usize = 50;
        let stop = StopStruct::new(vec![11., f64::NEG_INFINITY, 100.]);
        let iinit = IinitEnum::Zero;
        let local = 0;
        let gamma = 2e-10;
        let u = SVector::<f64, 6>::from_row_slice(&[-3.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[3.; 6]);
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 100);
        assert_eq!(ncloc, 0);
        assert_relative_eq!(xbest.as_slice(), [0.5092880150001403, 0.5092880150001403, 0.5599033356467378, 0.5092880150001403, 0.5092880150001403, 0.].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -0.8165894352179089);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]),
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]),
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]),
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]),
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]),
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]),
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.]),
            SVector::<f64, 6>::from_row_slice(&[0.5092880150001403, 0.299449812778105, 0.5092880150001403, 0.5092880150001403, 0.5092880150001403, 0.])
        ]);
        assert_eq!(fmi, vec![-0.8154675055618968, -0.18369566296880474, -0.8154675055618968, -0.18369566296880474, -0.8154675055618968, -0.18369566296880474, -0.8154675055618968, -0.18369566296880474]);
    }

    #[test]
    fn test_3() {
        const SMAX: usize = 100;
        let stop = StopStruct::new(vec![11., f64::NEG_INFINITY, 100.]);
        let iinit = IinitEnum::Zero;
        let local = 50;
        let gamma = 2e-10;
        let u = SVector::<f64, 6>::from_row_slice(&[-3.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[3.; 6]);
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 246);
        assert_eq!(ncloc, 188);
        assert_relative_eq!(xbest.as_slice(), [0.3840375197582117  , 1.0086370591370057  , 0.83694910437547    ,       0.5292791936678723  , 0.10626783160256341 , 0.013378574118123088].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -2.720677123715828);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.3840375197582117, 1.0086370591370057, 0.83694910437547, 0.5292791936678723, 0.10626783160256341, 0.013378574118123088]),
            SVector::<f64, 6>::from_row_slice(&[0.40280505110773396, 1.0185760300002806, 0.8542187590882532, 0.5092880150001403, 0.12196569488839512, 0.03842934900132638]),
            SVector::<f64, 6>::from_row_slice(&[0.40280505110773396, 0.5032406113241115, 0.5092880150001403, 0.5092880150001403, 0.299449812778105, 0.])
        ]);
        assert_eq!(fmi, vec![-2.720677123715828, -2.648749360565575, -0.975122854985175]);
    }

    //  SVector::<f64, 6>::from_row_slice()&

    #[test]
    fn test_random_1() {
        const SMAX: usize = 101;
        let stop = StopStruct::new(vec![14., f64::NEG_INFINITY, 101.]);
        let iinit = IinitEnum::Zero;
        let local = 14;
        let gamma = 2e-12;
        let u = SVector::<f64, 6>::from_row_slice(&[4.727314049468511, 0.6729453410064368, 0.3696063529368182, 0.011224068637479823, 3.5672862570948705, 4.051833711629832]);
        let v = SVector::<f64, 6>::from_row_slice(&[7.727314049468511, 3.6729453410064368, 3.369606352936818, 3.01122406863748, 6.567286257094871, 7.051833711629832]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.07459779468772165, 0.18847789193008613, 0.8094721251488728, 0.29159437541655264, 0.6237880657270168, 0.9685287338392282,
            0.6766394151615276, 0.41742372614225665, 0.6263689467222898, 0.8996234339899951, 0.45459801042237435, 0.039404723770840366,
            0.22125824689814066, 0.7135066372005419, 0.021872674550278526, 0.8502631362975381, 0.21093823698900427, 0.8234176333796507,
            0.4774003947631743, 0.010827623037043765, 0.9301261005699737, 0.6373221578763562, 0.4477966704247218, 0.10011848143887458,
            0.5379347361044786, 0.20326717081553425, 0.24670504237607038, 0.06824635059355644, 0.07277606505750234, 0.22214215734209453,
            0.8954440435903359, 0.15526584218928685, 0.9137102875926272, 0.7949626220636087, 0.2212099433145306, 0.2424741572631993
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 205);
        assert_eq!(ncloc, 146);
        assert_relative_eq!(xbest.as_slice(), [4.727314049468511 , 0.6729453410064368, 0.8307181913359948,       0.3738546506877803, 3.5672862570948705, 4.051833711629832 ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -8.454999787537318e-100);
        assert_relative_eq!(xmin.as_slice(), vec![
            SVector::<f64, 6>::from_row_slice(&[4.727314049468511, 0.6729453410064368, 0.8307181913359948, 0.3738546506877803, 3.5672862570948705, 4.051833711629832]),
            SVector::<f64, 6>::from_row_slice(&[4.727314049468511, 0.6729453410064368, 0.8307232233520431, 0.37379694824296633, 3.5672862570948705, 4.051833711629832]),
            SVector::<f64, 6>::from_row_slice(&[4.727314049468511, 0.6729453410064368, 0.7813364364685025, 0.3746140498873046, 3.5672862570948705, 4.051833711629832])
        ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-8.454999787537318e-100, -8.454999779615405e-100, -8.111906516040064e-100]);
    }

    #[test]
    fn test_random_2() {
        const SMAX: usize = 101;
        let stop = StopStruct::new(vec![14., f64::NEG_INFINITY, 101.]);
        let iinit = IinitEnum::Zero;
        let local = 14;
        let gamma = 2e-12;
        let u = SVector::<f64, 6>::from_row_slice(&[-4.727314049468511, -0.6729453410064368, -0.3696063529368182, -0.011224068637479823, -3.5672862570948705, -4.051833711629832]);
        let v = SVector::<f64, 6>::from_row_slice(&[-3.727314049468511, 0.32705465899356323, 0.6303936470631818, 0.9887759313625202, -2.5672862570948705, -3.0518337116298317]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.07459779468772165, 0.18847789193008613, 0.8094721251488728, 0.29159437541655264, 0.6237880657270168, 0.9685287338392282,
            0.6766394151615276, 0.41742372614225665, 0.6263689467222898, 0.8996234339899951, 0.45459801042237435, 0.039404723770840366,
            0.22125824689814066, 0.7135066372005419, 0.021872674550278526, 0.8502631362975381, 0.21093823698900427, 0.8234176333796507,
            0.4774003947631743, 0.010827623037043765, 0.9301261005699737, 0.6373221578763562, 0.4477966704247218, 0.10011848143887458,
            0.5379347361044786, 0.20326717081553425, 0.24670504237607038, 0.06824635059355644, 0.07277606505750234, 0.22214215734209453,
            0.8954440435903359, 0.15526584218928685, 0.9137102875926272, 0.7949626220636087, 0.2212099433145306, 0.2424741572631993
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 206);
        assert_eq!(ncloc, 147);
        assert_relative_eq!(xbest.as_slice(), [-3.727314049468511   ,  0.1706764288762547  ,        0.5569723444858972  ,  0.012261741062775881,       -2.5672862570948705  , -3.0518337116298317  ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -6.085020790782224e-120);
        assert_relative_eq!(xmin.as_slice(), vec![
             SVector::<f64, 6>::from_row_slice(&[-3.727314049468511   ,  0.1706764288762547  ,         0.5569723444858972  ,  0.012261741062775881,        -2.5672862570948705  , -3.0518337116298317  ]),
             SVector::<f64, 6>::from_row_slice(&[-3.727314049468511   ,  0.16903108328269634 ,         0.5577318081166199  ,  0.012399931291049738,        -2.5672862570948705  , -3.0518337116298317  ]),
             SVector::<f64, 6>::from_row_slice(&[-3.727314049468511   ,  0.17032057929123579 ,         0.553447248647957   ,  0.009173805411918867,        -2.5672862570948705  , -3.0518337116298317  ])].as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-6.085020790782224e-120, -6.084965406311038e-120, -6.0835785981210616e-120]);
    }


    #[test]
    fn test_random_3() {
        const SMAX: usize = 101;
        let stop = StopStruct::new(vec![21., f64::NEG_INFINITY, 101.]);
        let iinit = IinitEnum::Zero;
        let local = 21;
        let gamma = 2e-12;
        let u = SVector::<f64, 6>::from_row_slice(&[-1.5963680834773746, -1.702096253354359, -0.3129868586761164, -0.42277410386119807, -1.2353044063604557, -1.3252710096724756]);
        let v = SVector::<f64, 6>::from_row_slice(&[-0.5963680834773746, -0.7020962533543591, 0.6870131413238836, 0.5772258961388019, -0.2353044063604557, -0.3252710096724756]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.7610862893603182, 0.15561139671229107, 0.0414571725711701, 0.768214621760824, 0.3623003529256186, 0.7237493292892264,
            0.9802764801645983, 0.9208355845976742, 0.7710372979193391, 0.7250798823142288, 0.9243176233916595, 0.5582646569502009,
            0.4151029180599519, 0.5954064369303929, 0.8140252048030944, 0.6038218205816934, 0.1537579535064416, 0.6949865078666053,
            0.19907195322465754, 0.7311388550777824, 0.7019720767106651, 0.652460347712853, 0.4159602166483142, 0.854563385975831,
            0.5436556123230359, 0.48981216802677363, 0.7613861056820711, 0.04525373095163232, 0.9855285117090369, 0.8925047290734425,
            0.07396847834524134, 0.9249142825555874, 0.39542437348302395, 0.7255942609674336, 0.3014258849016729, 0.5500039335110584
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 186);
        assert_eq!(ncloc, 127);
        assert_relative_eq!(xbest.as_slice(), [-0.5963680834773746 , -0.7020962533543591 ,  0.5396573533314668 ,        0.21225449313799843, -0.2353044063604557 , -0.3252710096724756 ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -1.5651035236609192e-07);
        assert_relative_eq!(xmin.as_slice(), vec![
             SVector::<f64, 6>::from_row_slice(&[-0.5963680834773746 , -0.7020962533543591 ,  0.5396573533314668 ,         0.21225449313799843, -0.2353044063604557 , -0.3252710096724756 ]),
             SVector::<f64, 6>::from_row_slice(&[-0.5963680834773746 , -0.7020962533543591 ,  0.5400041203184601 ,         0.21184439001334524, -0.2353044063604557 , -0.3252710096724756 ]),
             SVector::<f64, 6>::from_row_slice(&[-0.5963680834773746 , -0.7020962533543591 ,  0.547289700125349  ,         0.07702204416582413, -0.2353044063604557 , -0.3252710096724756 ])
        ].as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-1.5651035236609192e-07, -1.565100473659719e-07, -1.4330918719490945e-07]);
    }

    #[test]
    fn test_random_4() {
        const SMAX: usize = 101;
        let stop = StopStruct::new(vec![7., f64::NEG_INFINITY, 101.]);
        let iinit = IinitEnum::Zero;
        let local = 7;
        let gamma = 2e-7;
        let u = SVector::<f64, 6>::from_row_slice(&[-6.506834377244, -0.5547628574185793, -0.4896101151981129, -4.167584856725679, -6.389642504060847, -5.528716818248636]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.6136260223676221, 3.3116327823744762, 1.815553122672147, 0.06874148889830267, 0.7052383406994288, 0.93288192217477]);
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.8277209419820275, 0.35275501307855395, 0.252012633495165, 0.5667951361102919, 0.19630620226079598, 0.0648101272618129,
            0.5081006457816327, 0.2660878681097819, 0.09782770288876363, 0.43830363933100314, 0.4746456902322366, 0.4661411009402323,
            0.19980055789123086, 0.4986248326438728, 0.012620127489665345, 0.19089710870186494, 0.4362731501809838, 0.6063090941013247,
            0.7310040262066118, 0.4204623417897273, 0.8664287267092771, 0.9742278318360923, 0.6386093993614557, 0.27981042978028847,
            0.6800547697745852, 0.5742073425616279, 0.8821852581714857, 0.13408110711794174, 0.04935188705985705, 0.9987572981515097,
            0.6187202250393025, 0.1377423026724791, 0.8070825819627165, 0.2817037864244687, 0.5842187774516107, 0.09751501025007547
        ]);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6>(&u, &v, &stop, &iinit, local, gamma, &hess);

        assert_eq!(ncall, 222);
        assert_eq!(ncloc, 162);
        assert_relative_eq!(xbest.as_slice(), [0.17893289851322705, 0.15050297148806233, 0.5093505507161508 ,  0.06874148889830267, 0.3277860135659822 , 0.6461894522528354 ].as_slice(), epsilon = TOLERANCE);
        assert_eq!(fbest, -2.3417974980696994);
        assert_relative_eq!(xmin.as_slice(), vec![
            SVector::<f64, 6>::from_row_slice(&[0.17893289851322705, 0.15050297148806233, 0.5093505507161508 , 0.06874148889830267, 0.3277860135659822 , 0.6461894522528354 ]),
            SVector::<f64, 6>::from_row_slice(&[0.17891587003871542, 0.150482894697167  , 0.5093410639955795 , 0.06874148889830267, 0.3278075419276147 , 0.6461749608224651 ]),
            SVector::<f64, 6>::from_row_slice(&[0.17828933536282762, 0.16875439312523205, 0.5823529521265135 , 0.06874148889830267, 0.33828521303632625, 0.7069002116463341 ]),
        ].as_slice(),
        epsilon = TOLERANCE);
        assert_eq!(fmi, vec![-2.3417974980696994, -2.341784320434968, -2.2316087735356884]);
    }
}