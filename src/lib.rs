#![allow(non_snake_case)]
#![allow(unused_variables)]

use nalgebra::{Const, DimMin, Matrix2xX, SMatrix, SVector};

#[cfg(test)]
mod test_functions;

pub mod minq;
pub mod gls;
pub mod mcs_utils;
use mcs_utils::add_basket::add_basket;
use mcs_utils::basket_func::basket;
use mcs_utils::basket_func::basket1;
use mcs_utils::chk_locks::chkloc;
use mcs_utils::chk_locks::fbestloc;
use mcs_utils::exgain::exgain;
use mcs_utils::init_func::init;
use mcs_utils::init_func::initbox;
use mcs_utils::lsearch::lsearch;
use mcs_utils::splinit::splinit;
use mcs_utils::split::split;
use mcs_utils::splrnk::splrnk;
use mcs_utils::strtsw::strtsw;
use mcs_utils::updtrec::updtrec;
use mcs_utils::vertex::vertex;


#[derive(Debug, PartialEq)]
pub enum ExitFlagEnum {
    NormalShutdown,         // flag 1: true in Matlab
    StopNfExceeded,         // flag 2: ncall >= nf
    StopNsweepsExceeded,    // flag 3: nsweep - nsweepbest >= nsweeps
}

const INIT_VEC_CAPACITY: usize = 10_000;

// Default values for the input parameters
// smax = 5*N+10;
// nf = 50*N^2;
// stop = 3*N;
// local = 50;
// gamma = eps;
// hess = ones(N,N);
//
// ONLY SIMPLE INITIALIZATION IS SUPPORTED (IinitEnum::Zero / iinit=0 in Matlab)
pub fn mcs<const SMAX: usize, const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    nsweeps: usize, // should be > 1
    nf: usize,
    local: usize,
    gamma: f64,
    hess: &SMatrix<f64, N, N>,
) ->
    Result<(
        SVector<f64, N>,       // xbest
        f64,                   // fbest
        Vec<SVector<f64, N>>,  // xmin
        Vec<f64>,              // fmi
        usize,                 // ncall
        usize,                 // ncloc
        ExitFlagEnum,          // ExitFlag
    ), String>
where
    Const<N>: DimMin<Const<N>, Output=Const<N>>,
{
    // record: -1 from Matlab and has .len() +1 from Matlab
    // p: -1 from Matlab

    if v <= u { return Err(format!("Error MCS main: v should be > u:\nv = {v:?}\nu = {u:?}")); }
    if nsweeps < 2 { return Err(format!("Error MCS main: nsweeps should be >= 2\nnsweeps = {nsweeps}")); }

    let (mut xmin, mut fmi): (Vec<SVector<f64, N>>, Vec<f64>) = (Vec::with_capacity(INIT_VEC_CAPACITY), Vec::with_capacity(INIT_VEC_CAPACITY));

    let mut ncloc = 0_usize; // number of function evaluations used for local search

    // We handle vectors differently, so no step1 = 10000;    // step = 1000;    // dim = step1;

    let mut isplit = vec![0_isize; INIT_VEC_CAPACITY]; // can be any negative or positive integer number
    let mut level = vec![0_usize; INIT_VEC_CAPACITY]; // the same numeration as in matlab
    let mut ipar = vec![Some(0); INIT_VEC_CAPACITY]; // as in Matlab; can be >=0 or -1 (None)
    let mut f = Matrix2xX::<f64>::zeros(INIT_VEC_CAPACITY);
    let mut z = Matrix2xX::<f64>::zeros(INIT_VEC_CAPACITY);
    let mut ichild = vec![0_isize; INIT_VEC_CAPACITY]; // can be negative
    let mut nogain = vec![false; INIT_VEC_CAPACITY];

    // l is always 1; L is always 2 for IinitEnum::Zero (2 and 3 respectively for Matlab)
    let x0 = SMatrix::<f64, N, 3>::from_row_iterator((0..N).flat_map(|row_n| [u[row_n], (u[row_n] + v[row_n]) / 2.0, v[row_n]]));

    // Check whether there are infinities in the initialization list
    if x0.iter().any(|&x0_i| x0_i.is_infinite()) {
        return Err(format!("Error MCS main: infinities in initialization list.\nx0 = {x0:?}"));
    };

    // ncall=0 before this if IinitEnum::Zero
    let (mut f0, istar, mut ncall) = init::<N>(func, &x0);

    // definition of the base vertex of the original box (get middle element; l=2 in Matlab)
    let x: SVector<f64, N> = x0.column(1).into_owned();

    // definition of the opposite vertex v1 of the original box
    let v1: SVector<f64, N> = SVector::from_fn(|i, _| {
        if (x[i] - u[i]).abs() > (x[i] - v[i]).abs() {
            u[i] // go left; corener at the lower bound side (left of mid point)
        } else {
            v[i] // go right of mid point; corener of the upper bound side
        }
    });


    let mut nboxes = 1_usize; // as in Matlab; counter for boxes not in the 'shopping basket'
    // counter for boxes in the 'shopping basket'
    let (mut nbasket, mut nbasket0): (usize, usize) = (0, 0); // as in Matlab
    let mut nsweep = 0_usize; // sweep counter
    let mut nsweepbest = 0_usize; // number of sweep in which fbest was updated for the last time
    let mut m = N;
    let mut xloc: Vec<SVector<f64, N>> = Vec::with_capacity(INIT_VEC_CAPACITY); // (for local ~= 0) columns are the points that have been used as starting points for local search

    // p: -1 from Matlab
    let (p, mut xbest, mut fbest) = initbox(&x0, &f0, &istar, u, v, &mut isplit, &mut level, &mut ipar, &mut ichild, &mut f, &mut nboxes);
    let f0min = fbest;

    // stop(1) in matlab is nsweeps; always > 0 => flag won't change

    // record(i): -1 from Matlab; points to the best non-split box at level i; record.len() is +1 from Matlab and from what is needed due to generic_const_exprs
    let mut record = [None; SMAX]; // not SMAX-1 as it will be hard in terms of generic_const_exprs. will account for +1 len() later
    // s: same as in Matlab;
    let mut s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
    nsweep += 1;

    // VARIABLES USED LATER
    let mut e_min: f64;
    let mut xmin1: SVector<f64, N>;
    let (mut n0, mut x, mut y, mut x1, mut x2, mut f1, mut f2) = (SVector::<usize, N>::zeros(), SVector::<f64, N>::repeat(f64::NAN), SVector::<f64, N>::repeat(f64::NAN), SVector::<f64, N>::repeat(f64::NAN), SVector::<f64, N>::repeat(f64::NAN), SVector::<f64, N>::repeat(f64::NAN), SVector::<f64, N>::repeat(f64::NAN));
    // VARIABLES USED LATER


    while s < SMAX && ncall + 1 <= nf {
        // par: -1 from Matlab as record -1 from Matlab
        let par = record[s - 1].unwrap();
        vertex(par, u, v, &v1, &x0, &f0, &ipar, &isplit, &ichild, &z, &f, &mut n0, &mut x, &mut y, &mut x1, &mut x2, &mut f1, &mut f2);

        let splt = if s > 2 * N * (n0.min() + 1) {
            // Splitting index and splitting value z[1][par] for splitting by rank
            (isplit[par], z[(1, par)]) = splrnk(&n0, &p, &x, &y);
            true
        } else {
            // Box has already been marked as not eligible for splitting by expected gain
            if nogain[par] {
                false
            } else {
                // Splitting by expected gain
                (e_min, isplit[par], z[(1, par)]) = exgain(&n0, &x, &y, &x1, &x2, f[(0, par)], &f0, &f1, &f2);
                let fexp = f[(0, par)] + e_min;
                if fexp < fbest {
                    true
                } else {
                    nogain[par] = true;  // The box is marked as not eligible for splitting by expected gain
                    false             // The box is not split since we expect no improvement
                }
            }
        };

        if splt {
            debug_assert!(isplit[par] > 0);
            let i = isplit[par] as usize - 1; // i is used only for indexing => smart to -1 right away

            level[par] = 0;
            if z[(1, par)] == f64::INFINITY {
                m += 1;
                z[(1, par)] = m as f64;
                splinit::<N, SMAX>(func, i, s, par, &x0, u, v, &x, &mut xmin, &mut fmi,
                                   &mut ipar, &mut level, &mut ichild, &mut isplit, &mut nogain, &mut f, &mut z,
                                   &mut xbest, &mut fbest, &mut record, &mut nboxes, &mut nbasket, &mut nsweepbest,
                                   &mut nsweep, &mut f0);
                ncall += 2; // splinit does exactly 2 calls
            } else {
                z[(0, par)] = x[i];
                split::<N, SMAX>(func, i, s, par, &x, &mut y, z[(0, par)], z[(1, par)], &mut xmin, &mut fmi, &mut ipar, &mut level,
                                 &mut ichild, &mut isplit, &mut nogain, &mut f, &mut z, &mut xbest, &mut fbest, &mut record, &mut nboxes,
                                 &mut nbasket, &mut nsweepbest, &mut nsweep);
                ncall += 1; // split does exactly 1 call
            }
        } else {
            if s + 1 < SMAX {
                level[par] = s + 1;
                updtrec(par, s + 1, f.row(0), &mut record);
            } else {
                level[par] = 0;
                add_basket(&mut nbasket, &mut xmin, &mut fmi, &x, f[(0, par)]);
            }
        }

        // Update s to split boxes
        s += 1;
        while s < SMAX {
            if record[s - 1].is_none() {
                s += 1;
            } else {
                break;
            }
        }

        if s == SMAX { // if smax is reached, a new sweep is started
            if local != 0 {
                // f(1:3) in Matlab equals &f[0..3] in Rust. Note that 3 does not change
                fmi[nbasket0..nbasket].sort_by(|a, b| a.total_cmp(b));
                xmin[nbasket0..nbasket].sort_by(|a, b| func(a).total_cmp(&func(b))); // is actually very small range; easier to evaluate function one more time

                for j in nbasket0..nbasket {
                    x = xmin[j].clone();
                    let mut f1 = fmi[j];
                    if chkloc(&xloc, &x) {
                        xloc.push(x);
                        if basket(func, &mut x, &mut f1, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket0, nsweep, &mut nsweepbest, &mut ncall) {
                            let nf_left = match nf >= ncall {
                                true => { Some(nf - ncall) }
                                false => { None }
                            };
                            let (fmi1, nc);
                            (xmin1, fmi1, nc) = lsearch(func, &mut x, f1, f0min, u, v, nf_left, local, gamma, hess);
                            ncall = ncall + nc;
                            ncloc = ncloc + nc;
                            if fmi1 < fbest {
                                xbest = xmin1.clone();
                                fbest = fmi1;
                                nsweepbest = nsweep;
                            }
                            if basket1(func, &mut xmin1, fmi1, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket0, nsweep, &mut nsweepbest, &mut ncall) {
                                add_basket(&mut nbasket0, &mut xmin, &mut fmi, &xmin1, fmi1);
                                fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
                            }
                        }
                    }
                }
                nbasket = nbasket0;
            }
            record = [None; SMAX];
            s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
            if nsweep - nsweepbest >= nsweeps {
                return Ok((xbest, fbest, xmin, fmi, ncall, ncloc, ExitFlagEnum::StopNsweepsExceeded));
            }
            nsweep += 1;
        }
    }

    let exit_flag = if ncall >= nf { ExitFlagEnum::StopNfExceeded } else { ExitFlagEnum::NormalShutdown };
    Ok((xbest, fbest, xmin, fmi, ncall, ncloc, exit_flag))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    // 1. In cargo.toml uncomment [profile.release] debug = true
    // 2. Run Cmd as admin
    // 3. cd C:\Users\serge\RustroverProjects\Rust_MCS (or your path)
    // 4. cargo flamegraph --unit-test -- tests::test_for_flamegraph_0
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_for_flamegraph_0() {
        // Define your optimization bounds
        let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]); // lower bounds
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]); // upper bounds

        let nsweeps = 1_000;                // maximum number of sweeps
        let nf = 1_000_000;              // maximum number of function evaluations

        // Additional parameters
        const SMAX: usize = 1_000;                      // number of levels used
        let local = 100;                                 // local search level
        let gamma = 2e-10;                               // acceptable relative accuracy for local search
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);    // sparsity pattern of Hessian

        // Run the optimization
        let (_, _, _, _, _, _, _) = mcs::<SMAX, 6>(hm6, &u, &v, nsweeps, nf, local, gamma, &hess).unwrap();
    }

    #[test]
    fn test_0() {
        // Matlab equivalent test
        // DISCLAIMER: change split.m and split function to split_.m and split_ as new Matlab versions have reserved function name split
        //
        // clearvars;
        // clear global;
        //
        // fcn = "feval"; % do not change
        // data = "hm6"; % do not change
        // prt = 0; % do not change
        // u = [0; 0; 0; 0; 0; 0];
        // v = [1; 1; 1; 1; 1; 1];
        // smax = 20;
        // nf = 1000;
        // stop = [18];
        // iinit = 0; // Simple initialization list aka IinitEnum::Zero here
        // local = 50;
        // gamma= 2e-6;
        // hess = ones(6,6); % 6x6 matrix for hm6
        //
        // format long g;
        // [xbest,fbest,xmin,fmi,ncall,ncloc,flag]=mcs(fcn,data,u,v,prt,smax,nf,stop,iinit,local,gamma,hess)

        let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        const SMAX: usize = 20;
        let nf = 1000; // maximum number of function evaluations
        let nsweeps = 18;  // maximum number of sweeps
        // stop(1) - nsweeps
        let local = 50;
        let gamma = 2e-6;
        let hess = SMatrix::<f64, 6, 6>::repeat(1.);

        let (xbest, fbest, xmin, fmi, ncall, ncloc, ExitFlag) = mcs::<SMAX, 6>(hm6, &u, &v, nsweeps, nf, local, gamma, &hess).unwrap();

        // assert_relative_eq!(xbest.as_slice(), [0.201689511010837, 0.150010691921827, 0.476873974679078, 0.275332430524218, 0.311651616585802, 0.657300534081583].as_slice(), epsilon = TOLERANCE);
        // assert_relative_eq!(fbest, -3.32236801141551, epsilon = TOLERANCE);
        // assert_eq!(fmi, vec![-3.32236801141551, -0.988341220232772]);
        // assert_eq!(ncall, 262); // Due to unstable sorting in lslocal
        // assert_eq!(ncloc, 194);  // Due to unstable sorting in lslocal
        // assert_eq!(ExitFlag, ExitFlagEnum::StopNsweepsExceeded);
    }
}


#[cfg(test)]
mod tests_hm6 {
    use crate::test_functions::hm6;
    use nalgebra::SVector;

    #[test]
    fn test_hm6_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.29398867, 0.62112999]);
        let result = hm6(&x);
        (result, -2.872724123715199);
    }

    #[test]
    fn test_hm6_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let result = hm6(&x);
        (result, -1.4069105761385299);
    }


    #[test]
    fn test_hm6_3() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.1213, 0.2414, 0.1243, 0.345680344, 0.1237595, 0.1354856796]);
        let result = hm6(&x);
        (result, -0.16821471453083264);
    }

    #[test]
    fn test_hm6_4() {
        let x = SVector::<f64, 6>::from_row_slice(&[0., 0.9009009009009009, 0.5961844197086474, 0.40540540540540543, 0.03685503127875094, 0.6756756756756757]);
        let result = hm6(&x);
        (result, -0.12148933685954287);
    }

    #[test]
    fn test_hm6_5() {
        let x = SVector::<f64, 6>::from_row_slice(&[0., 0.6756756756756757, -11.029411764609979, -7.5, -0.6818180786573167, 1.3157894736842104]);
        let result = hm6(&x);
        (result, -9.600116638678902e-298);
    }
}