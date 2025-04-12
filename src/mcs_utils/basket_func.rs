use nalgebra::SVector;
use std::cmp::Ordering;

#[inline]
fn update_fbest_xbest_nsweepbest<const N: usize>(
    fbest: &mut f64, xbest: &mut SVector<f64, N>, nsweepbest: &mut usize,
    fbest_new: f64, xbest_new: &SVector<f64, N>, nsweepbest_new: usize,
) {
    *fbest = fbest_new;
    *xbest = *xbest_new;
    *nsweepbest = nsweepbest_new;
}


#[inline]
fn get_sorted_indices<const N: usize>(nbasket: usize, x: &SVector<f64, N>, xmin: &Vec<SVector<f64, N>>) -> Vec<usize> {
    let xmin_len = xmin.len();

    let mut indices: Vec<usize> = (0..nbasket).collect();
    indices.sort_unstable_by(|&i, &j| {
        match (i >= xmin_len, j >= xmin_len) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // Norm norm_squared() is used for performance as just norm() is self.norm_squared().simd_sqrt()
            (false, false) => (x - xmin[i]).norm_squared().total_cmp(&(x - xmin[j]).norm_squared()),
        }
    });
    indices
}

/// Checks whether a candidate for local search lies in the 'domain of
/// attraction' of a point in the 'shopping basket'
/// function [xbest,fbest,xmin,fmi,nbasket,loc,flag] = basket(fcn,data,x,
/// f,xmin,fmi,xbest,fbest,stop,nbasket)
/// Input:
/// fcn = 'fun' 	name of function fun(data,x), x an n-vector
/// data		data vector (or other data structure)
/// x(1:n)	candidate for the 'shopping basket'
/// f		its function value
/// xmin(1:n,:)  	columns are the base vertices of the boxes in the
///              	shopping basket
/// fmi          	fmi(j) is the function value at xmin(:,j)
/// xbest       	current best vertex
/// fbest    	current best function value
/// stop          stop(1) in ]0,1[:  relative error with which the known
/// 		 global minimum of a test function should be found
/// 		 stop(2) = fglob known global minimum of a test function
/// 		 stop(3) = safeguard parameter for absolutely small
/// 		 fglob
/// 		stop(1) >= 1: the program stops if the best function
/// 		 value has not been improved for stop(1) sweeps
/// 		stop(1) = 0: the user can specify a function value that
/// 		 should be reached
///                stop(2) = function value that is to be achieved
/// nbasket	current number of points in the 'shopping basket'
/// Output:
/// xbest       	current best vertex
/// fbest    	current best function value
/// xmin(1:n,:)	updated version of the points in the shopping basket
/// fmi		their function values
/// loc           = 0  candidate lies in the 'domain of attraction' of a
/// 		     point in the shopping basket
/// 		= 1  otherwise
/// ncall		number of function calls used in the program
pub fn basket<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &mut SVector<f64, N>,
    f: &mut f64,
    xmin: &Vec<SVector<f64, N>>,
    fmi: &Vec<f64>,
    xbest: &mut SVector<f64, N>,
    fbest: &mut f64,
    nbasket: usize,
    nsweep: usize,
    nsweepbest: &mut usize,
    ncall: &mut usize,
) ->
    bool   // loc
{
    // flag will always be true as nsweeps != 0 => no need
    // ncall is passed by &mut and incremented => no need for buffer var

    let mut loc = true;
    let (mut p, mut y1, mut y2): (SVector<f64, N>, SVector<f64, N>, SVector<f64, N>);

    if nbasket == 0 { return loc; }

    for i in get_sorted_indices(nbasket, x, xmin) {
        if fmi[i] <= *f {
            p = xmin[i] - *x;
            // y1 = x + p/3
            y1 = *x + (p.scale(1. / 3.));
            let f1 = func(&y1);
            *ncall += 1;
            if f1 <= *f {
                // Compute y2 = x + 2/3 * p
                y2 = *x + (p.scale(2. / 3.));
                let f2 = func(&y2);
                *ncall += 1;

                if f2 > f1.max(fmi[i]) {
                    if f1 < *f {
                        *x = y1;
                        *f = f1;
                        if *f < *fbest {
                            update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, *f, x, nsweep);
                        }
                    }
                } else if f1 < f2.min(fmi[i]) {
                    *f = f1;
                    *x = y1;
                    if *f < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, *f, x, nsweep);
                    }
                } else if f2 < f1.min(fmi[i]) {
                    *f = f2;
                    *x = y2;
                    if *f < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, *f, x, nsweep);
                    }
                } else {
                    loc = false;
                    break;
                }
            }
        }
    }
    loc
}


pub fn basket1<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    f: f64,
    xmin: &mut Vec<SVector<f64, N>>,
    fmi: &mut Vec<f64>,
    xbest: &mut SVector<f64, N>,
    fbest: &mut f64,
    nbasket: usize,
    nsweep: usize,
    nsweepbest: &mut usize,
    ncall: &mut usize,
) ->
    bool   // loc
{
    // flag will always be true as nsweeps != 0 => no need
    // ncall is passed by &mut and incremented => no need for buffer var

    let mut loc = true;

    if nbasket == 0 { return loc; }

    let (mut p, mut y1, mut y2): (SVector<f64, N>, SVector<f64, N>, SVector<f64, N>);
    for i in get_sorted_indices(nbasket, x, xmin) {
        // Compute p = xmin[i] - x
        p = xmin[i] - x;

        // Compute y1 = x + p / 3
        y1 = x + p.scale(1. / 3.);

        // Evaluate f1
        let f1 = func(&y1);
        *ncall += 1;

        if f1 <= fmi[i].max(f) {
            // Compute y2 = x + 2/3*p
            y2 = x + p.scale(2. / 3.);

            // Evaluate f2
            let f2 = func(&y2);
            *ncall += 1;

            if f2 <= f1.max(fmi[i]) {
                if f < f1.min(f2).min(fmi[i]) {
                    fmi[i] = f;
                    xmin[i] = *x;
                    if fmi[i] < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, fmi[i], &mut xmin[i], nsweep);
                    }
                    loc = false;
                    break;
                } else if f1 < f.min(f2).min(fmi[i]) {
                    fmi[i] = f1;
                    xmin[i] = y1;
                    if fmi[i] < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, fmi[i], &mut xmin[i], nsweep);
                    }
                    loc = false;
                    break;
                } else if f2 < f.min(f1).min(fmi[i]) {
                    fmi[i] = f2;
                    xmin[i] = y2;
                    if fmi[i] < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, fmi[i], &mut xmin[i], nsweep);
                    }
                    loc = false;
                    break;
                } else {
                    loc = false;
                    break;
                }
            }
        }
    }
    loc
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-16;


    #[test]
    fn test_0_immediate_return() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let mut f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = -3.3;
        let nbasket = 0;
        let nsweep = 15;
        let mut nsweepbest = 1;
        let mut ncall = 0;

        let loc = basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(x.as_slice(), [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest.as_slice(), [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
        ]);
        assert_eq!(fmi, vec![-3.3, -3.3, -3.1]);
        assert_eq!((loc, ncall, nsweepbest), (true, 0, 1));
    }

    #[test]
    fn test_2() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let mut f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        let mut fbest = -2.9;
        let nbasket = 2;
        let nsweep = 10;
        let mut nsweepbest = 1;
        let mut ncall = 0;

        let loc = basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(x.as_slice(), [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest.as_slice(), [0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        assert_eq!(fbest, -2.9);
        assert_eq!(xmin, vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3]);
        assert_eq!(fmi, vec![-2.9, -2.8, -2.7]);
        assert_eq!((loc, ncall, nsweepbest), (true, 1, 1));
    }


    #[test]
    fn test_4() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        let mut f = 0.01;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]),
        ];
        let mut fmi = vec![100.; 3];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]);
        let mut fbest = 100.;
        let nbasket = 2;
        let nsweep = 20;
        let mut nsweepbest = 20;
        let mut ncall = 0;

        let loc = basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(f, 0.01);
        assert_eq!(x, SVector::<f64, 6>::from_row_slice(&[-0.2, 0.0, -0.1, -10.15, -0.29, -0.62]));
        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]));
        assert_eq!(fbest, 100.);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ]);
        assert_eq!(fmi, vec![100.0, 100.0, 100.0]);
        assert_eq!((loc, ncall, nsweepbest), (true, 0, 20));
    }


    #[test]
    fn test_0_bask1_immediate_return() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = -3.3;
        let nbasket = 0;
        let nsweep = 15;
        let mut nsweepbest = 1;
        let mut ncall = 0;

        let loc = basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]));
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
        ]);
        assert_eq!(fmi, vec![-3.3, -3.3, -3.1]);
        assert_eq!((loc, ncall, nsweepbest), (true, 0, 1));
    }
    #[test]
    fn test_1_backet_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let f = -2.8727241412052247;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]),
        ];
        let mut fmi = vec![-3.3, -3.2, -3.1];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = -3.3;
        let nbasket = 3;
        let nsweep = 20;
        let mut nsweepbest = 2;
        let mut ncall = 0;

        let loc = basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]));
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]),
        ]);
        assert_eq!(fmi, vec![-3.3, -3.2, -3.1]);
        assert_eq!((loc, ncall, nsweepbest), (false, 4, 2));
    }
    #[test]
    fn test_2_backet_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        let mut fbest = -2.9;
        let nbasket = 2;
        let nsweep = 10;
        let mut nsweepbest = 1;
        let mut ncall = 0;

        let loc = basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.15, 0.1, 0.3, 0.2, 0.25, 0.55]));
        assert_eq!(fbest, -2.9);
        assert_eq!(xmin, vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3]);
        assert_eq!(fmi, vec![-2.9, -2.8, -2.7]);
        assert_eq!((loc, ncall, nsweepbest), (true, 2, 1));
    }
    #[test]
    fn test_3_basket1() {
        let x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        let f = 0.01;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ];
        let mut fmi = vec![-1., -2., -35.5];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]);
        let mut fbest = -2.3;
        let nbasket = 3;
        let nsweep = 20;
        let mut nsweepbest = 20;
        let mut ncall = 0;

        let loc = basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[-1., 0.15, 0.47, -0.27, 0.31, 0.65]));
        assert_eq!(fbest, -2.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ]);
        assert_eq!(fmi, vec![-1., -2., -35.5]);
        assert_eq!((loc, ncall, nsweepbest), (false, 2, 20));
    }

    #[test]
    fn test_4_basket1() {
        let x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        let f = 0.01;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ];
        let mut fmi = vec![100.; 3];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]);
        let mut fbest = -2.3;
        let nbasket = 3;
        let nsweep = 20;
        let mut nsweepbest = 20;
        let mut ncall = 0;

        let loc = basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]));
        assert_eq!(fbest, -2.3);
        assert_relative_eq!(xmin.as_slice(), &[
             SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
             SVector::<f64, 6>::from_row_slice(&[0.06666666666666665, 0.13333333333333333, 0.26, -3.283333333333333, 0.09666666666666662, 0.20666666666666667]),
             SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ].as_slice() , epsilon = TOLERANCE);
        assert_eq!(fmi, vec![100.0, -8.594107654969967e-08, 100.0]);
        assert_eq!((loc, ncall, nsweepbest), (false, 2, 20));
    }


    #[test]
    fn better_cover_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, -0.2, -0.4, 0.15, -0.29, 0.62]);
        let f = 10_000.0;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[-0.2, -0.2, -0.4, -0.15, -0.29, -0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
        ];
        let mut fmi = vec![-300.0, -300.0, -300.0, -300.0, -300.0, -300.0];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[1.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = 100_000.0;
        let nbasket = 4;
        let nsweep = 15;
        let mut nsweepbest = 1;
        let mut ncall = 0;

        let loc = basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[1.2, 0.15, 0.47, 0.27, 0.31, 0.65]));
        assert_eq!(fbest, 100000.0);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[-0.2, -0.2, -0.4, -0.15, -0.29, -0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62])
        ]);
        assert_eq!(fmi, vec![-300.0, -300.0, -300., -300., -300., -300.]);
        assert_eq!((loc, ncall, nsweepbest), (false, 2, 1));
    }
}
