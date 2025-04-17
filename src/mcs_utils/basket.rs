use crate::mcs_utils::helper_funcs::{get_sorted_indices, update_fbest_xbest_nsweepbest};
use nalgebra::SVector;


pub(crate) fn basket<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &mut SVector<f64, N>,
    f: &mut f64,
    xmin: &[SVector<f64, N>],
    fmi: &[f64],
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

    // i: -1 from Matlab
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_matlab_0() {
        // Matlab Equivalent test
        //
        // clearvars;
        // clear global;
        //
        // fcn = "feval"; % do not change
        // data = "hm6"; % do not change
        // path(path,'jones'); % do not change
        // stop = [100]; % do not change
        // x = [0.1, 0.11, 0.1, 0.9, 0.6, 0.3]'; % Note: column vector
        // f = 5.;
        // xmin = [[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; [0.2, 0.2, 0.4, 0.15, 0.29, 0.62];[0.2, 0.2, 0.4, 0.15, 0.29, 0.62];[0.2, 0.2, 0.4, 0.15, 0.29, 0.62];[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]]'; % Note: column vector
        // fmi = [-3.3, -10.3, -1.1, -3., -2., -2.];
        // xbest = [0.2, 0.15, 1.47, 0.27, 0.31, 0.65];
        // fbest = 3.3;
        // nbasket = 6;
        // global nsweepbest;
        // global nsweep;
        // nsweep = 15;
        // nsweepbest = 1;
        // ncall = 0;
        //
        // [xbest,fbest,xmin,fmi,x,f,loc,flag,ncall] = basket(fcn,data,x,f,xmin,fmi,xbest,fbest,stop,nbasket)
        // disp(nsweepbest)

        let mut x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.11, 0.1, 0.9, 0.6, 0.3]);
        let mut f = 5.;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 6];
        let mut fmi = vec![-3.3, -10.3, -1.1, -3., -2., -2.];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 1.47, 0.27, 0.31, 0.65]);
        let mut fbest = 3.3;
        let nbasket = 6;
        let nsweep = 15;
        let mut nsweepbest = 1;
        let mut ncall = 0;

        let loc = basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest.as_slice(), [0.2000, 0.1500, 1.4700, 0.2700, 0.3100, 0.6500]);
        assert_eq!(fbest, 3.3);
        assert_eq!(xmin, vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 6]);
        assert_eq!(fmi, vec![-3.3000, -10.3000, -1.1000, -3.0000, -2.0000, -2.0000]);
        assert_eq!(x.as_slice(), [0.1000, 0.1100, 0.1000, 0.9000, 0.6000, 0.3000]);
        assert_eq!(f, 5.);
        assert_eq!((loc, ncall, nsweepbest), (false, 2, 1));
    }

    #[test]
    fn test_matlab_1() {
        // Matlab Equivalent test
        // Test case for `if fmi[i] <= *f` (true), but `if f1 <= *f` (false).
        // We need hm6(&y1) > *f.
        //
        // clearvars; clear global;
        // fcn = "feval"; data = "hm6"; path(path,'jones'); stop = [100];
        // x = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]';
        // f = -1.0;
        // xmin = [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]';
        // fmi = [-2.0];
        // xbest = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]';
        // fbest = 10.0;
        // nbasket = 1;
        // global nsweepbest;
        // global nsweep;
        // nsweep = 5;
        // nsweepbest = 0;
        // ncall = 0;
        //
        // format long g;
        // [xbest,fbest,xmin,fmi,x,f,loc,flag,ncall] = basket(fcn,data,x,f,xmin,fmi,xbest,fbest,stop,nbasket)
        // disp(nsweepbest)

        let mut x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let mut f = -1.0;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.9, 0.9, 0.9, 0.9, 0.9, 0.9])];
        let mut fmi = vec![-2.0];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut fbest = 10.0;
        let nbasket = 1;
        let nsweep = 5;
        let mut nsweepbest = 0;
        let mut ncall = 0;

        let loc = basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest.as_slice(), [0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667]);
        assert_eq!(fbest, -1.1719579288098294);
        assert_eq!(xmin, vec![SVector::<f64, 6>::from_row_slice(&[0.9, 0.9, 0.9, 0.9, 0.9, 0.9])]);
        assert_eq!(fmi, vec![-2.0000]);
        assert_eq!(x.as_slice(), [0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667]);
        assert_eq!(f, -1.1719579288098294);
        assert_eq!((loc, ncall, nsweepbest), (true, 2, 5));
    }

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
}
