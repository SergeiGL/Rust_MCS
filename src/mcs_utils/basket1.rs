use crate::mcs_utils::helper_funcs::{get_sorted_indices, update_fbest_xbest_nsweepbest};
use nalgebra::SVector;

pub(crate) fn basket1<const N: usize>(
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
        // p = xmin[i] - x
        p = xmin[i] - x;
        // y1 = x + p / 3
        y1 = x + p.scale(1. / 3.);
        let f1 = func(&y1);
        *ncall += 1;
        if f1 <= fmi[i].max(f) {
            // y2 = x + 2/3*p
            y2 = x + p.scale(2. / 3.);
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

    #[test]
    fn test_4_basket1() {
        // Matlab Equivalent test
        //
        // clearvars; clear global;
        // fcn = "feval"; data = "hm6"; path(path,'jones'); stop = [100];
        // x = [-0.2, 0., -0.1, -10.15, -0.29, -0.62]';
        // f = 0.01;
        // xmin = [[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]; [0.2, 0.2, 0.44, 0.15, 0.29, 0.62]; [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]]';
        // fmi = [100., 100., 100.];
        // xbest = [-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]';
        // fbest = -2.3;
        // nbasket = 3;
        // global nsweepbest;
        // global nsweep;
        // nsweep = 20;
        // nsweepbest = 21;
        // ncall = 0;
        //
        // format long g;
        // [xbest,fbest,xmin,fmi,loc,flag,ncall] = basket1(fcn,data,x,f,xmin,fmi,xbest,fbest,stop,nbasket)
        // disp(nsweepbest)

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
        let mut nsweepbest = 21;
        let mut ncall = 0;

        let loc = basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, nbasket, nsweep, &mut nsweepbest, &mut ncall);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]));
        assert_eq!(fbest, -2.3);
        assert_eq!(xmin.as_slice(), [
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.06666666666666665, 0.13333333333333333, 0.26, -3.283333333333333, 0.09666666666666662, 0.20666666666666667]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ]);
        assert_eq!(fmi, vec![100.0, -8.594107654969967e-08, 100.0]);
        assert_eq!((loc, ncall, nsweepbest), (false, 2, 21));
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