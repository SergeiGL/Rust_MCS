use crate::mcs_utils::helper_funcs::{get_sorted_indices, update_fbest_xbest_nsweepbest};
use nalgebra::SVector;


pub(crate) fn basket<const N: usize>(
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

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
