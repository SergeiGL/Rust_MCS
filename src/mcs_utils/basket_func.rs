use crate::mcs_utils::helper_funcs::update_flag;
use crate::StopStruct;
use nalgebra::SVector;
use std::cmp::Ordering;

fn update_fbest_xbest_nsweepbest<const N: usize>(fbest: &mut f64, xbest: &mut SVector<f64, N>, nsweepbest: &mut usize,
                                                 fbest_new: f64, xbest_new: &SVector<f64, N>, nsweepbest_new: usize) {
    *fbest = fbest_new;
    *xbest = *xbest_new;
    *nsweepbest = nsweepbest_new;
}


fn distance_squared<const N: usize>(a: &SVector<f64, N>, b: &SVector<f64, N>) -> f64 {
    let diff = a - b;
    diff.component_mul(&diff).sum() // we do not take sqrt as later there will be sort and
}

// Helper function
fn get_sorted_indices<const N: usize>(nbasket_plus_1: usize, x: &SVector<f64, N>, xmin: &Vec<SVector<f64, N>>) -> Vec<usize> {
    let xmin_len = xmin.len();

    let mut indices: Vec<usize> = (0..nbasket_plus_1).collect();
    indices.sort_unstable_by(|&i, &j| {
        match (i >= xmin_len, j >= xmin_len) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => distance_squared(&x, &xmin[i]).total_cmp(&distance_squared(&x, &xmin[j])),
        }
    });
    indices
}

pub fn basket<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &mut SVector<f64, N>,
    f: &mut f64,
    xmin: &Vec<SVector<f64, N>>,
    fmi: &Vec<f64>,
    xbest: &mut SVector<f64, N>,
    fbest: &mut f64,
    stop_struct: &StopStruct,
    nbasket: &Option<usize>,
    nsweep: usize,
    nsweepbest: &mut usize,
) -> (
    bool,   // loc
    bool,   // flag
    usize,  // ncall
) {
    let (mut loc, mut flag, mut ncall) = (true, true, 0_usize);
    let (mut p, mut y1, mut y2): (SVector<f64, N>, SVector<f64, N>, SVector<f64, N>);

    let nbasket_plus_1 = match nbasket {
        Some(0) => return (loc, flag, ncall),
        None => 0,
        Some(n) => n + 1,
    };

    for i in get_sorted_indices(nbasket_plus_1, x, xmin) {
        if fmi[i] <= *f {
            p = xmin[i] - *x;

            // y1 = x + p/3
            y1 = *x + (p.scale(1. / 3.));

            // Evaluate f1
            let f1 = func(&y1);
            ncall += 1;

            if f1 <= *f {
                // Compute y2 = x + 2/3 * p
                y2 = *x + (p.scale(2. / 3.));

                // Evaluate f2
                let f2 = func(&y2);
                ncall += 1;

                if f2 > f1.max(fmi[i]) {
                    if f1 < *f {
                        *x = y1;
                        *f = f1;
                        if *f < *fbest {
                            update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, *f, x, nsweep);
                            update_flag(&mut flag, *fbest, stop_struct);
                            if !flag { return (loc, flag, ncall); }
                        }
                    }
                } else {
                    if f1 < f2.min(fmi[i]) {
                        *f = f1;
                        *x = y1;
                        if *f < *fbest {
                            update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, *f, x, nsweep);
                            update_flag(&mut flag, *fbest, stop_struct);
                            if !flag { return (loc, flag, ncall); }
                        } else if f2 < f1.min(fmi[i]) {
                            *f = f2;
                            *x = y2;
                            if *f < *fbest {
                                update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, *f, x, nsweep);
                                update_flag(&mut flag, *fbest, stop_struct);
                                if !flag { return (loc, flag, ncall); }
                            }
                        } else {
                            loc = false;
                            break;
                        }
                    }
                }
            }
        }
    }
    (loc, flag, ncall)
}


pub fn basket1<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    f: f64,
    xmin: &mut Vec<SVector<f64, N>>,
    fmi: &mut Vec<f64>,
    xbest: &mut SVector<f64, N>,
    fbest: &mut f64,
    stop_struct: &StopStruct,
    nbasket: &Option<usize>,
    nsweep: usize,
    nsweepbest: &mut usize,
) -> (
    bool,   // loc
    bool,   // flag
    usize,  // ncall
) {
    let (mut loc, mut flag, mut ncall) = (true, true, 0_usize);


    let nbasket_plus_1 = match nbasket {
        Some(0) => return (loc, flag, ncall),
        None => 0,
        Some(n) => n + 1,
    };

    let (mut p, mut y1, mut y2): (SVector<f64, N>, SVector<f64, N>, SVector<f64, N>);
    for i in get_sorted_indices(nbasket_plus_1, x, xmin) {
        // Compute p = xmin[i] - x
        p = xmin[i] - x;

        // Compute y1 = x + p / 3
        y1 = x + p.scale(1. / 3.);

        // Evaluate f1
        let f1 = func(&y1);
        ncall += 1;

        if f1 <= fmi[i].max(f) {
            // Compute y2 = x + 2/3*p
            y2 = x + p.scale(2. / 3.);

            // Evaluate f2
            let f2 = func(&y2);
            ncall += 1;

            if f2 <= f1.max(fmi[i]) {
                if f < f1.min(f2).min(fmi[i]) {
                    fmi[i] = f;
                    xmin[i] = *x;
                    if fmi[i] < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, fmi[i], &mut xmin[i], nsweep);
                        update_flag(&mut flag, *fbest, stop_struct);
                        if !flag { return (loc, flag, ncall); }
                    }
                    loc = false;
                    break;
                } else if f1 < f.min(f2).min(fmi[i]) {
                    fmi[i] = f1;
                    xmin[i] = y1;
                    if fmi[i] < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, fmi[i], &mut xmin[i], nsweep);
                        update_flag(&mut flag, *fbest, stop_struct);
                        if !flag { return (loc, flag, ncall); }
                    }
                    loc = false;
                    break;
                } else if f2 < f.min(f1).min(fmi[i]) {
                    fmi[i] = f2;
                    xmin[i] = y2;
                    if fmi[i] < *fbest {
                        update_fbest_xbest_nsweepbest(fbest, xbest, nsweepbest, fmi[i], &mut xmin[i], nsweep);
                        update_flag(&mut flag, *fbest, stop_struct);
                        if !flag { return (loc, flag, ncall); }
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
    (loc, flag, ncall)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-16;

    #[test]
    fn test_cover_1() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.2, -0.2, -0.4, 0.15, -0.29, 0.62]);
        let mut f = 10_000.0;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[-0.2, -0.2, -0.4, -0.15, -0.29, -0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
        ];
        let mut fmi = vec![-300.0, -300.0, -300.0, -300.0, -300.0, -300.0];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[1.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = 100_000.0;
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket: Option<usize> = Some(3);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest.as_slice(), [0.06666666666666668, -0.2, -0.4, 0.05, -0.29, 0.20666666666666667]);
        assert_eq!(fbest, -0.00018095444596200413);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[-0.2, -0.2, -0.4, -0.15, -0.29, -0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62])
        ]);
        assert_eq!(fmi, vec![-300.0, -300.0, -300., -300., -300., -300.]);
        assert_eq!(x.as_slice(), [0.06666666666666668, -0.2, -0.4, 0.05, -0.29, 0.20666666666666667]);
        assert_eq!(f, -0.00018095444596200413);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 8, 15));
    }


    #[test]
    fn test_0() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let mut f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = -3.3;
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(0);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(x.as_slice(), [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest.as_slice(), [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
        ]);
        assert_eq!(fmi, vec![-3.3, -3.3, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 0, 1));
    }


    #[test]
    fn test_1() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let mut f = -2.8727241412052247;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]),
        ];
        let mut fmi = vec![-3.3, -3.2, -3.1];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = -3.3;
        let stop = StopStruct {
            nsweeps: 1,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 2;

        let (loc, flag, ncall) =
            basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(x.as_slice(), [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest.as_slice(), [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]),
        ]);
        assert_eq!(fmi, vec![-3.3, -3.2, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 4, 2));
    }


    #[test]
    fn test_2() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let mut f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        let mut fbest = -2.9;
        let stop = StopStruct {
            nsweeps: 0,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(1);
        let nsweep = 10;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(x.as_slice(), [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest.as_slice(), [0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        assert_eq!(fbest, -2.9);
        assert_eq!(xmin, vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3]);
        assert_eq!(fmi, vec![-2.9, -2.8, -2.7]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 1, 1));
    }


    #[test]
    fn test_3() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        let mut f = 0.01;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ];
        let mut fmi = vec![-1., -2., -35.5];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]);
        let mut fbest = -2.3;
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 20;

        let (loc, flag, ncall) =
            basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(x.as_slice(), [-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        assert_eq!(xbest.as_slice(), [-1., 0.15, 0.47, -0.27, 0.31, 0.65]);
        assert_eq!(fbest, -2.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ]);
        assert_eq!(fmi, vec![-1., -2., -35.5]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 6, 20));
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
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 20;

        let (loc, flag, ncall) =
            basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

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
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 0, 20));
    }

    #[test]
    fn test_5() {
        let mut x = SVector::<f64, 6>::from_row_slice(&[-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        let mut f = 100.;
        let mut xmin = vec![
            SVector::<f64, 6>::from_row_slice(&[-10., 1.0, -0.13, 10.2, -0.31, -0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]),
        ];
        let mut fmi = vec![100.; 3];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]);
        let mut fbest = 100.;
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 20;

        let (loc, flag, ncall) =
            basket(hm6, &mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(f, -8.490701428811232e-25);
        assert_eq!(x, SVector::<f64, 6>::from_row_slice(&[-3.466666666666667, 0.3333333333333333, -0.11, -3.366666666666667, -0.29666666666666663, -0.63]));
        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[-3.466666666666667, 0.3333333333333333, -0.11, -3.366666666666667, -0.29666666666666663, -0.63]));
        assert_eq!(fbest, -8.490701428811232e-25);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[-10.0, 1.0, -0.13, 10.2, -0.31, -0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ]);
        assert_eq!(fmi, vec![100.0, 100.0, 100.0]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 6, 20));
    }

    #[test]
    fn test_0_bask1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        let mut fbest = -3.3;
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(0);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]));
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
        ]);
        assert_eq!(fmi, vec![-3.3, -3.3, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 0, 1));
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
        let stop = StopStruct {
            nsweeps: 1,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 2;

        let (loc, flag, ncall) =
            basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]));
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.15, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]),
        ]);
        assert_eq!(fmi, vec![-3.3, -3.2, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (false, true, 4, 2));
    }
    #[test]
    fn test_2_backet_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        let f = -2.8727241412052247;
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        let mut fbest = -2.9;
        let stop = StopStruct {
            nsweeps: 0,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(1);
        let nsweep = 10;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[0.15, 0.1, 0.3, 0.2, 0.25, 0.55]));
        assert_eq!(fbest, -2.9);
        assert_eq!(xmin, vec![SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]); 3]);
        assert_eq!(fmi, vec![-2.9, -2.8, -2.7]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 2, 1));
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
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 20;

        let (loc, flag, ncall) =
            basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[-1., 0.15, 0.47, -0.27, 0.31, 0.65]));
        assert_eq!(fbest, -2.3);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.44, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ]);
        assert_eq!(fmi, vec![-1., -2., -35.5]);
        assert_eq!((loc, flag, ncall, nsweepbest), (false, true, 2, 20));
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
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 20;

        let (loc, flag, ncall) =
            basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[-1.0, 0.15, 0.47, -0.27, 0.31, 0.65]));
        assert_eq!(fbest, -2.3);
        assert_relative_eq!(xmin.as_slice(), &[
             SVector::<f64, 6>::from_row_slice(&[0.2, 0.0, 0.47, 0.27, 0.31, 0.65]),
             SVector::<f64, 6>::from_row_slice(&[0.06666666666666665, 0.13333333333333333, 0.26, -3.283333333333333, 0.09666666666666662, 0.20666666666666667]),
             SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62])
        ].as_slice() , epsilon = TOLERANCE);
        assert_eq!(fmi, vec![100.0, -8.594107654969967e-08, 100.0]);
        assert_eq!((loc, flag, ncall, nsweepbest), (false, true, 2, 20));
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
        let stop = StopStruct {
            nsweeps: 18,                // maximum number of sweeps
            freach: f64::NEG_INFINITY,  // target function value
            nf: 0,              // maximum number of function evaluations
        };
        let nbasket = Some(3);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(hm6, &x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[1.2, 0.15, 0.47, 0.27, 0.31, 0.65]));
        assert_eq!(fbest, 100000.0);
        assert_eq!(xmin, vec![
            SVector::<f64, 6>::from_row_slice(&[-0.2, -0.2, -0.4, -0.15, -0.29, -0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]),
            SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.4, 0.15, 0.29, 0.62])
        ]);
        assert_eq!(fmi, vec![-300.0, -300.0, -300., -300., -300., -300.]);
        assert_eq!((loc, flag, ncall, nsweepbest), (false, true, 2, 1));
    }
}
