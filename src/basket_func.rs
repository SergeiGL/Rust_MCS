use crate::chk_flag::{chrelerr, chvtr};
use crate::feval::feval;

fn euclidean_distance<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn argsort(v: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..v.len()).collect();
    indices.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    indices
}


pub fn basket<const N: usize>(
    x: &mut [f64; N],
    f: &mut f64,
    xmin: &Vec<[f64; N]>,
    fmi: &Vec<f64>,
    xbest: &mut [f64; N],
    fbest: &mut f64,
    stop: &[f64],
    nbasket: &Option<usize>,
    nsweep: usize,
    nsweepbest: &mut usize,
) -> (
    bool,   // loc
    bool,   // flag
    usize,  // ncall
) {
    let mut loc = true;
    let mut flag = true;
    let mut ncall: usize = 0;

    let nbasket_plus_1 = match nbasket {
        Some(0) => return (loc, flag, ncall),
        None => 0,
        Some(n) => n + 1,
    };
    let mut dist: Vec<f64> = vec![0.0; nbasket_plus_1];
    for k in 0..dist.len() {
        if k < xmin.len() {
            dist[k] = euclidean_distance(x, &xmin[k]);
        } else {
            dist[k] = f64::INFINITY;
        }
    }

    // Get sorted indices based on distances
    let ind = argsort(&dist);

    for &k in ind.iter().take(nbasket_plus_1) {
        if fmi[k] <= *f {
            let p: Vec<f64> = xmin[k]
                .iter()
                .zip(&mut *x)
                .map(|(&xmin_k, &mut x_val)| xmin_k - x_val)
                .collect();

            let mut y1: [f64; N] = [0.0; N];
            for i in 0..N {
                y1[i] = x[i] + p[i] / 3.0;
            };

            // Evaluate f1
            let f1 = feval(&y1);
            ncall += 1;

            if f1 <= *f {
                // Compute y2 = x + 2/3 * p
                let mut y2: [f64; N] = [0.0; N];
                for i in 0..N {
                    y2[i] = x[i] + 2.0 * p[i] / 3.0;
                }
                // Evaluate f2
                let f2 = feval(&y2);
                ncall += 1;

                if f2 > f1.max(fmi[k]) {
                    if f1 < *f {
                        *x = y1;
                        *f = f1;
                        if *f < *fbest {
                            *fbest = *f;
                            *xbest = x.clone();
                            *nsweepbest = nsweep;
                            if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                                flag = chrelerr(*fbest, stop);
                            } else if stop.len() > 1 && stop[0] == 0.0 {
                                flag = chvtr(*fbest, stop[1]);
                            }
                            if !flag {
                                return (loc, flag, ncall);
                            }
                        }
                    }
                } else {
                    if f1 < f2.min(fmi[k]) {
                        *f = f1;
                        *x = y1;
                        if *f < *fbest {
                            *fbest = *f;
                            *xbest = x.clone();
                            *nsweepbest = nsweep;
                            if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                                flag = chrelerr(*fbest, stop);
                            } else if stop.len() > 1 && stop[0] == 0.0 {
                                flag = chvtr(*fbest, stop[1]);
                            }
                            if !flag { return (loc, flag, ncall); }
                        } else if f2 < f1.min(fmi[k]) {
                            *f = f2;
                            *x = y2.clone();
                            if *f < *fbest {
                                *fbest = *f;
                                *xbest = x.clone();
                                *nsweepbest = nsweep;
                                if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                                    flag = chrelerr(*fbest, stop);
                                } else if stop.len() > 1 && stop[0] == 0.0 {
                                    flag = chvtr(*fbest, stop[1]);
                                }
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
    x: &[f64; N],
    f: f64,
    xmin: &mut Vec<[f64; N]>,
    fmi: &mut Vec<f64>,
    xbest: &mut [f64; N],
    fbest: &mut f64,
    stop: &[f64],
    nbasket: &Option<usize>,
    nsweep: usize,
    nsweepbest: &mut usize,
) -> (
    bool,            // loc
    bool,            // flag
    usize,          // ncall
) {
    let mut loc = true;
    let mut flag = true;
    let mut ncall: usize = 0;

    let nbasket_plus_1 = match nbasket {
        Some(0) => return (loc, flag, ncall),
        None => 0,
        Some(n) => n + 1,
    };
    // Initialize distance vector
    let mut dist: Vec<f64> = vec![0.0; nbasket_plus_1];
    for k in 0..dist.len() {
        if k < xmin.len() {
            dist[k] = euclidean_distance(&x, &xmin[k]);
        } else {
            dist[k] = f64::INFINITY;
        }
    }
    let ind = argsort(&dist);

    for &k in ind.iter().take(nbasket_plus_1) {
        // if k >= fmi.len() {
        //     continue;
        // }

        // Compute p = xmin[k] - x
        let mut p = [0.0; N];
        for i in 0..N {
            p[i] = xmin[k][i] - x[i]
        }

        // Compute y1 = x + p / 3
        let mut y1: [f64; N] = [0.0; N];
        for i in 0..N {
            y1[i] = x[i] + p[i] / 3.0;
        }

        // Evaluate f1
        let f1 = feval(&y1);
        ncall += 1;

        if f1 <= fmi[k].max(f) {
            // Compute y2 = x + 2/3 * p
            let mut y2: [f64; N] = [0.0; N];
            for i in 0..N {
                y2[i] = x[i] + 2.0 * p[i] / 3.0;
            }

            // Evaluate f2
            let f2 = feval(&y2);
            ncall += 1;

            if f2 <= f1.max(fmi[k]) {
                if f < f1.min(f2).min(fmi[k]) {
                    fmi[k] = f;
                    xmin[k] = x.clone();
                    if fmi[k] < *fbest {
                        *fbest = fmi[k];
                        *xbest = xmin[k].clone();
                        *nsweepbest = nsweep;
                        if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                            flag = chrelerr(*fbest, stop);
                        } else if stop.len() > 1 && stop[0] == 0.0 {
                            flag = chvtr(*fbest, stop[1]);
                        }
                        if !flag {
                            return (loc, flag, ncall);
                        }
                    }
                    loc = false;
                    break;
                } else if f1 < f.min(f2).min(fmi[k]) {
                    fmi[k] = f1;
                    xmin[k] = y1;
                    if fmi[k] < *fbest {
                        *fbest = fmi[k];
                        *xbest = xmin[k].clone();
                        *nsweepbest = nsweep;
                        if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                            flag = chrelerr(*fbest, stop);
                        } else if stop.len() > 1 && stop[0] == 0.0 {
                            flag = chvtr(*fbest, stop[1]);
                        }
                        if !flag {
                            return (loc, flag, ncall);
                        }
                    }
                    loc = false;
                    break;
                } else if f2 < f.min(f1).min(fmi[k]) {
                    fmi[k] = f2;
                    xmin[k] = y2;
                    if fmi[k] < *fbest {
                        *fbest = fmi[k];
                        *xbest = xmin[k].clone();
                        *nsweepbest = nsweep;
                        if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                            flag = chrelerr(*fbest, stop);
                        } else if stop.len() > 1 && stop[0] == 0.0 {
                            flag = chvtr(*fbest, stop[1]);
                        }
                        if !flag {
                            return (loc, flag, ncall);
                        }
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

    #[test]
    fn test_cover_1() {
        let mut x = [0.2, -0.2, -0.4, 0.15, -0.29, 0.62];
        let mut f = 10_000.0;
        let mut xmin = vec![
            [-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
        ];
        let mut fmi = vec![-300.0, -300.0, -300.0, -300.0, -300.0, -300.0];
        let mut xbest = [1.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = 100_000.0;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket: Option<usize> = Some(3);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, [0.06666666666666668, -0.2, -0.4, 0.05, -0.29, 0.20666666666666667]);
        assert_eq!(fbest, -0.00018095444596200413);
        assert_eq!(xmin, vec![
            [-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]
        ]);
        assert_eq!(fmi, vec![-300.0, -300.0, -300., -300., -300., -300.]);
        assert_eq!(x, [0.06666666666666668, -0.2, -0.4, 0.05, -0.29, 0.20666666666666667]);
        assert_eq!(f, -0.00018095444596200413);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 8, 15));
    }


    #[test]
    fn test_0() {
        let mut x = [0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = [0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = Some(0);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(x, [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest, [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
        ]);
        assert_eq!(fmi, vec![-3.3, -3.3, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 0, 1));
    }


    #[test]
    fn test_1() {
        let mut x = [0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![
            [0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
        ];
        let mut fmi = vec![-3.3, -3.2, -3.1];
        let mut xbest = [0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![1.0, f64::NEG_INFINITY];
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 2;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(x, [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest, [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            [0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
        ]);
        assert_eq!(fmi, vec![-3.3, -3.2, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 4, 2));
    }


    #[test]
    fn test_2() {
        let mut x = [0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = [0.15, 0.1, 0.3, 0.2, 0.25, 0.55];
        let mut fbest = -2.9;
        let stop = vec![0.1, f64::NEG_INFINITY];
        let nbasket = Some(1);
        let nsweep = 10;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(x, [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]);
        assert_eq!(xbest, [0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        assert_eq!(fbest, -2.9);
        assert_eq!(xmin, vec![[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3]);
        assert_eq!(fmi, vec![-2.9, -2.8, -2.7]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 1, 1));
    }


    #[test]
    fn test_3() {
        let mut x = [-0.2, 0., -0.1, -10.15, -0.29, -0.62];
        let mut f = 0.01;
        let mut xmin = vec![
            [0.2, 0.0, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]
        ];
        let mut fmi = vec![-1., -2., -35.5];
        let mut xbest = [-1.0, 0.15, 0.47, -0.27, 0.31, 0.65];
        let mut fbest = -2.3;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 20;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(x, [-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        assert_eq!(xbest, [-1., 0.15, 0.47, -0.27, 0.31, 0.65]);
        assert_eq!(fbest, -2.3);
        assert_eq!(xmin, vec![
            [0.2, 0.0, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]
        ]);
        assert_eq!(fmi, vec![-1., -2., -35.5]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 6, 20));
    }


    #[test]
    fn test_0_bask1() {
        let x = [0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let f = -2.8727241412052247;
        let mut xmin = vec![[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = [0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = Some(0);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(&x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
        ]);
        assert_eq!(fmi, vec![-3.3, -3.3, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 0, 1));
    }
    #[test]
    fn test_1_backet_1() {
        let x = [0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let f = -2.8727241412052247;
        let mut xmin = vec![
            [0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
        ];
        let mut fmi = vec![-3.3, -3.2, -3.1];
        let mut xbest = [0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![1.0, f64::NEG_INFINITY];
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 2;

        let (loc, flag, ncall) =
            basket1(&x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);


        assert_eq!(xbest, [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, -3.3);
        assert_eq!(xmin, vec![
            [0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
        ]);
        assert_eq!(fmi, vec![-3.3, -3.2, -3.1]);
        assert_eq!((loc, flag, ncall, nsweepbest), (false, true, 4, 2));
    }
    #[test]
    fn test_2_backet_1() {
        let x = [0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let f = -2.8727241412052247;
        let mut xmin = vec![[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = [0.15, 0.1, 0.3, 0.2, 0.25, 0.55];
        let mut fbest = -2.9;
        let stop = vec![0.1, f64::NEG_INFINITY];
        let nbasket = Some(1);
        let nsweep = 10;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(&x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, [0.15, 0.1, 0.3, 0.2, 0.25, 0.55]);
        assert_eq!(fbest, -2.9);
        assert_eq!(xmin, vec![[0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3]);
        assert_eq!(fmi, vec![-2.9, -2.8, -2.7]);
        assert_eq!((loc, flag, ncall, nsweepbest), (true, true, 2, 1));
    }
    #[test]
    fn test_3_basket1() {
        let x = [-0.2, 0., -0.1, -10.15, -0.29, -0.62];
        let f = 0.01;
        let mut xmin = vec![
            [0.2, 0.0, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]
        ];
        let mut fmi = vec![-1., -2., -35.5];
        let mut xbest = [-1.0, 0.15, 0.47, -0.27, 0.31, 0.65];
        let mut fbest = -2.3;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = Some(2);
        let nsweep = 20;
        let mut nsweepbest = 20;

        let (loc, flag, ncall) =
            basket1(&x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(x, [-0.2, 0., -0.1, -10.15, -0.29, -0.62]);
        assert_eq!(xbest, [-1., 0.15, 0.47, -0.27, 0.31, 0.65]);
        assert_eq!(fbest, -2.3);
        assert_eq!(xmin, vec![
            [0.2, 0.0, 0.47, 0.27, 0.31, 0.65],
            [0.2, 0.2, 0.44, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62]
        ]);
        assert_eq!(fmi, vec![-1., -2., -35.5]);
        assert_eq!((loc, flag, ncall, nsweepbest), (false, true, 2, 20));
    }

    #[test]
    fn better_cover_2() {
        let x = [0.2, -0.2, -0.4, 0.15, -0.29, 0.62];
        let f = 10_000.0;
        let mut xmin = vec![
            [-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
        ];
        let mut fmi = vec![-300.0, -300.0, -300.0, -300.0, -300.0, -300.0];
        let mut xbest = [1.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = 100_000.0;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = Some(3);
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(&x, f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, nsweep, &mut nsweepbest);

        assert_eq!(xbest, [1.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, 100000.0);
        assert_eq!(xmin, vec![
            [-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            [0.2, 0.2, 0.4, 0.15, 0.29, 0.62]
        ]);
        assert_eq!(fmi, vec![-300.0, -300.0, -300., -300., -300., -300.]);
        assert_eq!((loc, flag, ncall, nsweepbest), (false, true, 2, 1));
    }
}
