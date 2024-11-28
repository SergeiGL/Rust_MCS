use crate::chk_flag::{chrelerr, chvtr};
use crate::feval::feval;

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
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


pub fn basket(
    x: &mut Vec<f64>,
    f: &mut f64,
    xmin: &mut Vec<Vec<f64>>,
    fmi: &mut Vec<f64>,
    xbest: &mut Vec<f64>,
    fbest: &mut f64,
    stop: &[f64],
    nbasket: &usize,
    nsweep: &usize,
    nsweepbest: &mut usize,
) -> (
    bool,            // loc
    bool,            // flag
    usize,           // ncall
) {
    let mut loc = true;
    let mut flag = true;
    let mut ncall: usize = 0;

    if *nbasket == 0 {
        return (loc, flag, ncall);
    }

    let mut dist: Vec<f64> = vec![0.0; nbasket + 1];
    for k in 0..dist.len() {
        if k < xmin.len() {
            dist[k] = euclidean_distance(&x, &xmin[k]);
        } else {
            dist[k] = f64::INFINITY;
        }
    }

    // Get sorted indices based on distances
    let ind = argsort(&dist);

    for &k in ind.iter().take(nbasket + 1) {
        if fmi[k] <= *f {
            let p: Vec<f64> = xmin[k]
                .iter()
                .zip(&mut *x)
                .map(|(&xmin_k, &mut x_val)| xmin_k - x_val)
                .collect();

            let y1: Vec<f64> = x
                .iter()
                .zip(&p)
                .map(|(&x_val, p_val)| x_val + p_val / 3.0)
                .collect();

            // Evaluate f1
            let f1 = feval(&y1);
            ncall += 1;

            if f1 <= *f {
                // Compute y2 = x + 2/3 * p
                let y2: Vec<f64> = x
                    .iter()
                    .zip(&p)
                    .map(|(x_val, p_val)| x_val + (2.0 / 3.0) * p_val)
                    .collect();

                // Evaluate f2
                let f2 = feval(&y2);
                ncall += 1;

                if f2 > f1.max(fmi[k]) {
                    if f1 < *f {
                        *x = y1.clone();
                        *f = f1;
                        if f < fbest {
                            *fbest = *f;
                            *xbest = x.clone();
                            *nsweepbest = *nsweep;
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
                        *x = y1.clone();
                        if f < fbest {
                            *fbest = *f;
                            *xbest = x.clone();
                            *nsweepbest = *nsweep;
                            if stop.len() > 0 && stop[0] > 0.0 && stop[0] < 1.0 {
                                flag = chrelerr(*fbest, stop);
                            } else if stop.len() > 1 && stop[0] == 0.0 {
                                flag = chvtr(*fbest, stop[1]);
                            }
                            if !flag { return (loc, flag, ncall); }
                        } else if f2 < f1.min(fmi[k]) {
                            *f = f2;
                            *x = y2.clone();
                            if f < fbest {
                                *fbest = *f;
                                *xbest = x.clone();
                                *nsweepbest = *nsweep;
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


pub fn basket1(
    x: &mut Vec<f64>,
    f: &mut f64,
    xmin: &mut Vec<Vec<f64>>,
    fmi: &mut Vec<f64>,
    xbest: &mut Vec<f64>,
    fbest: &mut f64,
    stop: &[f64],
    nbasket: &usize,
    nsweep: &usize,
    nsweepbest: &mut usize,
) -> (
    bool,            // loc
    bool,            // flag
    usize,          // ncall
) {
    let mut loc = true;
    let mut flag = true;
    let mut ncall: usize = 0;

    if *nbasket == 0 {
        return (loc, flag, ncall);
    }

    // Initialize distance vector
    let mut dist: Vec<f64> = vec![0.0; nbasket + 1];
    for k in 0..dist.len() {
        if k < xmin.len() {
            dist[k] = euclidean_distance(&x, &xmin[k]);
        } else {
            dist[k] = f64::INFINITY;
        }
    }
    let ind = argsort(&dist);

    for &k in ind.iter().take(nbasket + 1) {
        // if k >= fmi.len() {
        //     continue;
        // }

        // Compute p = xmin[k] - x
        let p: Vec<f64> = xmin[k]
            .iter()
            .zip(&mut *x)
            .map(|(&xmin_k, &mut x_val)| xmin_k - x_val)
            .collect();

        // Compute y1 = x + p / 3
        let y1: Vec<f64> = x
            .iter()
            .zip(&p)
            .map(|(&x_val, p_val)| x_val + p_val / 3.0)
            .collect();

        // Evaluate f1
        let f1 = feval(&y1);
        ncall += 1;

        if f1 <= fmi[k].max(*f) {
            // Compute y2 = x + 2/3 * p
            let y2: Vec<f64> = x
                .iter()
                .zip(&p)
                .map(|(x_val, p_val)| x_val + (2.0 / 3.0) * p_val)
                .collect();

            // Evaluate f2
            let f2 = feval(&y2);
            ncall += 1;

            if f2 <= f1.max(fmi[k]) {
                if *f < f1.min(f2).min(fmi[k]) {
                    fmi[k] = *f;
                    xmin[k] = x.clone();
                    if fmi[k] < *fbest {
                        *fbest = fmi[k];
                        *xbest = xmin[k].clone();
                        *nsweepbest = *nsweep;
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
                    xmin[k] = y1.clone();
                    if fmi[k] < *fbest {
                        *fbest = fmi[k];
                        *xbest = xmin[k].clone();
                        *nsweepbest = *nsweep;
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
                    xmin[k] = y2.clone();
                    if fmi[k] < *fbest {
                        *fbest = fmi[k];
                        *xbest = xmin[k].clone();
                        *nsweepbest = *nsweep;
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
    fn better_cover_1() {
        let mut x = vec![0.2, -0.2, -0.4, 0.15, -0.29, 0.62];
        let mut f = 10_000.0;
        let mut xmin = vec![
            vec![-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
        ];
        let mut fmi = vec![-300.0, -300.0, -300.0, -300.0, -300.0, -300.0];
        let mut xbest = vec![1.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = 100_000.0;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = 3;
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(xbest, vec![0.06666666666666668, -0.2, -0.4, 0.05, -0.29, 0.20666666666666667]);
        assert_eq!(fbest, -0.00018095444596200413);
        assert_eq!(xmin, vec![
            vec![-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]
        ]);
        assert_eq!(fmi, vec![-300.0, -300.0, -300., -300., -300., -300.]);
        assert_eq!(x, vec![0.06666666666666668, -0.2, -0.4, 0.05, -0.29, 0.20666666666666667]);
        assert_eq!(f, -0.00018095444596200413);
        assert_eq!((loc, flag, ncall, nsweep, nsweepbest), (true, true, 8, 15, 15));
    }

    #[test]
    fn better_cover_2() {
        let mut x = vec![0.2, -0.2, -0.4, 0.15, -0.29, 0.62];
        let mut f = 10_000.0;
        let mut xmin = vec![
            vec![-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
        ];
        let mut fmi = vec![-300.0, -300.0, -300.0, -300.0, -300.0, -300.0];
        let mut xbest = vec![1.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = 100_000.0;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = 3;
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(xbest, vec![1.2, 0.15, 0.47, 0.27, 0.31, 0.65]);
        assert_eq!(fbest, 100000.0);
        assert_eq!(xmin, vec![
            vec![-0.2, -0.2, -0.4, -0.15, -0.29, -0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]
        ]);
        assert_eq!(fmi, vec![-300.0, -300.0, -300., -300., -300., -300.]);
        assert_eq!((loc, flag, ncall, nsweep, nsweepbest), (false, true, 2, 15, 1));
    }


    #[test]
    fn test_0() {
        let mut x = vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = 0;
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(
            (xbest, fbest, xmin, fmi, x, f, loc, flag, ncall, nsweepbest),
            (
                vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
                -3.3,
                vec![
                    vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                    vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                    vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                ],
                vec![-3.3, -3.3, -3.1],
                vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                -2.8727241412052247,
                true,
                true,
                0,
                1,
            )
        );
    }

    #[test]
    fn test_0_bask1() {
        let mut x = vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-3.3, -3.3, -3.1];
        let mut xbest = vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![18.0, f64::NEG_INFINITY];
        let nbasket = 0;
        let nsweep = 15;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(
            (xbest, fbest, xmin, fmi, loc, flag, ncall, nsweepbest),
            (
                vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
                -3.3,
                vec![
                    vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                    vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                    vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                ],
                vec![-3.3, -3.3, -3.1],
                true,
                true,
                0,
                1,
            )
        );
    }

    #[test]
    fn test_1() {
        let mut x = vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![
            vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
            vec![0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
        ];
        let mut fmi = vec![-3.3, -3.2, -3.1];
        let mut xbest = vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![1.0, f64::NEG_INFINITY];
        let nbasket = 2;
        let nsweep = 20;
        let mut nsweepbest = 2;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(
            (xbest, fbest, xmin, fmi, x, f, loc, flag, ncall, nsweepbest),
            (
                vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
                -3.3,
                vec![
                    vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
                    vec![0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
                    vec![0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
                ],
                vec![-3.3, -3.2, -3.1],
                vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                -2.8727241412052247,
                true,
                true,
                4,
                2,
            )
        );
    }

    #[test]
    fn test_1_backet_1() {
        let mut x = vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![
            vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
            vec![0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
            vec![0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
        ];
        let mut fmi = vec![-3.3, -3.2, -3.1];
        let mut xbest = vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let mut fbest = -3.3;
        let stop = vec![1.0, f64::NEG_INFINITY];
        let nbasket = 2;
        let nsweep = 20;
        let mut nsweepbest = 2;

        let (loc, flag, ncall) =
            basket1(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(
            (xbest, fbest, xmin, fmi, loc, flag, ncall, nsweepbest),
            (
                vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
                -3.3,
                vec![
                    vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65],
                    vec![0.2, 0.2, 0.44517882, 0.15, 0.29, 0.62],
                    vec![0.2, 0.2, 0.4, 0.15, 0.37887001, 0.62],
                ],
                vec![-3.3, -3.2, -3.1],
                false,
                true,
                4,
                2,
            )
        );
    }


    #[test]
    fn test_2() {
        let mut x = vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = vec![0.15, 0.1, 0.3, 0.2, 0.25, 0.55];
        let mut fbest = -2.9;
        let stop = vec![0.1, f64::NEG_INFINITY];
        let nbasket = 1;
        let nsweep = 10;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(
            (xbest, fbest, xmin, fmi, x, f, loc, flag, ncall, nsweepbest),
            (
                vec![0.15, 0.1, 0.3, 0.2, 0.25, 0.55],
                -2.9,
                vec![vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3],
                vec![-2.9, -2.8, -2.7],
                vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62],
                -2.8727241412052247,
                true,
                true,
                1,
                1,
            )
        );
    }


    #[test]
    fn test_2_backet_1() {
        let mut x = vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62];
        let mut f = -2.8727241412052247;
        let mut xmin = vec![vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3];
        let mut fmi = vec![-2.9, -2.8, -2.7];
        let mut xbest = vec![0.15, 0.1, 0.3, 0.2, 0.25, 0.55];
        let mut fbest = -2.9;
        let stop = vec![0.1, f64::NEG_INFINITY];
        let nbasket = 1;
        let nsweep = 10;
        let mut nsweepbest = 1;

        let (loc, flag, ncall) =
            basket1(&mut x, &mut f, &mut xmin, &mut fmi, &mut xbest, &mut fbest, &stop, &nbasket, &nsweep, &mut nsweepbest);

        assert_eq!(
            (xbest, fbest, xmin, fmi, loc, flag, ncall, nsweepbest),
            (
                vec![0.15, 0.1, 0.3, 0.2, 0.25, 0.55],
                -2.9,
                vec![vec![0.2, 0.2, 0.4, 0.15, 0.29, 0.62]; 3],
                vec![-2.9, -2.8, -2.7],
                true,
                true,
                2,
                1,
            )
        );
    }
}
