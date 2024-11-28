use crate::{init_func::subint, polint::polint, quadratic_func::quadmin, quadratic_func::quadpol};

pub fn exgain(n: usize,
              n0: &[usize],
              l: &[usize],
              L: &[usize],
              x: &[f64],
              y: &[f64],
              x1: &[f64],
              x2: &[f64],
              fx: f64,
              f0: &Vec<Vec<f64>>,
              f1: &[f64],
              f2: &[f64],
) -> (
    Vec<f64>,  //  e
    usize,    //   isplit
    f64       //   splval
) {
    let mut e = vec![0.0; n];
    let mut emin = f64::INFINITY;
    let mut isplit = 0;
    let mut splval = f64::INFINITY;

    for i in 0..n {
        if n0[i] == 0 {
            let min_value = f0[0..L[i] + 1].iter().map(|row| row[i]).fold(f64::INFINITY, f64::min);
            e[i] = min_value - f0[l[i]][i];

            if e[i] < emin {
                emin = e[i];
                isplit = i;
                splval = f64::INFINITY;
            }
        } else {
            let z1 = [x[i], x1[i], x2[i]];
            let z2 = [0.0, f1[i] - fx, f2[i] - fx];

            let d = polint(&z1, &z2);
            let mut eta1 = x[i];
            let mut eta2 = y[i];
            subint(&mut eta1, &mut eta2);
            let z = quadmin(
                eta1.min(eta2),
                eta1.max(eta2),
                &d,
                &z1,
            );
            e[i] = quadpol(z, &d, &z1);

            if e[i] < emin {
                emin = e[i];
                isplit = i;
                splval = z;
            }
        }
    }
    (e, isplit, splval)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let n = 6;
        let n0 = vec![1, 0, 0, 0, 0, 0];
        let l = vec![1, 1, 1, 1, 1, 1];
        let L = vec![2, 2, 2, 2, 2, 2];
        let x = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let y = vec![0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x2 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fx = -0.505;
        let f0 = vec![
            vec![-0.62, -0.8, -0.5, -0.9, -0.38, -0.05, -0.07],
            vec![-0.5, -0.62, -0.8, -0.8, -0.9, -0.9, -1.45893451],
            vec![-0.08, -0.09, -0.32, -0.025, -0.65, -0.37, -0.6],
        ];
        let f1 = vec![-0.62, -0.8, -0.5, -0.9, -0.38, -0.05];
        let f2 = vec![-0.08, -0.09, -0.32, -0.025, -0.65, -0.37];
        let (e, isplit, splval) = exgain(n, &n0, &l, &L, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);
        assert_eq!(e, vec![-0.0832, -0.18000000000000005, 0.0, -0.09999999999999998, 0.0, 0.0]);
        assert_eq!(isplit, 1);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_1() {
        let n = 4;
        let n0 = vec![1, 0, 0, 0, 0, 0];
        let l = vec![1, 1, 1, 1, 1, 1];
        let L = vec![2, 2, 2, 2, 2, 2];
        let x = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let y = vec![0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x2 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fx = -0.505;
        let f0 = vec![
            vec![-0.62, -0.8, -0.5, -0.9, -0.38, -0.05, -0.07],
            vec![-0.5, -0.62, -0.8, -0.8, -0.9, -0.9, -1.45893451],
            vec![-0.08, -0.09, -0.32, -0.025, -0.65, -0.37, -0.6],
        ];
        let f1 = vec![-0.62, -0.8, -0.5, -0.9, -0.38, -0.05];
        let f2 = vec![-0.08, -0.09, -0.32, -0.025, -0.65, -0.37];
        let (e, isplit, splval) = exgain(n, &n0, &l, &L, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);
        assert_eq!(e, vec![-0.0832, -0.18000000000000005, 0.0, -0.09999999999999998]);
        assert_eq!(isplit, 1);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_2() {
        let n = 6;
        let n0 = vec![1, 0, 1, 0, 1, 0];
        let l = vec![1, 1, 1, 1, 1, 1];
        let L = vec![2, 2, 2, 2, 2, 2];
        let x = vec![-0.5, -0.5, -0.5, -0.5, -0.5, -0.5];
        let y = vec![0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x2 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fx = -0.505;
        let f0 = vec![
            vec![-0.62, -0.8, -0.5, -0.9, -0.38, -0.05, -0.07],
            vec![-0.5, -0.62, -0.8, -0.8, -0.9, -0.9, -1.45893451],
            vec![-0.08, -0.09, -0.32, -0.025, -0.65, -0.37, -0.6],
        ];
        let f1 = vec![0.62, 0.8, 0.5, 0.9, 0.38, 0.05];
        let f2 = vec![0.08, 0.09, 0.32, 0.025, 0.65, 0.37];
        let (e, isplit, splval) = exgain(n, &n0, &l, &L, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);
        assert_eq!(e, vec![0.24249600000000007, -0.18000000000000005, 0.37815, -0.09999999999999998, 0.31800000000000006, 0.0]);
        assert_eq!(isplit, 1);
        assert_eq!(splval, f64::INFINITY);
    }

    #[test]
    fn test_3() {
        let n = 6;
        let n0 = vec![1; 6];
        let l = vec![1; 6];
        let L = vec![2; 6];
        let x = vec![-0.5; 6];
        let y = vec![0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x1 = vec![0.0; 6];
        let x2 = vec![1.0; 6];
        let fx = -0.505;
        let f0 = vec![
            vec![-0.62, -0.8, -0.5, -0.9, -0.38, -0.05, -0.07],
            vec![-0.5, -0.62, -0.8, -0.8, -0.9, -0.9, -1.45893451],
            vec![-0.08, -0.09, -0.32, -0.025, -0.65, -0.37, -0.6],
        ];
        let f1 = vec![0.62, 0.8, 0.5, 0.9, 0.38, 0.05];
        let f2 = vec![0.08, 0.09, 0.32, 0.025, 0.65, 0.37];
        let (e, isplit, splval) = exgain(n, &n0, &l, &L, &x, &y, &x1, &x2, fx, &f0, &f1, &f2);
        assert_eq!(e, vec![0.24249600000000007, 0.5077000000000002, 0.37815, 0.5300000000000002, 0.31800000000000006, 0.19415000000000004]);
        assert_eq!(isplit, 5);
        assert_eq!(splval, -0.35);
    }
}