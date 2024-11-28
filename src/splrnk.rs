use crate::split_func::split2;

pub fn splrnk(n: usize, n0: &[usize], p: &[usize], x: &[f64], y: &[f64]) ->
(
    usize, //  isplit
    f64   //   splval
) {
    let mut isplit = 0;
    let mut n1 = n0[0];
    let mut p1 = p[0];

    // Find the splitting index
    for i in 1..n {
        if n0[i] < n1 || (n0[i] == n1 && p[i] < p1) {
            isplit = i;
            n1 = n0[i];
            p1 = p[i];
        }
    }

    let splval = if n1 > 0 {
        split2(x[isplit], y[isplit])
    } else {
        f64::INFINITY
    };

    (isplit, splval)
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_inf() {
        let n = 2;
        let n0 = vec![0, 0, 0];
        let p = vec![4, 5, 6];
        let x = vec![7.0, 8.0, 9.0];
        let y = vec![1.0, 2.0, 3.0];
        assert_eq!(splrnk(n, &n0, &p, &x, &y), (0, f64::INFINITY));
    }

    #[test]
    fn test_0() {
        let n = 6;
        let n0 = vec![2, 2, 3, 2, 2, 3];
        let p = vec![3, 5, 1, 4, 2, 0];
        let x = vec![0.20, 0.20, 0.46, 0.16, 0.30, 0.62];
        let y = vec![0.07, 0.07, 0.45, 0.06, 0.42131067, 0.67];
        assert_eq!(splrnk(n, &n0, &p, &x, &y), (4, 0.38087378));
    }

    #[test]
    fn test_1() {
        let n = 4;
        let n0 = vec![2, 2, 3, 2, 2, 3];
        let p = vec![3, 5, 1, 4, 2, 0];
        let x = vec![0.20, 0.20, 0.46, 0.16, 0.30, 0.62];
        let y = vec![0.07, 0.07, 0.45, 0.06, 0.42131067, 0.67];
        assert_eq!(splrnk(n, &n0, &p, &x, &y), (0, 0.11333333333333334));
    }

    #[test]
    fn test_2() {
        let n = 6;
        let n0 = vec![2, 2, 3, 2, 2, 3];
        let p = vec![3, 5, 1, 4, 2, 0];
        let x = vec![-0.20, 0.20, -0.46, 0.16, -0.30, 0.62];
        let y = vec![-0.07, 0.07, -0.45, 0.06, -0.42131067, -0.67];
        assert_eq!(splrnk(n, &n0, &p, &x, &y), (4, -0.38087378));
    }
}