use crate::split_func::split2;
use nalgebra::SVector;

pub fn splrnk<const N: usize>(
    n0: &SVector<usize, N>,
    p: &SVector<usize, N>,
    x: &SVector<f64, N>,
    y: &SVector<f64, N>,
) -> (
    isize,  // isplit
    f64     // splval
) {
    let (mut isplit, mut n1, mut p1) = (0, n0[0], p[0]);

    // Find the splitting index
    for i in 1..N {
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

    (isplit as isize, splval)
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_inf() {
        let n0 = SVector::<usize, 2>::from_row_slice(&[0, 0]);
        let p = SVector::<usize, 2>::from_row_slice(&[4, 5]);
        let x = SVector::<f64, 2>::from_row_slice(&[7.0, 8.0]);
        let y = SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]);
        assert_eq!(splrnk(&n0, &p, &x, &y), (0, f64::INFINITY));
    }

    #[test]
    fn test_0() {
        let n0 = SVector::<usize, 6>::from_row_slice(&[2, 2, 3, 2, 2, 3]);
        let p = SVector::<usize, 6>::from_row_slice(&[3, 5, 1, 4, 2, 0]);
        let x = SVector::<f64, 6>::from_row_slice(&[0.20, 0.20, 0.46, 0.16, 0.30, 0.62]);
        let y = SVector::<f64, 6>::from_row_slice(&[0.07, 0.07, 0.45, 0.06, 0.42131067, 0.67]);
        assert_eq!(splrnk(&n0, &p, &x, &y), (4, 0.38087378));
    }

    #[test]
    fn test_1() {
        let n0 = SVector::<usize, 4>::from_row_slice(&[2, 2, 3, 2]);
        let p = SVector::<usize, 4>::from_row_slice(&[3, 5, 1, 4]);
        let x = SVector::<f64, 4>::from_row_slice(&[0.20, 0.20, 0.46, 0.16]);
        let y = SVector::<f64, 4>::from_row_slice(&[0.07, 0.07, 0.45, 0.06]);
        assert_eq!(splrnk(&n0, &p, &x, &y), (0, 0.11333333333333334));
    }

    #[test]
    fn test_2() {
        let n0 = SVector::<usize, 6>::from_row_slice(&[2, 2, 3, 2, 2, 3]);
        let p = SVector::<usize, 6>::from_row_slice(&[3, 5, 1, 4, 2, 0]);
        let x = SVector::<f64, 6>::from_row_slice(&[-0.20, 0.20, -0.46, 0.16, -0.30, 0.62]);
        let y = SVector::<f64, 6>::from_row_slice(&[-0.07, 0.07, -0.45, 0.06, -0.42131067, -0.67]);
        assert_eq!(splrnk(&n0, &p, &x, &y), (4, -0.38087378));
    }
}