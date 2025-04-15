use nalgebra::SVector;
use std::cmp::Ordering;

pub(super) fn lsrange<const N: usize>(
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    u: &SVector<f64, N>, // xl
    v: &SVector<f64, N>, // xu
    bend: bool,
) -> (
    f64, // amin
    f64, // amax
    f64, // scale
) {
    // Check for zero search direction
    debug_assert!(p.into_iter().fold(0.0_f64, |acc, &p_i| acc.max(p_i.abs())) != 0.0);

    // find sensible step size scale
    // pp=abs(p(p~=0));
    // u=abs(x(p~=0))./pp;
    // scale=min(u);
    // if scale==0,
    //   u(u==0)=1./pp(u==0);
    //   scale=min(u);
    // end;
    let scale = x.iter()
        .zip(p)
        .filter(|(_, p_i)| **p_i != 0.0)
        .map(
            |(x_i, pp_i)|
                match x_i.abs() / pp_i.abs() { // the only situation when scale==0 is when x_i==0 (as x_i.abs() / pp_i.abs() >=0)
                    0.0 => 1.0 / pp_i.abs(),
                    num => num
                }
        )
        .min_by(|a, b| a.total_cmp(b)).unwrap();

    if !bend {
        // find range of useful alp in truncated line search
        let (mut amin, mut amax) = (f64::NEG_INFINITY, f64::INFINITY);

        for (i, &p_i) in p.iter().enumerate() {
            match p_i.total_cmp(&0.0_f64) {
                Ordering::Greater => {
                    amin = amin.max((u[i] - x[i]) / p_i);
                    amax = amax.min((v[i] - x[i]) / p_i);
                }
                Ordering::Less => {
                    amin = amin.max((v[i] - x[i]) / p_i);
                    amax = amax.min((u[i] - x[i]) / p_i);
                }
                Ordering::Equal => {}
            }
        }
        debug_assert!(amin <= amax);
        (amin, amax, scale)
    } else {
        // find range of useful alp in bent line search
        let (mut amin, mut amax) = (f64::INFINITY, f64::NEG_INFINITY);

        for (i, &p_i) in p.iter().enumerate() {
            match p_i.total_cmp(&0.0_f64) {
                Ordering::Greater => {
                    amin = amin.min((u[i] - x[i]) / p_i);
                    amax = amax.max((v[i] - x[i]) / p_i);
                }
                Ordering::Less => {
                    amin = amin.min((v[i] - x[i]) / p_i);
                    amax = amax.max((u[i] - x[i]) / p_i);
                }
                Ordering::Equal => {}
            }
        }
        (amin, amax, scale)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let xl = SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]);
        let xu = SVector::<f64, 2>::from_row_slice(&[10.0, 10.0]);
        let x = SVector::<f64, 2>::from_row_slice(&[5.0, 5.0]);
        let p = SVector::<f64, 2>::from_row_slice(&[1.0, 1.0]);
        let bend = false;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -5.0);
        assert_eq!(amax, 5.0);
        assert_eq!(scale, 5.0);
    }

    #[test]
    fn test_1() {
        // Matlab equivalent code
        //
        // prt = 0;
        // xl = [0.1; 2.0; 0.1; -2.0; 0.1; 2.0];
        // xu = [1.1; 3.0; 4.1; 5.0; 6.1; 24.0];
        // x = [1.0; 2.1; 1.4; -1.2; 0.1; 23.1];
        // p = [1.0; -1.0; 2.1; -5.1; 0.0; -6.0];
        // bend = false;
        //
        // lsrange;
        //
        // disp(amin);
        // disp(amax);
        // disp(scale);

        let xl = SVector::<f64, 6>::from_row_slice(&[0.1, 2.0, 0.1, -2.0, 0.1, 2.0]);
        let xu = SVector::<f64, 6>::from_row_slice(&[1.1, 3.0, 4.1, 5.0, 6.1, 24.0]);
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.1, 1.4, -1.2, 0.1, 23.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, -1.0, 2.1, -5.1, 0.0, -6.0]);
        let bend = false;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -0.14999999999999977);
        assert_eq!(amax, 0.10000000000000009);
        assert_eq!(scale, 0.23529411764705882);
    }

    #[test]
    fn test_2() {
        // Matlab equivalent code
        //
        // prt = 0;
        // xl = [0.1; 2.0; 0.1; -2.0; 0.1; 2.0];
        // xu = [1.1; 3.0; 4.1; 5.0; 6.1; 24.0];
        // x = [1.0; 2.1; 1.4; -1.2; 0.1; 23.1];
        // p = [1.0; -1.0; 2.1; -5.1; 0.0; -6.0];
        // bend = true;
        //
        // lsrange;
        //
        // disp(amin);
        // disp(amax);
        // disp(scale);

        let xl = SVector::<f64, 6>::from_row_slice(&[0.1, 2.0, 0.1, -2.0, 0.1, 2.0]);
        let xu = SVector::<f64, 6>::from_row_slice(&[1.1, 3.0, 4.1, 5.0, 6.1, 24.0]);
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.1, 1.4, -1.2, 0.1, 23.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, -1.0, 2.1, -5.1, 0.0, -6.0]);
        let bend = true;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -1.215686274509804);
        assert_eq!(amax, 3.516666666666667);
        assert_eq!(scale, 0.23529411764705882);
    }

    #[test]
    #[should_panic]
    fn test_3() {
        let xl = SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]);
        let xu = SVector::<f64, 2>::from_row_slice(&[10.0, 10.0]);
        let x = SVector::<f64, 2>::from_row_slice(&[5.0, 5.0]);
        let p = SVector::<f64, 2>::zeros(); // zero search direction
        let bend = false;

        let _ = lsrange(&x, &p, &xl, &xu, bend);
    }

    #[test]
    fn test_5() {
        let xl = SVector::<f64, 2>::from_row_slice(&[0.0, 2.0]);
        let xu = SVector::<f64, 2>::from_row_slice(&[10.0, 20.0]);
        let x = SVector::<f64, 2>::from_row_slice(&[-5.0, 0.5]);
        let p = SVector::<f64, 2>::from_row_slice(&[-10.0, 1.0]);
        let bend = true;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -1.5);
        assert_eq!(amax, 19.5);
        assert_eq!(scale, 0.5);
    }

    #[test]
    fn test_6() {
        // Matlab equivalent code
        //
        // prt = 0;
        // xl = [-0.1; -2.0; 0.1; -2.0; 0.1; 2.0];
        // xu = [5.1; 2.0; 4.1; 5.0; 6.3; 4.5];
        // x = [1.1; -1.1; 2.4; 1.2; 4.1; 0.1];
        // p = [-1.0; 1.0; -2.1; -2.1; 3.0; -6.0];
        // bend = false;
        //
        // lsrange;
        //
        // disp(amin);
        // disp(amax);
        // disp(scale);

        let xl = SVector::<f64, 6>::from_row_slice(&[-0.1, -2.0, 0.1, -2.0, 0.1, 2.0]);
        let xu = SVector::<f64, 6>::from_row_slice(&[5.1, 2.0, 4.1, 5.0, 6.3, 4.5]);
        let x = SVector::<f64, 6>::from_row_slice(&[1.1, -1.1, 2.4, 1.2, 4.1, 0.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, 1.0, -2.1, -2.1, 3.0, -6.0]);
        let bend = false;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -0.7333333333333334);
        assert_eq!(amax, -0.31666666666666665);
        assert_eq!(scale, 0.016666666666666666);
    }

    #[test]
    fn test_7() {
        // Matlab equivalent code
        //
        // prt = 0;
        // xl = [-0.1; -2.0; 0.1; -2.0; 0.1; 2.0];
        // xu = [5.1; 2.0; 4.1; 5.0; 6.3; 4.5];
        // x = [1.1; -1.1; 2.4; 1.2; 4.1; 0.1];
        // p = [-1.3; 1.1; 2.5; -2.1; 3.0; 1.3];
        // bend = true;
        //
        // lsrange;
        //
        // disp(amin);
        // disp(amax);
        // disp(scale);

        let xl = SVector::<f64, 6>::from_row_slice(&[-0.1, -2.0, 0.1, -2.0, 0.1, 2.0]);
        let xu = SVector::<f64, 6>::from_row_slice(&[5.1, 2.0, 4.1, 5.0, 6.3, 4.5]);
        let x = SVector::<f64, 6>::from_row_slice(&[1.1, -1.1, 2.4, 1.2, 4.1, 0.1]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.3, 1.1, 2.5, -2.1, 3.0, 1.3]);
        let bend = true;

        let (amin, amax, scale) = lsrange(&x, &p, &xl, &xu, bend);

        assert_eq!(amin, -3.0769230769230766);
        assert_eq!(amax, 3.3846153846153846);
        assert_eq!(scale, 0.07692307692307693);
    }
}