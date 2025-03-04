use nalgebra::SVector;

pub fn feval<const N: usize>(x: &SVector<f64, N>) -> f64 {
    #[cfg(test)]
    {
        hm6(x.as_slice())  // Called only in test mode
    }

    #[cfg(not(test))]
    {
        release_func(x)   // Called in non-test configurations (e.g., release mode); for real world usage
    }
}

#[inline]
fn release_func<const N: usize>(_x: &SVector<f64, N>) -> f64 {
    1. // Implement your function here
}


#[cfg(test)]
const HM6_A: [[f64; 6]; 4] = [
    [10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00],
];
#[cfg(test)]
const HM6_P: [[f64; 6]; 4] = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
];
#[cfg(test)]
const C: [f64; 4] = [1.0, 1.2, 3.0, 3.2];

#[cfg(test)]
fn hm6(x: &[f64]) -> f64 {
    debug_assert!(x.len() == 6);
    let mut sum = 0.0;

    for i in 0..4 {
        let a = HM6_A[i];
        let p = HM6_P[i];
        let mut d_i = 0.0;

        for i in 0..6 {
            d_i += a[i] * (x[i] - p[i]).powi(2);
        }

        sum += C[i] * (-d_i).exp();
    }

    -sum
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hm6_1() {
        let x = [0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.29398867, 0.62112999];
        let result = hm6(&x);
        (result, -2.872724123715199);
    }

    #[test]
    fn test_hm6_2() {
        let x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let result = hm6(&x);
        (result, -1.4069105761385299);
    }


    #[test]
    fn test_hm6_3() {
        let x = [0.1213, 0.2414, 0.1243, 0.345680344, 0.1237595, 0.1354856796];
        let result = hm6(&x);
        (result, -0.16821471453083264);
    }

    #[test]
    fn test_hm6_4() {
        let x = [0., 0.9009009009009009, 0.5961844197086474, 0.40540540540540543, 0.03685503127875094, 0.6756756756756757];
        let result = hm6(&x);
        (result, -0.12148933685954287);
    }

    #[test]
    fn test_hm6_5() {
        let x = [0., 0.6756756756756757, -11.029411764609979, -7.5, -0.6818180786573167, 1.3157894736842104];
        let result = hm6(&x);
        (result, -9.600116638678902e-298);
    }
}

