#[cfg(test)]
use nalgebra::SVector;

#[cfg(test)] // hm6 function
const HM6_A: [[f64; 6]; 4] = [
    [10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00],
];

#[cfg(test)] // hm6 function
const HM6_P: [[f64; 6]; 4] = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
];

#[cfg(test)] // hm6 function
const C: [f64; 4] = [1.0, 1.2, 3.0, 3.2];


#[cfg(test)] // hm6 function
pub(crate) fn hm6<const N: usize>(x: &SVector<f64, N>) -> f64 {
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
