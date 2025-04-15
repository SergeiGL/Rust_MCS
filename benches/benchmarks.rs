use nalgebra::{SMatrix, SVector};
use Rust_MCS::*;

#[divan::bench(
    max_time = 200, // seconds
)]
fn bench_mcs() {
    let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]);
    let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
    const SMAX: usize = 2_000;
    let nsweeps = 1_000;  // maximum number of sweeps
    let nf = 1_000_000; // maximum number of function evaluations
    let local = 200;
    let gamma = 2e-12;
    let hess = SMatrix::<f64, 6, 6>::repeat(1.);

    // Use black_box to prevent the compiler from optimizing the function call away
    divan::black_box(mcs::<SMAX, 6>(hm6, &u, &v, nsweeps, nf, local, gamma, &hess).unwrap());
}


fn main() {
    // This runs all benchmarks annotated with #[divan::bench]
    divan::main();
    // ╰─ bench_mcs  3.868 s       │ 4.602 s       │ 4.149 s       │ 4.18 s        │ 60      │ 60
}


const HM6_A: [[f64; 6]; 4] = [
    [10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00],
];

const HM6_P: [[f64; 6]; 4] = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
];

const C: [f64; 4] = [1.0, 1.2, 3.0, 3.2];


fn hm6<const N: usize>(x: &SVector<f64, N>) -> f64 {
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
