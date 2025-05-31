use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{SMatrix, SVector};
use Rust_MCS::*;


criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::new(60, 0));
    targets = bench_mcs
}
criterion_main!(benches);


fn bench_mcs(c: &mut Criterion) {
    // Matlab equivalent test
    // clearvars;
    // clear global;
    //
    // path(path,'jones');
    // fcn = "feval"; % do not change
    // data = "hm6"; %  do not change
    // prt = 0; % do not change
    // iinit = 0; % do not change; Simple initialization list aka IinitEnum::Zero here
    // u = [0; 0; 0; 0; 0; 0];
    // v = [1; 1; 1; 1; 1; 1];
    // smax = 100;
    // nf = 1000000;
    // stop = [1000]; % nsweeps
    // local = 100;
    // gamma= 2e-10;
    // hess = ones(6,6); % 6x6 matrix for hm6
    //
    // format long g;
    // [xbest,fbest,xmin,fmi,ncall,ncloc,flag]=mcs(fcn,data,u,v,prt,smax,nf,stop,iinit,local,gamma,hess)

    let u = SVector::<f64, 6>::from_row_slice(&[0.; 6]); // lower bounds
    let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]); // upper bounds

    let nsweeps = 1_000;  // maximum number of sweeps
    let nf = 1_000_000;   // maximum number of function evaluations

    let local = 100;   // local search level
    let gamma = 2e-10;  // acceptable relative accuracy for local search
    let smax = 1_000; // number of levels used
    let hess = SMatrix::<f64, 6, 6>::repeat(1.);    // sparsity pattern of Hessian

    c.bench_function("bench_mcs", |b| b.iter(|| mcs::<6>(hm6, &u, &v, nsweeps, nf, local, gamma, smax, &hess).unwrap()));
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
