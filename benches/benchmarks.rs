// In order to run write in console: cargo bench

use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;


fn bench_test_1(c: &mut Criterion) {
    let mut group = c.benchmark_group("One core bench");

    group.measurement_time(Duration::from_secs(60))
        .warm_up_time(Duration::from_secs(10))
        .sample_size(10)
        .noise_threshold(0.05);

    // Basic benchmark
    group.bench_function("test_1_basic", |b| {
        b.iter(|| {
            bench_cpu_parallel::test_bench_0();
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .significance_level(0.01)
        .confidence_level(0.95);    // Added confidence level for better statistical validity
    targets = bench_test_1
}
criterion_main!(benches);




pub mod bench_cpu_parallel {
    use nalgebra::SMatrix;
    use Rust_MCS::*;

    pub fn test_bench_0() {
        const SMAX: usize = 1_000;
        let nf: usize = 1_000_000;
        let stop = vec![200., f64::NEG_INFINITY];
        let iinit = IinitEnum::Zero;
        let local = 5;
        let gamma = 2e-7;
        let u = [-6.506834377244, -0.5547628574185793, -0.4896101151981129, -4.167584856725679, -6.389642504060847, -5.528716818248636];
        let v = [0.6136260223676221, 3.3116327823744762, 1.815553122672147, 0.06874148889830267, 0.7052383406994288, 0.93288192217477];
        let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.8277209419820275, 0.35275501307855395, 0.252012633495165, 0.5667951361102919, 0.19630620226079598, 0.0648101272618129,
            0.5081006457816327, 0.2660878681097819, 0.09782770288876363, 0.43830363933100314, 0.4746456902322366, 0.4661411009402323,
            0.19980055789123086, 0.4986248326438728, 0.012620127489665345, 0.19089710870186494, 0.4362731501809838, 0.6063090941013247,
            0.7310040262066118, 0.4204623417897273, 0.8664287267092771, 0.9742278318360923, 0.6386093993614557, 0.27981042978028847,
            0.6800547697745852, 0.5742073425616279, 0.8821852581714857, 0.13408110711794174, 0.04935188705985705, 0.9987572981515097,
            0.6187202250393025, 0.1377423026724791, 0.8070825819627165, 0.2817037864244687, 0.5842187774516107, 0.09751501025007547
        ]);

        let _ = mcs::<SMAX, 6>(&u, &v, nf, &stop, iinit, local, gamma, &hess);
    }
}