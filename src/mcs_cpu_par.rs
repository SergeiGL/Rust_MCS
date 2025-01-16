// use rayon::prelude::*;
// use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
// use std::sync::Arc;
//
// pub fn mcs_cpu_par<const SMAX: usize, const N: usize>(
//     n_processes: usize,
//     u: &[f64; N],
//     v: &[f64; N],
//     nf: usize,
//     stop: &Vec<f64>,
//     iinit: IinitEnum,
//     local: usize,
//     gamma: f64,
//     hess: &SMatrix<f64, N, N>,
// ) -> (
//     [f64; N],       // xbest
//     f64,            // fbest
//     usize,          // ncall
//     usize,          // ncloc
//     bool,           // flag
// ) where
//     Const<N>: DimMin<Const<N>, Output=Const<N>>,
// {
//     let (u_split, v_split, nboxes) = split_box::split_box(u, v, n_processes);
//     println!("Box split into {} boxes; utilizing {} processes", v_split.len(), nboxes);
//
//     // Use Arc for thread-safe sharing of atomic values
//     let ncall_atomic = Arc::new(AtomicUsize::new(0));
//     let ncloc_atomic = Arc::new(AtomicUsize::new(0));
//     let flag_atomic = Arc::new(AtomicBool::new(true));
//
//     let nf_per_process = nf / nboxes + 1;
//
//     let results: Vec<([f64; N], f64)> = (0..nboxes)
//         .into_par_iter()
//         .map(|i| {
//             let ncall_atomic = Arc::clone(&ncall_atomic);
//             let ncloc_atomic = Arc::clone(&ncloc_atomic);
//             let flag_atomic = Arc::clone(&flag_atomic);
//
//             let (xbest, fbest, _, _, ncall, ncloc, flag) = mcs::<SMAX, N>(
//                 &u_split[i],
//                 &v_split[i],
//                 nf_per_process,
//                 stop,
//                 iinit,
//                 local,
//                 gamma,
//                 hess,
//             );
//
//             // Update atomic values
//             if !flag {
//                 flag_atomic.store(false, Ordering::Relaxed);
//             }
//             ncall_atomic.fetch_add(ncall, Ordering::Relaxed);
//             ncloc_atomic.fetch_add(ncloc, Ordering::Relaxed);
//
//             (xbest, fbest)
//         })
//         .collect();
//
//     // Find the best result using iterator methods
//     let (xbest_global, fbest_global) = results
//         .into_iter()
//         .min_by(|a, b| a.1.total_cmp(&b.1))
//         .unwrap_or(([0.0; N], f64::INFINITY));
//
//     (
//         xbest_global,
//         fbest_global,
//         ncall_atomic.load(Ordering::Relaxed),
//         ncloc_atomic.load(Ordering::Relaxed),
//         flag_atomic.load(Ordering::Relaxed),
//     )
// }
//
// #[cfg(test)]
// mod test_cpu_parallel {
//     use super::*;
//     static n_processes: usize = 4;
//
//     #[test]
//     fn test_0() {
//         const SMAX: usize = 1_000;
//         let nf = 10_000_000;
//         let stop = vec![7., f64::NEG_INFINITY];
//         let iinit = IinitEnum::Zero;
//         let local = 7;
//         let gamma = 2e-7;
//         let u = [-6.506834377244, -0.5547628574185793, -0.4896101151981129, -4.167584856725679, -6.389642504060847, -5.528716818248636];
//         let v = [0.6136260223676221, 3.3116327823744762, 1.815553122672147, 0.06874148889830267, 0.7052383406994288, 0.93288192217477];
//         let hess = SMatrix::<f64, 6, 6>::from_row_slice(&[
//             0.8277209419820275, 0.35275501307855395, 0.252012633495165, 0.5667951361102919, 0.19630620226079598, 0.0648101272618129,
//             0.5081006457816327, 0.2660878681097819, 0.09782770288876363, 0.43830363933100314, 0.4746456902322366, 0.4661411009402323,
//             0.19980055789123086, 0.4986248326438728, 0.012620127489665345, 0.19089710870186494, 0.4362731501809838, 0.6063090941013247,
//             0.7310040262066118, 0.4204623417897273, 0.8664287267092771, 0.9742278318360923, 0.6386093993614557, 0.27981042978028847,
//             0.6800547697745852, 0.5742073425616279, 0.8821852581714857, 0.13408110711794174, 0.04935188705985705, 0.9987572981515097,
//             0.6187202250393025, 0.1377423026724791, 0.8070825819627165, 0.2817037864244687, 0.5842187774516107, 0.09751501025007547
//         ]);
//
//         let (_xbest, fbest, _ncall, _ncloc, flag) = mcs_cpu_par::<SMAX, 6>(n_processes, &u, &v, nf, &stop, iinit, local, gamma, &hess);
//
//         assert!(fbest <= -2.343978461109969);
//     }
// }
//
