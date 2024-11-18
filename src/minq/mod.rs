mod getalp;
mod ldlrk1;
mod ldlup;
mod ldldown;
mod minqsub;

use crate::minq::{getalp::getalp, minqsub::minqsub};
use std::cmp::PartialEq;


#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
pub enum IerEnum {
    LocalMinimizerFound, //0
    UnboundedBelow, //1
    MaxitExceeded, //99
    InputError, //-1
}

pub fn minq(gam: f64, c: Vec<f64>, mut G: Vec<Vec<f64>>, xu: Vec<f64>, xo: Vec<f64>) -> (Vec<f64>, f64, IerEnum) {
    let (mut alp, mut alpu, mut alpo, mut lba, mut uba) = (0.0, 0.0, 0.0, true, true);
    let convex = false;
    let n = G.len();

    // Ensure input consistency
    let mut ier = IerEnum::LocalMinimizerFound;

    if G[0].len() != n {
        ier = IerEnum::InputError;
        eprintln!("minq: Hessian has wrong dimensions");
        return (vec![f64::NAN; n], f64::NAN, ier);
    }

    if c.len() != n || xu.len() != n || xo.len() != n {
        ier = IerEnum::InputError;
        eprintln!("minq: Dimension mismatch in inputs");
        return (vec![f64::NAN; n], f64::NAN, ier);
    }

    if ier == IerEnum::InputError {
        return (vec![f64::NAN; n], f64::NAN, ier);
    }

    let maxit = 3 * n;
    let nitrefmax = 3;

    // Force starting point into the box
    let mut xx: Vec<f64> = (0..n)
        .map(|i| xu[i].max(0.0_f64.min(xo[i])))
        .collect();

    // Regularization for low rank problems
    let eps = 2.2204e-16;
    let hpeps = 100.0 * eps;

    // Modify G to ensure numerical stability
    for i in 0..n {
        G[i][i] += hpeps * G[i][i];
    }

    // Initialize variables for LDL^T factorization
    let mut K = vec![false; n];
    let mut L = vec![vec![0.0; n]; n];
    for i in 0..n {
        L[i][i] = 1.0; // Initialize L to the identity matrix
    }
    let mut dd = vec![1.0; n];

    // Initialize other variables
    let mut free = vec![false; n];
    let mut nfree: i32 = 0;
    let mut nfree_old: i32 = -1;

    let mut fct = f64::INFINITY;
    let mut nsub = 0;
    let mut unfix = true;
    let mut nitref = 0;
    let mut improvement = true;

    // Main loop: alternating coordinate and subspace searches
    loop {
        if xx.iter().any(|&val| val.is_infinite()) {
            panic!("infinite xx in minq");
        }

        let mut g: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            g[i] = c[i] + G[i].iter().zip(&xx).map(|(&Gi, &xi)| Gi * xi).sum::<f64>();
        }

        // fctnew COMPUTE
        // Step 1: Element-wise addition of vectors c and g
        let mut c_plus_g: Vec<f64> = vec![0.0; c.len()];
        for i in 0..c.len() {
            c_plus_g[i] = c[i] + g[i];
        }

        // Step 2: Compute the dot product of 0.5 * xx and c_plus_g
        let mut dot_product: f64 = 0.0;
        for i in 0..xx.len() {
            dot_product += 0.5 * xx[i] * c_plus_g[i];
        }

        // Step 3: Final result
        let fctnew: f64 = gam + dot_product;


        if !improvement || nitref > nitrefmax || nitref > 0 && nfree_old == nfree && fctnew >= fct {
            ier = IerEnum::LocalMinimizerFound;
            break;
        } else if nitref == 0 {
            fct = fctnew.min(fct);
        } else {
            fct = fctnew;
        }

        if nitref == 0 && nsub >= maxit {
            ier = IerEnum::MaxitExceeded;
            break;
        }

        // Coordinate search
        let mut count: usize = 0;
        let mut k: i32 = -1;

        let mut x = xx.clone();
        loop {
            while count <= n {
                count += 1;
                if k == (n - 1).try_into().unwrap() {
                    k = -1; // reset k to -1 for python array first index
                }
                k = k + 1;  //# increase k
                if free[k as usize] || unfix {
                    break;
                }
            }

            if count > n {
                break;
            }

            let k = k as usize;
            let q = G.iter().map(|row| row[k]).collect::<Vec<f64>>();
            let alpu = xu[k] - x[k];
            let alpo = xo[k] - x[k];

            // Find step size
            let (alp, lba, uba, ier) = getalp(alpu, alpo, g[k], q[k]);
            if ier != IerEnum::LocalMinimizerFound {
                let mut x = vec![0.0; n];
                x[k] = if lba { -1.0 } else { 1.0 };
                return (x, fct, ier);
            }

            let xnew = x[k] + alp;
            if lba || xnew <= xu[k] {
                if alpu != 0.0 {
                    x[k] = xu[k];
                    for i in 0..n {
                        g[i] += alpu * q[i];
                    }
                    count = 0;
                }
                free[k] = false;
            } else if uba || xnew >= xo[k] {
                if alpo != 0.0 {
                    x[k] = xo[k];
                    for i in 0..n {
                        g[i] += alpo * q[i];
                    }
                    count = 0;
                }
                free[k] = false;
            } else {
                if alp != 0.0 {
                    x[k] = xnew;
                    for i in 0..n {
                        g[i] += alp * q[i];
                    }
                    free[k] = true;
                }
            }
        }

        nfree = free.iter().filter(|&&b| b).count() as i32;
        if unfix && nfree_old == nfree {
            for i in 0..n {
                g[i] = G[i].iter().zip(&x).map(|(&Gi, &xi)| Gi * xi).sum::<f64>() + c[i];
            }
            nitref += 1;
        } else {
            nitref = 0;
        }
        nfree_old = nfree;

        let gain_cs = fct - gam - 0.5 * x.iter().zip(&g).map(|(&x_i, &g_i)| x_i * g_i).sum::<f64>();
        improvement = gain_cs > 0.0 || !unfix;

        xx = x.clone();

        if nfree == 0 {
            unfix = true;
        } else {
            let mut subdone = false;
            // println!("{nsub}\n{free:?}\n{L:?}\n{dd:?}\n{K:?}\n{G:?}\n{g:?}\
            // \n{x:?}\n{xo:?}\n{xu:?}\n{convex}\n{xx:?}\n{nfree:?}\n{unfix:?}\n{alp:?}");
            minqsub(
                &mut nsub, &mut free, &mut L, &mut dd, &mut K, &G, &n, &mut g,
                &mut x, &xo, &xu, &convex, &mut xx, &mut nfree, &mut unfix,
                &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone,
            );

            if !subdone || (ier != IerEnum::LocalMinimizerFound) {
                return (xx, fct, ier);
            }
        }
    }
    (xx, fct, ier)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;
    use approx::assert_abs_diff_eq;
    use approx::assert_relative_eq;

    #[test]
    fn test_global_return_0() {
        let gam = 0.0;
        let c = vec![-1.0, 20.0];
        let G = vec![vec![123.0, 26.0], vec![-0.3, -9.5]];
        let xu = vec![0.0, 0.0];
        let xo = vec![11.0, 22.0];

        let (x, fct, ier) = minq(gam, c, G, xu, xo);

        assert_eq!(x, vec![0.0, 22.0]);
        assert_relative_eq!(fct, -1859.0000000000514);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
    #[test]
    fn test_munqsub_return() {
        let gam = 1.0;
        let c = vec![1.0, 2.0];
        let G = vec![vec![1.0, 2.0], vec![3.0, 5.0]];
        let xu = vec![0.0, 0.0];
        let xo = vec![1.0, 2.0];

        let (x, fct, ier) = minq(gam, c, G, xu, xo);

        assert_eq!(x, vec![0.0, 0.0]);
        assert_relative_eq!(fct, 1.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond() {
        let gam = 2.0;
        let c = vec![1.0, 2.0, 3.0];
        let G = vec![vec![1.0, 2.0, -4.0], vec![3.0, 5.0, -1.0], vec![0.0, -3.0, -10.0]];
        let xu = vec![-10.0, -10.0, -3.0];
        let xo = vec![1.0, 2.0, 4.0];

        let (x, fct, ier) = minq(gam, c, G, xu, xo);

        assert_abs_diff_eq!(x.as_slice(), vec![1.0, -0.2, 4.0].as_slice(), epsilon = 1e-6);
        assert_relative_eq!(fct, 2.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond_2() {
        let gam = 200.0;
        let c = vec![1.0, 2.0, 3.0];
        let G = vec![vec![1.0, 2.0, 4.0], vec![3.0, 5.0, -1.0], vec![0.0, -3.0, -10.0]];
        let xu = vec![0.0, 0.0, -3.0];
        let xo = vec![1.0, 2.0, 4.0];

        let (x, fct, ier) = minq(gam, c, G, xu, xo);

        assert_abs_diff_eq!(x.as_slice(),vec![0.0, 0.4, 4.0].as_slice(), epsilon = 1e-6);
        assert_relative_eq!(fct, 200.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_global_return_4() {
        let gam = -200.0;
        let c = vec![-1.0, -2.0];
        let G = vec![vec![-1.0, -2.0], vec![5.0, -1.0]];
        let xu = vec![0.0, 0.0];
        let xo = vec![10.0, 20.0];

        let (x, fct, ier) = minq(gam, c, G, xu, xo);

        assert_eq!(x, vec![10.0, 0.0]);
        assert_relative_eq!(fct, -260.00000000000114);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
}
