mod getalp;
mod ldlrk1;
mod ldlup;
mod ldldown;
mod minqsub;

use crate::minq::{getalp::getalp, minqsub::minqsub};
use nalgebra::{DMatrix, DVector};
use std::cmp::PartialEq;

#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
pub enum IerEnum {
    LocalMinimizerFound, //0
    UnboundedBelow, //1
    MaxitExceeded, //99
    InputError, //-1
}

pub fn minq(gam: f64, c: &[f64], G: &mut Vec<Vec<f64>>, xu: &[f64], xo: &[f64]) -> (Vec<f64>, f64, IerEnum) {
    let n = G.len();
    let mut x = vec![0.0; n];

    if G[0].len() != n {
        eprintln!("minq: Hessian has wrong dimensions");
        return (x, f64::NAN, IerEnum::InputError);
    }

    if c.len() != n || xu.len() != n || xo.len() != n {
        eprintln!("minq: Dimension mismatch in inputs");
        return (x, f64::NAN, IerEnum::InputError);
    }

    let (mut alp, mut alpu, mut alpo, mut lba, mut uba) = (0.0, 0.0, 0.0, true, true);

    let convex = false;

    // Ensure input consistency
    let mut ier = IerEnum::LocalMinimizerFound;

    let maxit = 3 * n;
    let nitrefmax = 3;

    x = (0..n)
        .map(|i| xu[i].max(0.0_f64.min(xo[i])))
        .collect::<Vec<f64>>();


    // Regularization for low rank problems
    let hpeps = 2.2204e-14;

    // Modify G to ensure numerical stability
    for i in 0..n {
        G[i][i] += hpeps * G[i][i];
    }
    // println!("{G:?}");

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
    let mut g;

    let c_na = DVector::from_vec(c.clone().to_vec());

    // Main loop: alternating coordinate and subspace searches
    loop {
        if x.iter().any(|&val| val.is_infinite()) {
            panic!("infinite x in minq {x:?}");
        }

        let x_na = DVector::from_vec(x.clone());
        let G_na = DMatrix::from_fn(G.len(), G[0].len(), |row, col| G[row][col]);

        // println!("input {G_na:?},\nx_na{x_na:?}\nc_na{c_na:?}");
        let g_na = G_na.clone() * x_na.clone() + c_na.clone();

        g = g_na.as_slice().to_vec();

        let fctnew = gam + (x_na.transpose().scale(0.5) * (c_na.clone() + g_na))[0];


        // println!("{improvement}, {nitref}, {nitrefmax}");
        if !improvement || nitref > nitrefmax || (nitref > 0 && nfree_old == nfree && fctnew >= fct) {
            // println!("--");
            ier = IerEnum::LocalMinimizerFound;
            break;
        } else if nitref == 0 {
            // println!("1234334345345");
            fct = fct.min(fctnew);
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
            alpu = xu[k] - x[k];
            alpo = xo[k] - x[k];

            // Find step size
            // !!!!! g[k] fails here
            // println!("PRIOR uba = {uba} {alpu}, {alpo}, {}, {}", g[k], q[k]);
            (alp, lba, uba, ier) = getalp(alpu, alpo, g[k], q[k]);
            if ier != IerEnum::LocalMinimizerFound {
                x = vec![0.0; n];
                x[k] = if lba { -1.0 } else { 1.0 };
                return (x, fct, ier);
            }

            let xnew = x[k] + alp;


            if lba || (xnew <= xu[k]) {
                if alpu != 0.0 {
                    x[k] = xu[k];
                    g.iter_mut().zip(q).for_each(|(g_i, q_i)| *g_i += alpu * q_i);
                    count = 0;
                }
                free[k] = false;
            } else if uba || (xnew >= xo[k]) {
                if alpo != 0.0 {
                    x[k] = xo[k];
                    g.iter_mut().zip(q).for_each(|(g_i, q_i)| *g_i += alpo * q_i);
                    count = 0;
                }
                free[k] = false;
            } else {
                if alp != 0.0 {
                    x[k] = xnew;
                    g.iter_mut().zip(q).for_each(|(g_i, q_i)| *g_i += alp * q_i);
                    free[k] = true;
                }
            }
        }

        nfree = free.iter().filter(|&&b| b).count() as i32;

        if unfix && nfree_old == nfree {
            let x_na = DVector::from_row_slice(&x.clone());
            g = (G_na * x_na + c_na.clone()).as_slice().to_vec();
            nitref += 1;
        } else {
            nitref = 0;
        }
        nfree_old = nfree;

        let x_na = DVector::from_row_slice(&x.clone());
        let g_na = DVector::from_row_slice(&g.clone());
        let gain_cs = fct - gam - x_na.scale(0.5).dot(&(c_na.clone() + g_na));
        // println!("gain_cs={gain_cs}");

        improvement = (gain_cs > 0.0) || !unfix;


        if nfree == 0 {
            unfix = true;
        } else {
            let mut subdone = false;
            // println!("PRE minqsub\nnsub={nsub}\nfree={free:?}\n{L:?}\n{dd:?}\n{K:?}\n{G:?}\nn={n}\ng={g:?}
            // \nx={x:?}\nxo={xo:?}\nxu={xu:?}\n{convex}\n\nnfree={nfree:?}\nunfix={unfix:?}\nalp={alp:?}\nalpu={alpu:?}\nalpo={alpo:?}\nlba={lba:?}\nuba={uba:?}\nier={ier:?}\nsubdone{subdone}");
            minqsub(&mut nsub, &mut free, &mut L, &mut dd, &mut K, &G, &n, &mut g,
                    &mut x, xo, xu, &convex, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

            if !subdone || (ier != IerEnum::LocalMinimizerFound) {
                return (x, fct, ier);
            }
        }
    }
    (x, fct, ier)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use approx::assert_relative_eq;

    #[test]
    fn test_real_mistake_0() {
        let gam = -3.2661659570240418;
        let c = vec![0.011491952485028996, 0.10990155244417238, -0.5975771816968101, -0.8069326056544889, 1.8713998467574868, -1.4958051414638351];
        let mut G = vec![
            vec![23.12798652584253, 0.08086473977917293, 1.7538162952525622, -1.9012829332291588, 1.7864612279290097, -0.7406818881433185],
            vec![0.08086473977917293, 18.576721298566618, -0.5909985456367551, 0.8013573491818613, -0.9992079198191761, 0.1810561706642408],
            vec![1.7538162952525622, -0.5909985456367551, 24.556579083791647, 3.371614208515673, -3.5009378170622605, 0.09958957165430643],
            vec![-1.9012829332291588, 0.8013573491818613, 3.371614208515673, 48.67847201840808, -1.0333246379471976, 0.9233898437170295],
            vec![1.7864612279290097, -0.9992079198191761, -3.5009378170622605, -1.0333246379471976, 89.37343113076405, 4.016171463395642],
            vec![-0.7406818881433185, 0.1810561706642408, 0.09958957165430643, 0.9233898437170295, 4.016171463395642, 48.170008410441206]
        ];
        let xo = [0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837];
        let xu = xo.iter().map(|&x| -x).collect::<Vec<f64>>();

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);
        // assert_eq!(x, vec![0.0019995286865852144, -0.007427824716643584, 0.018821593308224843, 0.01439613293535691, -0.021623304847149496, 0.03259925177469269]);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
        assert_eq!(fct, -3.3226086532809607);
    }


    #[test]
    fn test_global_return_0() {
        let gam = 0.0;
        let c = vec![-1.0, 20.0];
        let mut G = vec![vec![123.0, 26.0], vec![-0.3, -9.5]];
        let xu = vec![0.0, 0.0];
        let xo = vec![11.0, 22.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, vec![0.0, 22.0]);
        assert_relative_eq!(fct, -1859.0000000000514);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
    #[test]
    fn test_munqsub_return() {
        let gam = 1.0;
        let c = vec![1.0, 2.0];
        let mut G = vec![vec![1.0, 2.0], vec![3.0, 5.0]];
        let xu = vec![0.0, 0.0];
        let xo = vec![1.0, 2.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, vec![0.0, 0.0]);
        assert_relative_eq!(fct, 1.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond() {
        let gam = 2.0;
        let c = vec![1.0, 2.0, 3.0];
        let mut G = vec![vec![1.0, 2.0, -4.0], vec![3.0, 5.0, -1.0], vec![0.0, -3.0, -10.0]];
        let xu = vec![-10.0, -10.0, -3.0];
        let xo = vec![1.0, 2.0, 4.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_abs_diff_eq!(x.as_slice(), vec![1.0, -0.2, 4.0].as_slice(), epsilon = 1e-6);
        assert_relative_eq!(fct, 2.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond_2() {
        let gam = 200.0;
        let c = vec![1.0, 2.0, 3.0];
        let mut G = vec![vec![1.0, 2.0, 4.0], vec![3.0, 5.0, -1.0], vec![0.0, -3.0, -10.0]];
        let xu = vec![0.0, 0.0, -3.0];
        let xo = vec![1.0, 2.0, 4.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_abs_diff_eq!(x.as_slice(),vec![0.0, 0.4, 4.0].as_slice(), epsilon = 1e-6);
        assert_relative_eq!(fct, 200.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_global_return_4() {
        let gam = -200.0;
        let c = vec![-1.0, -2.0];
        let mut G = vec![vec![-1.0, -2.0], vec![5.0, -1.0]];
        let xu = vec![0.0, 0.0];
        let xo = vec![10.0, 20.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, vec![10.0, 0.0]);
        assert_relative_eq!(fct, -260.00000000000114);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
}
