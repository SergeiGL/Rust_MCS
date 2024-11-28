use crate::minq::getalp::getalp;
use crate::minq::ldldown::ldldown;
use crate::minq::ldlup::ldlup;
use crate::minq::IerEnum;
use nalgebra::{DMatrix, DVector};
use std::f64;

pub fn minqsub(
    nsub: &mut usize,
    free: &mut Vec<bool>,
    L: &mut Vec<Vec<f64>>,
    d: &mut Vec<f64>,
    K: &mut Vec<bool>,
    G: &Vec<Vec<f64>>,
    n: &usize,
    g: &mut Vec<f64>,
    x: &mut Vec<f64>,
    xo: &[f64],
    xu: &[f64],
    convex: &bool,
    nfree: &mut i32,
    unfix: &mut bool,
    alp: &mut f64,
    alpu: &mut f64,
    alpo: &mut f64,
    lba: &mut bool,
    uba: &mut bool,
    ier: &mut IerEnum,
    subdone: &mut bool,
) {
    *nsub += 1;
    let eps: f64 = 2.2204e-16;

    // Initialize empty p vector
    let mut p: Vec<f64> = vec![0.0; *n];

    // Downdate factorization, only for indices in freel
    let freelk_idxs: Vec<usize> = free
        .iter()
        .zip(K.iter())
        .enumerate()
        .filter(|&(_, (&f, &k))| f && !k)
        .map(|(i, _)| i)
        .collect();
    for &j in &freelk_idxs {
        ldldown(L, d, j);
        K[j] = false;
    }

    // Update factorization or find indefinite search direction
    let mut definite = true;
    let freeuk_idxs: Vec<usize> = free
        .iter()
        .zip(K.iter())
        .enumerate()
        .filter(|&(_, (&f, &k))| f && !k)
        .map(|(i, _)| i)
        .collect();

    for &j in &freeuk_idxs {
        p = vec![0.0; *n];
        if *n > 1 {
            for (i, &ki) in K.iter().enumerate() {
                if ki {
                    p[i] = G[i][j];
                }
            }
        }
        p[j] = G[j][j];

        // println!(
        //     "ldlup BEFORE\nL={:?}\nd={:?}\nj={}\np={:?}",
        //     L, d, j, p
        // );

        p = ldlup(L, d, j, &p);

        // println!(
        //     "ldlup AFTER\nL={:?}\nd={:?}\nj={}\np={:?}",
        //     L, d, j, p
        // );

        definite = p.is_empty();
        if !definite {
            break;
        }
        K[j] = true;
    }

    // println!("p={p:?}");
    if definite {
        p = vec![0.0; *n];
        for (i, &ki) in K.iter().enumerate() {
            if ki {
                p[i] = g[i];
            }
        }

        // println!("p before fight =\n {p:?}");
        let mut l_flat: Vec<f64> = Vec::with_capacity((*n) * (*n));
        for i in 0..L.len() {
            for j in 0..L[0].len() {
                l_flat.push(L[j][i]);
            }
        }

        // Create matrix and vectors
        let l_matrix = DMatrix::from_vec(L.len(), L[0].len(), l_flat);
        let p_vec = DVector::from_vec(p.clone());
        let d_vec = DVector::from_vec(d.clone());


        // println!("l_matrix {l_matrix:?};\np_vec {p_vec:?};\nd_vec {d_vec:?}");
        let mut LPsolve = l_matrix.clone().lu().solve(&p_vec).expect("Failed to solve L matrix");
        // println!("LPsolve before division: {:?}", LPsolve);

        LPsolve = LPsolve.component_div(&d_vec);

        // println!("LPsolve after division: {:?}", LPsolve);

        // Solve the transpose system
        let mut solve_2 = l_matrix.transpose().lu().solve(&LPsolve).expect("Failed to solve transpose L matrix");
        solve_2.scale_mut(-1.0);

        // Print the result
        // println!("p after fight =\n {:?}", solve_2);

        // Assigning solve_2 back to p
        p.copy_from_slice(solve_2.as_slice());
    }

    // Set tiny entries to zero (to mimic (x + p) - x in Python)
    let p_shifted: Vec<f64> = p.iter().enumerate().map(|(i, &pi)| pi + x[i]).collect();
    p = p_shifted
        .iter()
        .zip(x.iter())
        .map(|(ps, xi)| ps - xi)
        .collect();

    // Find indices where p is not zero
    let ind: Vec<usize> = p
        .iter()
        .enumerate()
        .filter(|(_, &pi)| pi != 0.0)
        .map(|(i, _)| i)
        .collect();

    if ind.is_empty() {
        *unfix = true;
        *subdone = false;
        return;
    }


    (*alpu, *alpo) = (f64::NEG_INFINITY, f64::INFINITY);
    let pp: Vec<f64> = ind.iter().map(|&i| p[i]).collect();
    let oo: Vec<f64> = ind.iter().map(|&i| xo[i] - x[i]).collect::<Vec<f64>>().iter().zip(&pp).map(|(&u_i, &p_i)| u_i / p_i).collect::<Vec<f64>>();
    let uu: Vec<f64> = ind.iter().map(|&i| xu[i] - x[i]).collect::<Vec<f64>>().iter().zip(&pp).map(|(&u_i, &p_i)| u_i / p_i).collect::<Vec<f64>>();

    for ((&pi, &o_i), &u_i) in pp.iter().zip(oo.iter()).zip(uu.iter()) {
        if pi < 0.0 {
            *alpu = alpu.max(o_i);
        } else if pi > 0.0 {
            *alpu = alpu.max(u_i);
        }

        if pi > 0.0 {
            *alpo = alpo.min(o_i);
        } else if pi < 0.0 {
            *alpo = alpo.min(u_i);
        }
    }

    // Check if alpo and alpu are valid
    if *alpo <= 0.0 || *alpu >= 0.0 {
        panic!("programming error: no alp");
    }

    // Find step size
    let gTp: f64 = g.iter().zip(p.iter()).map(|(gi, pi)| gi * pi).sum();
    let agTp: f64 = g.iter().map(|g_val| g_val.abs()).sum::<f64>() * p.iter().map(|p_val| p_val.abs()).sum::<f64>();
    let gTp_corr = if gTp.abs() < 100.0 * eps * agTp {
        0.0
    } else {
        gTp
    };

    // Compute pTGp
    let G_matrix = DMatrix::from_fn(G.len(), G[0].len(), |i, j| G[i][j]);
    let p_vector = DVector::from_vec(p.clone());
    let mut pTGp = (p_vector.transpose() * &G_matrix * &p_vector)[(0, 0)];

    if *convex {
        pTGp = pTGp.max(0.0);
    }

    if !definite && pTGp > 0.0 {
        pTGp = 0.0;
    }

    (*alp, *lba, *uba, *ier) = getalp(*alpu, *alpo, gTp_corr, pTGp);

    if *ier != IerEnum::LocalMinimizerFound {
        if *lba {
            *x = p.iter().map(|&p_i| -p_i).collect();
        } else {
            *x = p.clone();
        }
        return;
    }

    *unfix = !(*lba || *uba);

    // Update xx=x (xx is always = x)
    for k in 0..ind.len() {
        let ik = ind[k];
        if *alp == uu[k] {
            x[ik] = xu[ik];
            free[ik] = false;
        } else if *alp == oo[k] {
            x[ik] = xo[ik];
            free[ik] = false;
        } else {
            x[ik] += *alp * p[ik];
        }

        if x[ik].is_infinite() {
            panic!("Infinite xx in minq");
        }
    }

    *nfree = free.iter().filter(|&&f| f).count() as i32;
    *subdone = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_miskake_1() {
        let mut nsub = 1;
        let mut free = vec![true; 6];
        let mut L = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.07583091132005203, -0.03214451416643742, 1.0, 0.0, 0.0, 0.0],
            vec![-0.08220702356016504, 0.043496227547229385, 0.1451281097853027, 1.0, 0.0, 0.0],
            vec![0.07724240179458904, -0.054125235207729994, -0.1503305948215706, -0.006468139053208312, 1.0, 0.0],
            vec![-0.032025351074797884, 0.009885958049664674, 0.006624191581076727, 0.01732330782068277, 0.046407734608255605, 1.0]
        ];
        let mut d = vec![23.127986525843045, 18.57643856292412, 24.40439112304386, 47.97301864636891, 88.62749123902528, 47.93812981294381];
        let mut K = vec![true; 6];
        let mut G = vec![
            vec![23.127986525843045, 0.08086473977917293, 1.7538162952525622, -1.9012829332291588, 1.7864612279290097, -0.7406818881433185],
            vec![0.08086473977917293, 18.57672129856703, -0.5909985456367551, 0.8013573491818613, -0.9992079198191761, 0.1810561706642408],
            vec![1.7538162952525622, -0.5909985456367551, 24.55657908379219, 3.371614208515673, -3.5009378170622605, 0.09958957165430643],
            vec![-1.9012829332291588, 0.8013573491818613, 3.371614208515673, 48.67847201840916, -1.0333246379471976, 0.9233898437170295],
            vec![1.7864612279290097, -0.9992079198191761, -3.5009378170622605, -1.0333246379471976, 89.37343113076604, 4.016171463395642],
            vec![-0.7406818881433185, 0.1810561706642408, 0.09958957165430643, 0.9233898437170295, 4.016171463395642, 48.17000841044228]
        ];
        let n = 6;
        let mut g = vec![6.938893903907228e-18, 0.0, 1.1102230246251565e-16, 0.0, -2.220446049250313e-16, 0.0];
        let mut x = vec![0.001999528686585215, -0.007427824716643584, 0.018821593308224843, 0.01439613293535691, -0.0216233048471495, 0.032599251774692695];
        let xo = vec![0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837];
        let xu = vec![-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837];
        let convex = false;
        let mut nfree = 6;
        let mut unfix = true;
        let mut alp = 1.9926847409028064e-19;
        let mut alpu = -0.4050781679349764;
        let mut alpo = 0.33987966438559103;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &mut G, &n, &mut g, &mut x, &xo, &xu, &convex, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        assert_eq!(nsub, 2);
        assert_eq!(free, vec![true; 6]);
        assert_eq!(L, vec![vec![1., 0., 0., 0., 0., 0.],
                           vec![0.0034964020620132778, 1., 0., 0., 0., 0.],
                           vec![0.07583091132005203, -0.03214451416643742, 1., 0., 0., 0.],
                           vec![-0.08220702356016504, 0.043496227547229385, 0.1451281097853027, 1., 0., 0.],
                           vec![0.07724240179458904, -0.054125235207729994, -0.1503305948215706, -0.006468139053208312, 1., 0.],
                           vec![-0.032025351074797884, 0.009885958049664674, 0.006624191581076727, 0.01732330782068277, 0.046407734608255605, 1.]]);
        assert_eq!(d, vec![23.127986525843045, 18.57643856292412, 24.40439112304386, 47.97301864636891, 88.62749123902528, 47.93812981294381]);
        assert_eq!(K, vec![true; 6]);
        assert_eq!(G, G.clone());
        assert_eq!(n, 6);
        assert_eq!(g, vec![0.000000000000000006938893903907228, 0., 0.00000000000000011102230246251565, 0., -0.0000000000000002220446049250313, 0.]);
        assert_eq!(x, vec![0.001999528686585215, -0.007427824716643584, 0.01882159330822484, 0.01439613293535691, -0.021623304847149496, 0.032599251774692695]);
        assert_eq!(xo, vec![0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837]);
        assert_eq!(xu, vec![-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837]);
        assert_eq!(convex, false);
        assert_eq!(nfree, 6);
        assert_eq!(alp, 0.7938353009736487);
        assert_eq!(alpu, -8.934536499651395e+16);
        assert_eq!(alpo, 1.0181035157598803e+17);
        assert_eq!(lba, false);
        assert_eq!(uba, false);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
        assert_eq!(unfix, true);
        assert_eq!(subdone, true);
    }


    #[test]
    fn test_real_miskake_0() {
        let mut nsub = 0;
        let mut free = vec![true, true, true, true, true, true];
        let mut L = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ];
        let mut d = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut K = vec![false, false, false, false, false, false];
        let mut G = vec![
            vec![23.127986525843045, 0.08086473977917293, 1.7538162952525622, -1.9012829332291588, 1.7864612279290097, -0.7406818881433185],
            vec![0.08086473977917293, 18.57672129856703, -0.5909985456367551, 0.8013573491818613, -0.9992079198191761, 0.1810561706642408],
            vec![1.7538162952525622, -0.5909985456367551, 24.55657908379219, 3.371614208515673, -3.5009378170622605, 0.09958957165430643],
            vec![-1.9012829332291588, 0.8013573491818613, 3.371614208515673, 48.67847201840916, -1.0333246379471976, 0.9233898437170295],
            vec![1.7864612279290097, -0.9992079198191761, -3.5009378170622605, -1.0333246379471976, 89.37343113076604, 4.016171463395642],
            vec![-0.7406818881433185, 0.1810561706642408, 0.09958957165430643, 0.9233898437170295, 4.016171463395642, 48.17000841044228]
        ];
        let n = 6;
        let mut g = vec![-0.045952748319448775, 0.023404204756152784, 0.1232952812014726, 0.05044112085181421, 0.13007152652763362, 0.0];

        let mut x = vec![-0.0004968851253950693, -0.005913926908419606, 0.024227865888591913, 0.014976641791235562, -0.019873081026496084, 0.03238694555577046];
        let xo = vec![0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837];
        let xu = vec![-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837];
        let convex = false;
        let mut nfree = 6;
        let mut unfix = true;
        let mut alp = 0.03238694555577046;
        let mut alpu = -0.3724789161602837;
        let mut alpo = 0.3724789161602837;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &mut G, &n, &mut g, &mut x, &xo, &xu, &convex, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        assert_eq!(nsub, 1);
        assert_eq!(free, vec![true; 6]);
        assert_eq!(L, vec![vec![1., 0., 0., 0., 0., 0.],
                           vec![0.0034964020620132778, 1., 0., 0., 0., 0.],
                           vec![0.07583091132005203, -0.03214451416643742, 1., 0., 0., 0.],
                           vec![-0.08220702356016504, 0.043496227547229385, 0.1451281097853027, 1., 0., 0.],
                           vec![0.07724240179458904, -0.054125235207729994, -0.1503305948215706, -0.006468139053208312, 1., 0.],
                           vec![-0.032025351074797884, 0.009885958049664674, 0.006624191581076727, 0.01732330782068277, 0.046407734608255605, 1.]]);
        assert_eq!(d, vec![23.127986525843045, 18.57643856292412, 24.40439112304386, 47.97301864636891, 88.62749123902528, 47.93812981294381]);
        assert_eq!(K, vec![true; 6]);
        assert_eq!(G, G.clone());
        assert_eq!(n, 6);
        assert_eq!(g, vec![-0.045952748319448775, 0.023404204756152784, 0.1232952812014726, 0.05044112085181421, 0.13007152652763362, 0.0]);
        assert_eq!(x, vec![0.0019995286865852144, -0.007427824716643584, 0.01882159330822484, 0.01439613293535691, -0.021623304847149496, 0.032599251774692695]);
        assert_eq!(xo, vec![0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837]);
        assert_eq!(xu, vec![-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837]);
        assert_eq!(convex, false);
        assert_eq!(nfree, 6);
        assert_eq!(alp, 0.9999999999999998);
        assert_eq!(alpu, -62.99290436226673);
        assert_eq!(alpo, 71.95577685100348);
        assert_eq!(lba, false);
        assert_eq!(uba, false);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
        assert_eq!(unfix, true);
        assert_eq!(subdone, true);
    }

    #[test]
    fn test_minqsub_0() {
        let mut nsub = 0;
        let mut free = vec![true; 10];
        let mut L = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ];
        let mut d = vec![1.0; 10];
        let mut K = vec![false; 10];
        let mut G = vec![
            vec![-2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            vec![-3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            vec![-3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            vec![-3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            vec![-3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            vec![-3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0],
            vec![-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0],
            vec![-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0],
            vec![-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0],
            vec![-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0]
        ];
        let n = 10;
        let mut g = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let xo = vec![f64::INFINITY; 10];
        let xu = vec![f64::NEG_INFINITY; 10];
        let convex = false;
        let mut nfree = 10;
        let mut unfix = false;
        let mut alp = 1.0;
        let mut alpu = f64::NEG_INFINITY;
        let mut alpo = f64::INFINITY;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &mut G, &n, &mut g, &mut x, &xo, &xu, &convex, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        assert_eq!(nsub, 1);
        assert_eq!(free, vec![true; 10]);
        assert_eq!(L, L.clone());
        assert_eq!(d, vec![1.0; 10]);
        assert_eq!(K, vec![false; 10]);
        assert_eq!(G, G.clone());
        assert_eq!(n, 10);
        assert_eq!(g, g.clone());
        assert_eq!(x, vec![-1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]);
        assert_eq!(xo, vec![f64::INFINITY; 10]);
        assert_eq!(xu, vec![f64::NEG_INFINITY; 10]);
        assert_eq!(convex, false);
        assert_eq!(nfree, 10);
        assert_eq!(alp.is_nan(), true);
        assert_eq!(alpu, f64::NEG_INFINITY);
        assert_eq!(alpo, f64::INFINITY);
        assert_eq!(lba, true);
        assert_eq!(uba, true);
        assert_eq!(ier, IerEnum::UnboundedBelow);
        assert_eq!(unfix, false);
        assert_eq!(subdone, false);
    }
}