use crate::minq::getalp::getalp;
use crate::minq::ldldown::ldldown;
use crate::minq::ldlup::ldlup;


use crate::minq::IerEnum;
use nalgebra::{Cholesky, DMatrix, DVector};

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
    xo: &Vec<f64>,
    xu: &Vec<f64>,
    convex: &bool,
    xx: &mut Vec<f64>,
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
    let mut p = vec![0.0; *n];

    // Downdate factorization, only for indices in freelK
    let freelk_idxs: Vec<usize> = free
        .iter()
        .zip(K.iter())
        .enumerate()
        .filter(|&(_, (&f, &k))| f && !k)
        .map(|(i, _)| i)
        .collect();

    for &j in &freelk_idxs {
        let (new_l, new_d) = ldldown(L.clone(), d.clone(), j);
        *L = new_l;
        *d = new_d;
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
        if *n > 1 {
            for (i, &ki) in K.iter().enumerate() {
                if ki {
                    p[i] = G[i][j];
                }
            }
        }
        p[j] = G[j][j];

        let (new_l, new_d, new_p) = ldlup(L.clone(), d.clone(), j, p.clone());
        *L = new_l;
        *d = new_d;
        p = new_p;

        definite = p.is_empty();
        if !definite {
            break;
        }
        K[j] = true;
    }

    if definite {
        // Find reduced Newton direction
        let mut p_new = vec![0.0; *n];
        for (i, &ki) in K.iter().enumerate() {
            if ki {
                p_new[i] = g[i];
            }
        }

        // Convert L to Crate Matrix (nalgebra)
        let l_flat: Vec<f64> = L.iter().flatten().cloned().collect();
        let l = DMatrix::from_row_slice(*n, *n, &l_flat);

        // Create DVector for p_new
        let p_vec = DVector::from_vec(p_new.clone());

        if let Some(cholesky) = Cholesky::new(l) {
            let lp_solved = cholesky.solve(&p_vec);

            // Solve and scale by d
            let lp_solved_scaled = lp_solved.component_div(&DVector::from_vec(d.clone()));

            // Solve for p and apply negation
            let solved_p = -cholesky.solve(&lp_solved_scaled);
            p = solved_p.data.as_vec().clone();
        }
    }

    // Set tiny entries to zero
    let p_shifted: Vec<f64> = p.iter().enumerate().map(|(i, &pi)| pi + x[i]).collect();
    p = p_shifted.iter().zip(x.iter()).map(|(ps, xi)| ps - xi).collect();

    let ind: Vec<usize> = p
        .iter()
        .enumerate()
        .filter(|&(_, &pi)| pi != 0.0)
        .map(|(i, _)| i)
        .collect();

    if ind.is_empty() {
        *unfix = true;
        *subdone = false;
        return;
    }

    // Find range of step sizes
    let pp: Vec<f64> = ind.iter().map(|&i| p[i]).collect();
    let oo: Vec<f64> = ind.iter().map(|&i| (xo[i] - x[i]) / pp[i]).collect();
    let uu: Vec<f64> = ind.iter().map(|&i| (xu[i] - x[i]) / pp[i]).collect();

    // Here we handle defaults properly in case no valid extremes exist
    let mut alpu_new = f64::NEG_INFINITY;
    let mut alpo_new = f64::INFINITY;

    for ((&pi, &o_i), &u_i) in pp.iter().zip(oo.iter()).zip(uu.iter()) {
        if pi < 0.0 {
            alpu_new = alpu_new.max(o_i);
        } else if pi > 0.0 {
            alpu_new = alpu_new.max(u_i);
        }
        if pi > 0.0 {
            alpo_new = alpo_new.min(o_i);
        } else if pi < 0.0 {
            alpo_new = alpo_new.min(u_i);
        }
    }

    *alpu = alpu_new;
    *alpo = alpo_new;

    // Check again if alpo and alpu are valid
    if *alpo <= 0.0 || *alpu >= 0.0 {
        panic!("programming error: no alp");
    }

    // Find step size
    let gTp: f64 = g.iter().zip(p.iter()).map(|(gi, pi)| gi * pi).sum();
    let agTp: f64 = g.iter().map(|g_val| g_val.abs()).sum::<f64>() * p.iter().map(|p_val| p_val.abs()).sum::<f64>();
    let gTp_corr = if gTp.abs() < 100.0 * eps * agTp { 0.0 } else { gTp };

    // Solve pTGp
    let G_matrix = DMatrix::from_fn(G.len(), G[0].len(), |i, j| G[i][j]);
    let p_vector = DVector::from_vec(p.clone());

    let mut pTGp = (p_vector.transpose() * &G_matrix * &p_vector)[(0, 0)];

    if *convex {
        pTGp = pTGp.max(0.0)
    }

    if !definite && pTGp > 0.0 {
        pTGp = 0.0;
    }

    let (new_alp, new_lba, new_uba, new_ier) = getalp(*alpu, *alpo, gTp_corr, pTGp);
    *alp = new_alp;
    *lba = new_lba;
    *uba = new_uba;
    *ier = new_ier;

    if *ier != IerEnum::LocalMinimizerFound {
        if *lba {
            *x = p.into_iter().map(|x| -x).collect();
        }
        return;
    }

    *unfix = !(*lba || *uba);

    for k in &ind {
        let ik = *k;
        if *alp == uu[ik] {
            xx[ik] = xu[ik];
            free[ik] = false;
        } else if *alp == oo[ik] {
            xx[ik] = xo[ik];
            free[ik] = false;
        } else {
            xx[ik] += *alp * p[ik];
        }
        if xx[ik].abs() == f64::INFINITY {
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
        let mut xx = x.clone();
        let mut nfree = 10;
        let mut unfix = false;
        let mut alp = 1.0;
        let mut alpu = f64::NEG_INFINITY;
        let mut alpo = f64::INFINITY;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &mut G, &n, &mut g, &mut x, &xo, &xu, &convex, &mut xx, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

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
        assert_eq!(xx, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
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