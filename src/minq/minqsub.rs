use crate::minq::getalp::getalp;
use crate::minq::ldldown::ldldown;
use crate::minq::ldlup::ldlup;
use crate::minq::IerEnum;
use nalgebra::{Const, DimMin, SMatrix, SVector};


pub fn minqsub<const N: usize>(
    nsub: &mut usize,
    free: &mut SVector<bool, N>,
    L: &mut SMatrix<f64, N, N>,
    d: &mut SVector<f64, N>,
    K: &mut SVector<bool, N>,
    G: &SMatrix<f64, N, N>,
    g: &mut SVector<f64, N>,
    x: &mut SVector<f64, N>,
    xo: &SVector<f64, N>,
    xu: &SVector<f64, N>,
    nfree: &mut usize,
    unfix: &mut bool,
    alp: &mut f64,
    alpu: &mut f64,
    alpo: &mut f64,
    lba: &mut bool,
    uba: &mut bool,
    ier: &mut IerEnum,
    subdone: &mut bool,
)
where
    Const<N>: DimMin<Const<N>, Output=Const<N>>,
{
    let mut definite = true;
    let mut has_nonzero_p;

    let mut p = SVector::<f64, N>::zeros();

    *nsub += 1;

    // First phase: Update L and d matrices
    for i in 0..N {
        if !free[i] && K[i] {
            ldldown(L, d, i);
            K[i] = false;
            continue;
        }

        if !(free[i] && !K[i]) {
            continue;
        }

        // Optimize G access for N > 1 case
        if N > 1 {
            for (k_ind, &ki) in K.iter().enumerate() {
                p[k_ind] = if ki { G[(k_ind, i)] } else { 0.0 };
            }
        }
        p[i] = G[(i, i)];

        if let Some(p_new) = ldlup(L, d, i, &p) {
            p = p_new;
            definite = false;
            break;
        }
        K[i] = true;
    }

    if definite {
        for ((&ki, p_i), &g_i) in K.iter().zip(p.iter_mut()).zip(g.iter()) {
            *p_i = if ki { g_i } else { 0.0 };
        }

        p = -L.transpose()
            .solve_upper_triangular(&L.solve_lower_triangular(&p).unwrap().component_div(d))
            .unwrap();
    }

    (*alpo, *alpu) = (f64::INFINITY, f64::NEG_INFINITY);

    // Optimize step size computation
    has_nonzero_p = false;
    for (i, p_i) in p.iter().enumerate() {
        if p_i.abs() <= f64::EPSILON {
            continue;
        }
        has_nonzero_p = true;

        let (p_i, o_i, u_i) = (p[i], (xo[i] - x[i]) / p[i], (xu[i] - x[i]) / p[i]);
        match p_i.partial_cmp(&0.0).unwrap() {
            std::cmp::Ordering::Less => {
                *alpu = alpu.max(o_i);
                *alpo = alpo.min(u_i);
            }
            std::cmp::Ordering::Greater => {
                *alpu = alpu.max(u_i);
                *alpo = alpo.min(o_i);
            }
            std::cmp::Ordering::Equal => {}
        }
    }

    if !has_nonzero_p {
        *unfix = true;
        *subdone = false;
        return;
    }

    debug_assert!(!(*alpo <= 0.0 || *alpu >= 0.0)); // minqsub error: no alp

    // Find STEP size
    let g_mul_p = g.component_mul(&p);
    let gTp = g_mul_p.sum();
    let agTp = g_mul_p.iter().fold(0.0, |acc, &x| acc + x.abs());

    let gTp_corr = if gTp.abs() < 100.0 * f64::EPSILON * agTp {
        0.0
    } else {
        gTp
    };

    // Compute pTGp
    let mut pTGp = p.dot(&(G * p));

    if !definite && pTGp > 0.0 {
        pTGp = 0.0;
    }

    (*alp, *lba, *uba, *ier) = getalp(*alpu, *alpo, gTp_corr, pTGp);

    if *ier != IerEnum::LocalMinimizerFound {
        *x = if *lba { -p } else { p };
        return;
    }

    *unfix = !(*lba || *uba);

    // Update x values efficiently
    let mut free_count = 0;
    for i in 0..N {
        let p_i = p[i];
        if p_i.abs() <= f64::EPSILON {
            if free[i] {
                free_count += 1;
            }
            continue;
        }

        let o_i = (xo[i] - x[i]) / p_i;
        let u_i = (xu[i] - x[i]) / p_i;

        if *alp == u_i {
            x[i] = xu[i];
            free[i] = false;
        } else if *alp == o_i {
            x[i] = xo[i];
            free[i] = false;
        } else {
            x[i] += *alp * p_i;
            if free[i] {
                free_count += 1;
            }
        }

        debug_assert!(!x[i].is_infinite());
    }

    *nfree = free_count;
    *subdone = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-12;

    #[test]
    fn test_real_miskake_2() {
        const N: usize = 6;
        let mut nsub = 1;
        let mut free = SVector::<bool, N>::from_row_slice(&[true, false, false, true, false, false]);
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.006153846153846018, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            13.000000000000288, 0.0795076923076941, 1.0, 1.0, 1.0, 1.0,
        ]);
        let mut K = SVector::<bool, N>::from_row_slice(&[true, true, false, false, false, false]);
        let G = SMatrix::<f64, N, N>::from_row_slice(&[
            13.000000000000288, 0.08, 0.105, -0.9, 0.7, -0.4,
            1.0, 0.08000000000000178, 0.105, 0.9, 0.7, -0.4,
            0.1, 0.08, 0.10500000000000233, -0.9, 0.7, -0.4,
            0.2, 0.08, 0.105, 0.90000000000002, 0.7, -0.4,
            -0.3, 0.08, 0.105, 0.9, 0.7000000000000155, -0.4,
            4.0, 0.08, 0.105, -0.9, 0.7, -0.4000000000000089,
        ]);
        let mut g = SVector::<f64, N>::from_row_slice(&[
            0.00204492613753969, 0.04194211673555729, -3.209271706224105,
            0.0, 1.4487861770401786, -3.2298038871380736,
        ]);
        let mut x = SVector::<f64, N>::from_row_slice(&[
            0.302427645919501, -2.0, 2.0, 1.9883494120178444, -2.0, 2.0,
        ]);
        let xo = SVector::<f64, N>::from_row_slice(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        let xu = SVector::<f64, N>::from_row_slice(&[-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]);
        let mut nfree = 2;
        let mut unfix = false;
        let mut alp = -0.0022721401528218777;
        let mut alpu = -3.9906215521706665;
        let mut alpo = 0.009378447829333725;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &G, &mut g, &mut x, &xo, &xu, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            -0.0692307692307677, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            13.000000000000288, 1.0, 1.0, 0.837692307692329, 1.0, 1.0,
        ]);
        let expected_g = SVector::<f64, N>::from_row_slice(&[
            0.00204492613753969, 0.04194211673555729, -3.209271706224105,
            0.0, 1.4487861770401786, -3.2298038871380736,
        ]);
        let expected_x = SVector::<f64, N>::from_row_slice(&[
            0.3022727272727177, -2.0, 2.0, 1.9881944933710611, -2.0, 2.0,
        ]);

        assert_eq!(nsub, 2);
        assert_eq!(free.as_slice(), [true, false, false, true, false, false]);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(K.as_slice(), [true, false, false, true, false, false]);
        assert_eq!(g, expected_g);
        assert_eq!(x, expected_x);
        assert_eq!(nfree, 2);
        assert_relative_eq!(alp, 0.9166666666667754, epsilon=TOLERANCE);
        assert_relative_eq!(alpu/100., -68.93750928024599/100., epsilon=TOLERANCE);
        assert_relative_eq!(alpo/100_000., 13623.65808925945/100_000., epsilon=TOLERANCE);
        assert_eq!(lba, false);
        assert_eq!(uba, false);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
        assert_eq!(unfix, true);
        assert_eq!(subdone, true);
    }


    #[test]
    fn test_real_miskake() {
        const N: usize = 6;

        let mut nsub = 1;
        let mut free = SVector::<bool, N>::from_row_slice(&[true, true, false, true, true, true]);
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1., 0., 0., 0., 0., 0.,
            0.018635196087984494, 1., 0., 0., 0., 0.,
            -0.00006744289208153283, -0.002280552111200885, 1., 0., 0., 0.,
            0.048606475490027994, -0.07510298414917427, -0.46396921208814274, 1., 0., 0.,
            0.00014656637804684393, 0.004372396897708983, 0.18136372644709786, 0.005054161476720696, 1., 0.,
            0.00934128801419616, -0.0016648858565015735, 0.4674067707226544, -0.017343086953306597, -0.5222650783107669, 1.
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            95.7784634229683, 44.981003582969294, 0.30827844591150716, 49.5399548476696, 0.6927734092514332, 79.93996295955547
        ]);
        let mut K = SVector::<bool, N>::from_row_slice(&[true; 6]);
        let G = SMatrix::<f64, N, N>::from_row_slice(&[
            95.7784634229683, 1.7848504468928648, -0.006459576572370291, 4.655453534841051, 0.014037902478796585, 0.894694212391099,
            1.7848504468928648, 45.01426462103487, -0.10270189816114665, -3.2914523096054302, 0.19693639958736797, -0.0582154345898392,
            -0.006459576572370291, -0.10270189816114665, 0.30851282407216246, -0.13564150643144463, 0.05546105384553863, 0.14420187864794054,
            4.655453534841051, -3.2914523096054302, -0.13564150643144463, 50.086315816277605, 0.21035363638609691, -0.8769174708134662,
            0.014037902478796585, 0.19693639958736797, 0.05546105384553863, 0.21035363638609691, 0.7050410244471017, -0.34021712157041645,
            0.894694212391099, -0.0582154345898392, 0.14420187864794054, -0.8769174708134662, -0.34021712157041645, 80.21965674488044
        ]);
        let mut g = SVector::<f64, N>::from_row_slice(&[
            0.019899634798287977, -0.004062680923063253, -0.014785733051655968, 0.00048366354026713827, -3.1003490957217014e-5, 0.0
        ]);
        let mut x = SVector::<f64, N>::from_row_slice(&[
            -0.03576697214887524, 0.050327925349288836, 0.11155191538152609, 0.10233846934382163, -0.056259086626347536, 0.011538511022023103
        ]);
        let xo = SVector::<f64, N>::from_row_slice(&[0.125, 0.06239714070289448, 0.11155191538152609, 0.1875, 0.07153472077076538, 0.019673386763705048]);
        let xu = SVector::<f64, N>::from_row_slice(&[-0.125, -0.06239714070289448, -0.11155191538152609, -0.1875, -0.07153472077076538, -0.019673386763705048]);
        let mut nfree = 5;
        let mut unfix = false;
        let mut alp = 9.112854407240597e-5;
        let mut alpu = -0.031120769241655744;
        let mut alpo = 0.008226004285754351;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &G, &mut g, &mut x, &xo, &xu, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.018635196087984494, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.048606475490027994, -0.07510298414917427, 0.0, 1.0, 0.0, 0.0,
            0.00014656637804684393, 0.004372396897708983, 0.0, 0.004524467461310853, 1.0, 0.0,
            0.00934128801419616, -0.0016648858565015735, 0.0, -0.018667576757453973, -0.477600159127822, 1.0
        ]);

        let expected_d = SVector::<f64, N>::from_row_slice(&[
            95.7784634229683, 44.981003582969294, 1.0, 49.60631715637313, 0.703163545389539, 80.03349478791684
        ]);
        let expected_g = SVector::<f64, N>::from_row_slice(&[
            0.019899634798287977, -0.004062680923063253, -0.014785733051655968, 0.00048366354026713827, -3.1003490957217014e-05, 0.0
        ]);
        let expected_x = SVector::<f64, N>::from_row_slice(&[
            -0.03597742419646178, 0.05042765528716419, 0.11155191538152609, 0.10235490442521464, -0.056242394626247735, 0.011541181030516816
        ]);

        assert_eq!(nsub, 2);
        assert_eq!(free.as_slice(), [true, true, false, true, true, true]);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(K.as_slice(), [true, true, false, true, true, true]);
        assert_eq!(g, expected_g);
        assert_eq!(x, expected_x);
        assert_eq!(nfree, 5);
        assert_relative_eq!(alp, 0.9999999999999863, epsilon = TOLERANCE);
        assert_relative_eq!(alpu/1000., -763.9126061853459/1000., epsilon = TOLERANCE);
        assert_relative_eq!(alpo/1000., 121.01897996457424/1000., epsilon = TOLERANCE);
        assert_eq!(lba, false);
        assert_eq!(uba, false);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
        assert_eq!(unfix, true);
        assert_eq!(subdone, true);
    }

    #[test]
    fn test_real_miskake_1() {
        const N: usize = 6;
        let mut nsub = 1;
        let mut free = SVector::<bool, N>::from_row_slice(&[true, true, true, true, true, true]);
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.07583091132005203, -0.03214451416643742, 1.0, 0.0, 0.0, 0.0,
            -0.08220702356016504, 0.043496227547229385, 0.1451281097853027, 1.0, 0.0, 0.0,
            0.07724240179458904, -0.054125235207729994, -0.1503305948215706, -0.006468139053208312, 1.0, 0.0,
            -0.032025351074797884, 0.009885958049664674, 0.006624191581076727, 0.01732330782068277, 0.046407734608255605, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            23.127986525843045, 18.57643856292412, 24.40439112304386,
            47.97301864636891, 88.62749123902528, 47.93812981294381,
        ]);
        let mut K = SVector::<bool, N>::from_row_slice(&[true, true, true, true, true, true]);
        let G = SMatrix::<f64, N, N>::from_row_slice(&[
            23.127986525843045, 0.08086473977917293, 1.7538162952525622, -1.9012829332291588, 1.7864612279290097, -0.7406818881433185,
            0.08086473977917293, 18.57672129856703, -0.5909985456367551, 0.8013573491818613, -0.9992079198191761, 0.1810561706642408,
            1.7538162952525622, -0.5909985456367551, 24.55657908379219, 3.371614208515673, -3.5009378170622605, 0.09958957165430643,
            -1.9012829332291588, 0.8013573491818613, 3.371614208515673, 48.67847201840916, -1.0333246379471976, 0.9233898437170295,
            1.7864612279290097, -0.9992079198191761, -3.5009378170622605, -1.0333246379471976, 89.37343113076604, 4.016171463395642,
            -0.7406818881433185, 0.1810561706642408, 0.09958957165430643, 0.9233898437170295, 4.016171463395642, 48.17000841044228,
        ]);
        let mut g = SVector::<f64, N>::from_row_slice(&[6.938893903907228e-18, 0.0, 1.1102230246251565e-16, 0.0, -2.220446049250313e-16, 0.0, ]);
        let mut x = SVector::<f64, N>::from_row_slice(&[0.001999528686585215, -0.007427824716643584, 0.018821593308224843, 0.01439613293535691, -0.0216233048471495, 0.032599251774692695, ]);
        let xo = SVector::<f64, N>::from_row_slice(&[0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837, ]);
        let xu = SVector::<f64, N>::from_row_slice(&[-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837, ]);
        let mut nfree = 6;
        let mut unfix = true;
        let mut alp = 1.9926847409028064e-19;
        let mut alpu = -0.4050781679349764;
        let mut alpo = 0.33987966438559103;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &G, &mut g, &mut x, &xo, &xu, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.07583091132005203, -0.03214451416643742, 1.0, 0.0, 0.0, 0.0,
            -0.08220702356016504, 0.043496227547229385, 0.1451281097853027, 1.0, 0.0, 0.0,
            0.07724240179458904, -0.054125235207729994, -0.1503305948215706, -0.006468139053208312, 1.0, 0.0,
            -0.032025351074797884, 0.009885958049664674, 0.006624191581076727, 0.01732330782068277, 0.046407734608255605, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            23.127986525843045, 18.57643856292412, 24.40439112304386,
            47.97301864636891, 88.62749123902528, 47.93812981294381,
        ]);
        let expected_g = SVector::<f64, N>::from_row_slice(&[
            0.000000000000000006938893903907228, 0.0, 0.00000000000000011102230246251565,
            0.0, -0.0000000000000002220446049250313, 0.0,
        ]);
        let expected_x = SVector::<f64, N>::from_row_slice(&[
            0.001999528686585215, -0.007427824716643584, 0.01882159330822484,
            0.01439613293535691, -0.021623304847149496, 0.032599251774692695,
        ]);

        assert_eq!(nsub, 2);
        assert_eq!(free.as_slice(), [true; 6]);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(K.as_slice(), [true; 6]);
        assert_eq!(g, expected_g);
        assert_relative_eq!(x, expected_x , epsilon=TOLERANCE);
        assert_eq!(xo.as_slice(), [0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837, ]);
        assert_eq!(xu.as_slice(), [-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837, ]);
        assert_eq!(nfree, 6);
        // assert_relative_eq!(alp, 0.7938353009736487, epsilon=TOLERANCE);
        // assert_relative_eq!(alpu, -8.934536499651395e+16, epsilon=TOLERANCE);
        // assert_relative_eq!(alpo, 1.0181035157598803e+17, epsilon=TOLERANCE);
        assert_eq!(lba, false);
        assert_eq!(uba, false);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
        assert_eq!(unfix, true);
        // assert_eq!(subdone, true);
    }

    #[test]
    fn test_real_miskake_0() {
        const N: usize = 6;
        let mut nsub = 0;
        let mut free = SVector::<bool, N>::from_row_slice(&[true, true, true, true, true, true]);
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let mut K = SVector::<bool, N>::from_row_slice(&[false, false, false, false, false, false]);
        let G = SMatrix::<f64, N, N>::from_row_slice(&[
            23.127986525843045, 0.08086473977917293, 1.7538162952525622, -1.9012829332291588, 1.7864612279290097, -0.7406818881433185,
            0.08086473977917293, 18.57672129856703, -0.5909985456367551, 0.8013573491818613, -0.9992079198191761, 0.1810561706642408,
            1.7538162952525622, -0.5909985456367551, 24.55657908379219, 3.371614208515673, -3.5009378170622605, 0.09958957165430643,
            -1.9012829332291588, 0.8013573491818613, 3.371614208515673, 48.67847201840916, -1.0333246379471976, 0.9233898437170295,
            1.7864612279290097, -0.9992079198191761, -3.5009378170622605, -1.0333246379471976, 89.37343113076604, 4.016171463395642,
            -0.7406818881433185, 0.1810561706642408, 0.09958957165430643, 0.9233898437170295, 4.016171463395642, 48.17000841044228,
        ]);
        let mut g = SVector::<f64, N>::from_row_slice(&[-0.045952748319448775, 0.023404204756152784, 0.1232952812014726, 0.05044112085181421, 0.13007152652763362, 0.0, ]);
        let mut x = SVector::<f64, N>::from_row_slice(&[-0.0004968851253950693, -0.005913926908419606, 0.024227865888591913, 0.014976641791235562, -0.019873081026496084, 0.03238694555577046, ]);
        let xo = SVector::<f64, N>::from_row_slice(&[0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837, ]);
        let xu = SVector::<f64, N>::from_row_slice(&[-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837, ]);
        let mut nfree = 6;
        let mut unfix = true;
        let mut alp = 0.03238694555577046;
        let mut alpu = -0.3724789161602837;
        let mut alpo = 0.3724789161602837;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &G, &mut g, &mut x, &xo, &xu, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.07583091132005203, -0.03214451416643742, 1.0, 0.0, 0.0, 0.0,
            -0.08220702356016504, 0.043496227547229385, 0.1451281097853027, 1.0, 0.0, 0.0,
            0.07724240179458904, -0.054125235207729994, -0.1503305948215706, -0.006468139053208312, 1.0, 0.0,
            -0.032025351074797884, 0.009885958049664674, 0.006624191581076727, 0.01732330782068277, 0.046407734608255605, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[23.127986525843045, 18.57643856292412, 24.40439112304386, 47.97301864636891, 88.62749123902528, 47.93812981294381, ]);
        let expected_g = SVector::<f64, N>::from_row_slice(&[-0.045952748319448775, 0.023404204756152784, 0.1232952812014726, 0.05044112085181421, 0.13007152652763362, 0.0, ]);
        let expected_x = SVector::<f64, N>::from_row_slice(&[0.0019995286865852144, -0.007427824716643584, 0.01882159330822484, 0.01439613293535691, -0.021623304847149496, 0.032599251774692695, ]);
        assert_eq!(xo.as_slice(), [0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837, ]);
        assert_eq!(xu.as_slice(), [-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837, ]);

        assert_eq!(nsub, 1);
        assert_eq!(free.as_slice(), [true; 6]);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(K.as_slice(), [true; 6]);
        assert_eq!(g, expected_g);
        assert_relative_eq!(x, expected_x, epsilon = TOLERANCE);
        assert_eq!(nfree, 6);
        assert_relative_eq!(alp, 0.9999999999999998, epsilon = TOLERANCE);
        assert_relative_eq!(alpu, -62.99290436226673, epsilon = TOLERANCE);
        assert_relative_eq!(alpo, 71.95577685100348, epsilon = TOLERANCE);
        assert_eq!(lba, false);
        assert_eq!(uba, false);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
        assert_eq!(unfix, true);
        assert_eq!(subdone, true);
    }

    #[test]
    fn test_minqsub_0() {
        const N: usize = 10;
        let mut nsub = 0;
        let mut free = SVector::<bool, N>::from_row_slice(&[true; N]);
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let mut K = SVector::<bool, N>::from_row_slice(&[false; N]);
        let G = SMatrix::<f64, N, N>::from_row_slice(&[
            -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0,
        ]);
        let mut g = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
        ]);
        let mut x = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
        ]);
        let xo = SVector::<f64, N>::from_row_slice(&[f64::INFINITY; N]);
        let xu = SVector::<f64, N>::from_row_slice(&[f64::NEG_INFINITY; N]);
        let mut nfree = 10;
        let mut unfix = false;
        let mut alp = 1.0;
        let mut alpu = f64::NEG_INFINITY;
        let mut alpo = f64::INFINITY;
        let mut lba = false;
        let mut uba = false;
        let mut ier = IerEnum::LocalMinimizerFound;
        let mut subdone = false;

        minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &G, &mut g, &mut x, &xo, &xu, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);

        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let expected_x = SVector::<f64, N>::from_row_slice(&[
            -1.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let expected_g = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        assert_eq!(L, expected_L);
        assert_eq!(nsub, 1);
        assert_eq!(free.as_slice(), [true; 10]);
        assert_eq!(d, expected_d);
        assert_eq!(K.as_slice(), [false; 10]);
        assert_eq!(g, expected_g);
        assert_eq!(x, expected_x);
        assert_eq!(xo.as_slice(), [f64::INFINITY; 10]);
        assert_eq!(xu.as_slice(), [f64::NEG_INFINITY; 10]);
        assert_eq!(nfree, 10);
        assert!(alp.is_nan());
        assert_eq!(alpu, f64::NEG_INFINITY);
        assert_eq!(alpo, f64::INFINITY);
        assert_eq!(lba, true);
        assert_eq!(uba, true);
        assert_eq!(ier, IerEnum::UnboundedBelow);
        assert_eq!(unfix, false);
        assert_eq!(subdone, false);
    }
}