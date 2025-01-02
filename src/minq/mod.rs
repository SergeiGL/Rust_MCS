mod getalp;
mod ldlrk1;
mod ldlup;
mod ldldown;
mod minqsub;

use crate::minq::{getalp::getalp, minqsub::minqsub};
use nalgebra::{Const, DimMin, SMatrix, SVector};


#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
pub enum IerEnum {
    LocalMinimizerFound, // 0
    UnboundedBelow,      // 1
    MaxitExceeded,       // 99
    InputError,          // -1
}

pub fn minq<const N: usize>(
    gam: f64,
    c: &SVector<f64, N>,
    G: &mut SMatrix<f64, N, N>,
    xu: &[f64; N],
    xo: &[f64; N],
) -> (
    SVector<f64, N>,   // x
    f64,               // fct
    IerEnum            // ier
) where
    Const<N>: DimMin<Const<N>, Output=Const<N>>,
{
    let (mut alp, mut alpu, mut alpo, mut lba, mut uba) = (0.0, 0.0, 0.0, true, true);

    let convex = false;

    // Ensure input consistency
    let mut ier = IerEnum::LocalMinimizerFound;

    let maxit = 3 * N;
    let nitrefmax = 3;

    let mut x = SVector::<f64, N>::from_fn(|i, _| 0.0_f64.clamp(xu[i], xo[i]));

    let hpeps = 2.2204e-14;

    for i in 0..N {
        G[(i, i)] += hpeps * G[(i, i)];
    }
    // println!("{G:?}");

    let mut K = [false; N];
    let mut L = SMatrix::<f64, N, N>::identity();
    let mut d = SVector::<f64, N>::repeat(1.0);

    let mut free = [false; N];
    let mut nfree: usize = 0;
    let mut nfree_old_option: Option<usize> = None;

    let mut fct = f64::INFINITY;
    let mut nsub = 0;
    let mut unfix = true;
    let mut nitref = 0;
    let mut improvement = true;
    let mut g;

    // Main loop: alternating coordinate and subspace searches
    loop {
        if x.iter().any(|&val| val.is_infinite()) {
            panic!("infinite x in minq {x:?}");
        }

        g = *G * x + c;

        let fctnew = gam + (x.transpose().scale(0.5) * (c + g))[0];

        // println!("{improvement}, {nitref}, {nitrefmax}");
        if !improvement || nitref > nitrefmax || (nitref > 0 && nfree_old_option.map_or(false, |nfree_old| nfree_old == nfree) && fctnew >= fct) {
            // println!("--");
            ier = IerEnum::LocalMinimizerFound;
            break;
        } else if nitref == 0 {
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
        let mut k: usize = 0;
        let mut k_init = true;
        loop {
            while count <= N {
                count += 1;
                k = match (k_init, k + 1) {
                    (false, k_new) if N == k_new => 0,
                    (true, _) => {
                        k_init = false;
                        0
                    }
                    _ => k + 1,
                };
                if free[k] || unfix {
                    break;
                }
            }

            if count > N {
                break;
            }

            let q = G.column(k);
            alpu = xu[k] - x[k];
            alpo = xo[k] - x[k];

            // Find STEP size
            (alp, lba, uba, ier) = getalp(alpu, alpo, g[k], q[k]);
            if ier != IerEnum::LocalMinimizerFound {
                x = SVector::<f64, N>::zeros();
                x[k] = if lba { -1.0 } else { 1.0 };
                return (x, fct, ier);
            }

            let xnew = x[k] + alp;

            if lba || (xnew <= xu[k]) {
                if alpu != 0.0 {
                    x[k] = xu[k];
                    g += q.scale(alpu);
                    count = 0;
                }
                free[k] = false;
            } else if uba || (xnew >= xo[k]) {
                if alpo != 0.0 {
                    x[k] = xo[k];
                    g += q.scale(alpo);
                    count = 0;
                }
                free[k] = false;
            } else {
                if alp != 0.0 {
                    x[k] = xnew;
                    g += q.scale(alp);
                    free[k] = true;
                }
            }
        }

        nfree = free.iter().filter(|&&b| b).count();

        if unfix && nfree_old_option.map_or(false, |nfree_old| nfree_old == nfree) {
            g = *G * x + c;
            nitref += 1;
        } else {
            nitref = 0;
        }
        nfree_old_option = Some(nfree);

        let gain_cs: f64 = fct - gam - x.scale(0.5).dot(&(c + g));

        improvement = (gain_cs > 0.0) || !unfix;

        if !(!improvement || nitref > nitrefmax) && !(nitref > nitrefmax) && nfree == 0 {
            unfix = true;
        } else if !(!improvement || nitref > nitrefmax) && !(nitref > nitrefmax) && !(nfree == 0) {
            let mut subdone = false;
            // println!("PRE minqsub\nnsub={nsub}\nfree={free:?}\n{d:?}\n{K:?}\ng={g:?}x={x:?}\n{convex}\nnfree={nfree:?}\nunfix={unfix:?}\nalp={alp:?}\nalpu={alpu:?}\nalpo={alpo:?}\nlba={lba:?}\nuba={uba:?}\nier={ier:?}\nsubdone{subdone}");
            // println!("{L:.15}\n{G:.15}");
            minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &G, &mut g,
                    &mut x, xo, xu, &convex, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo, &mut lba, &mut uba, &mut ier, &mut subdone);
            // println!("PRE minqsub\nnsub={nsub}\nfree={free:?}\n{d:?}\n{K:?}\ng={g:?}x={x:?}\n{convex}\nnfree={nfree:?}\nunfix={unfix:?}\nalp={alp:?}\nalpu={alpu:?}\nalpo={alpo:?}\nlba={lba:?}\nuba={uba:?}\nier={ier:?}\nsubdone{subdone}");
            // println!("{L:.15}\n{G:.15}");

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
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_real_mistake() {
        let gam = -2.908187629699183;
        let c = SVector::<f64, 6>::from_row_slice(&[2.8705323582816686, -1.845648568065381, -0.02892553811394936, -4.756026093917556, 0.006436196481451522, -0.8361689370862301]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            95.77846342296617, 1.7848504468928648, -0.006459576572370291, 4.655453534841051, 0.014037902478796585, 0.894694212391099,
            1.7848504468928648, 45.01426462103387, -0.10270189816114665, -3.2914523096054302, 0.19693639958736797, -0.0582154345898392,
            -0.006459576572370291, -0.10270189816114665, 0.30851282407215563, -0.13564150643144463, 0.05546105384553863, 0.14420187864794054,
            4.655453534841051, -3.2914523096054302, -0.13564150643144463, 50.08631581627649, 0.21035363638609691, -0.8769174708134662,
            0.014037902478796585, 0.19693639958736797, 0.05546105384553863, 0.21035363638609691, 0.705041024447086, -0.34021712157041645,
            0.894694212391099, -0.0582154345898392, 0.14420187864794054, -0.8769174708134662, -0.34021712157041645, 80.21965674487866
        ]);
        let xu = [-0.125, -0.06239714070289448, -0.11155191538152609, -0.1875, -0.07153472077076538, -0.019673386763705048];
        let xo = [0.125, 0.06239714070289448, 0.11155191538152609, 0.1875, 0.07153472077076538, 0.019673386763705048];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-0.03597742419646178, 0.05042765528716418, 0.11155191538152609, 0.10235490442521464, -0.056242394626247735, 0.011541181030516816];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -3.2572067396795092);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
    #[test]
    fn test_coverage_0() {
        let gam = -5.0;
        let c = SVector::<f64, 6>::from_row_slice(&[0.01, 0.1, -0.7, -0.8, 1.0, -0.5]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            23., 0.08, 0.105, -0.9, 0.7, -0.4,
            1., 0.08, 0.105, -0.9, 0.7, -0.4,
            0.1, 0.08, 0.105, -0.9, 0.7, -0.4,
            0.2, 0.08, 0.105, -0.9, 0.7, -0.4,
            -0.3, 0.08, 0.105, -0.9, 0.7, -0.4,
            4., 0.08, 0.105, -0.9, 0.7, -0.4,
        ]);
        let xu = [-1.; 6];
        let xo = [1.; 6];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [0.042489270386265286, 1.0, 1.0, 1.0, 0.18249540159410974, 1.0];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -8.67044972409567);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_coverage_1() {
        let gam = -5.0;
        let c = SVector::<f64, 6>::from_row_slice(&[0.01, 0.1, -0.7, -0.8, 1.8, -1.5]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            22., 0.08, 0.105, -0.9, 0.7, -0.4,
            0.4, 0.083, 0.307, -0.9, 0.7, -0.4,
            -0.7, 0.08, 0.107, 0.9, 0.17, 0.74,
            1., 0.038, 0.27, -0.49, 0.7, -0.24,
            0.6, 0.08, -0.07, -0.9, 0.7, -0.41,
            0.001, 0.018, 0.074, -0.9, 0.7, -0.7,
        ]);
        let xu = [-1.; 6];
        let xo = [1.; 6];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[0.0913551401869131, 1., -1., 1., -0.9925901201601895, 1.]);

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -11.132255223631478);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_coverage_2() {
        let gam = -5.0;
        let c = SVector::<f64, 6>::from_row_slice(&[0.01, 0.1, 0.7, 0.3, 1.9, -0.5]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            13., 0.08, 0.105, -0.9, 0.7, -0.4,
            1., 0.08, 0.105, 0.9, 0.7, -0.4,
            0.1, 0.08, 0.105, -0.9, 0.7, -0.4,
            0.2, 0.08, 0.105, 0.9, 0.7, -0.4,
            -0.3, 0.08, 0.105, 0.9, 0.7, -0.4,
            4., 0.08, 0.105, -0.9, 0.7, -0.4
        ]);
        let xu = [-2.; 6];
        let xo = [2.; 6];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[0.302259618771836, -2.0, 2.0, 1.9883867513839923, -2.0, 2.0]);

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -14.442743529914747);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_coverage_3() {
        let gam = -5.0;
        let c = SVector::<f64, 6>::from_row_slice(&[0.01, 0.1, 0.7, 0.3, 0.9, -0.5]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            13., 0.08, 0.105, -0.9, 0.7, -0.4,
            1., 0.08, 0.105, 0.9, 0.7, -0.4,
            0.1, 0.08, 0.105, -0.9, 0.7, -0.4,
            0.2, 0.08, 0.105, 0.9, 0.7, -0.4,
            -0.3, 0.08, 0.105, 0.9, 0.7, -0.4,
            4., 0.08, 0.105, -0.9, 0.7, -0.4
        ]);
        let xu = [-3.; 6];
        let xo = [3.; 6];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[0.4549999999999899, -3., 3., 3., -3., 3., ]);

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fct, -20.444599767258207, epsilon = TOLERANCE);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_coverage_4() {
        let gam = 1.0;
        let c = SVector::<f64, 6>::from_row_slice(&[-0.01, -0.1, -0.8, -0.6, -0.9, -0.5]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            13., 0.08, 0.105, -0.9, 0.7, -0.4,
            1., 0.08, 0.105, 0.9, 0.7, -0.4,
            0.1, 0.08, 0.105, -0.9, 0.7, -0.4,
            0.2, 0.08, 0.105, 0.9, 0.7, -0.4,
            -0.3, 0.08, 0.105, 0.9, 0.7, -0.4,
            4., 0.08, 0.105, -0.9, 0.7, 0.0
        ]);
        let xu = [-3.; 6];
        let xo = [3.; 6];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[-0.36999271657216287, -3., -3., -2.30111272965058, 3., -3.]);

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -6.751454506630278); //76
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_coverage_5() {
        let gam = -5.0;
        let c = SVector::<f64, 3>::from_row_slice(&[0.2, 0.0, 0.02]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[1.0, 2.0, 4.0, 2.0, 5.0, 0.0, -0.1, -0.5, -0.005]);
        let xu = [0.0, -2.0, 1.0];
        let xo = [1.0, 2.0, 5.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 3>::from_row_slice(&[0.0, 0.0, 1.0]);

        assert_eq!(x, expected_x);
        assert_eq!(fct, -4.9825);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);

        assert_eq!(gam, -5.) // check copy
    }

    #[test]
    fn test_random_0() {
        let gam = 0.1;
        let c = SVector::<f64, 6>::from_row_slice(&[
            0.26027596522878105, 0.012123239938926389, 0.976390077851892, 0.1291111085459904, 0.4733627957116796, 0.5163968201397172
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            0.5364339063687579, 0.2297669716502435, 0.5038995286989546, 0.7277274960472341, 0.8545905110558993, 0.7777655686147649,
            0.9160796444159384, 0.3641837883133806, 0.18272914225872838, 0.1519322751704012, 0.1124045992097733, 0.6635303861287662,
            0.9354540217237614, 0.8175396106964223, 0.3188913100047359, 0.9915655818487893, 0.13427947848573107, 0.37342282365153623,
            0.6288192276751352, 0.8433039707736678, 0.7321362226848295, 0.2748447596658169, 0.6948392389253439, 0.318415880192617,
            0.6200434355044614, 0.9546431929998173, 0.6513411970145672, 0.788714900630957, 0.6639219390032586, 0.8826883301558496,
            0.3084173620636186, 0.677362811459593, 0.5567194989820659, 0.11920779835756423, 0.02987925514845613, 0.4411971174881518
        ]);
        let xu = [0.9558285681648798, 0.5194757464362438, 0.3015417041472618, 0.8101331118648574, 0.061618326392057776, 0.5421792878798574];
        let xo = [2.172274965654365, 2.8179460480833494, 2.1123039892888325, 2.4283704885558492, 2.500248662862857, 2.874993639004977];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[0.9558285681648798, 0.5194757464362438, 0.3015417041472618, 0.8101331118648574, 0.061618326392057776, 0.5421792878798574]);

        assert_eq!(x, expected_x);
        assert_relative_eq!(fct, 3.8061446326834183, epsilon = TOLERANCE);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_random_1() {
        let gam = -0.1;
        let c = SVector::<f64, 6>::from_row_slice(&[
            6.462580762560808, 5.626749806173567, 4.005918748483873, 6.238520572936755, 1.8034289585073848, 4.131859269316073
        ]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            3.3511760107572677, 8.079843032573027, 7.991191997916239, 1.5340713993172395, 9.282859758186207, 2.562733496882168,
            7.561980917613901, 3.0929688161231885, 8.70847857211592, 5.3019490271867955, 1.5887051838984823, 1.9689449654877478,
            1.769548530579672, 5.397793381627557, 2.606872766367867, 4.138553177184834, 3.015594458862539, 5.913321245206986,
            3.1871525595335157, 1.580619531755375, 5.119519400101709, 6.0460245881504004, 3.061597618428238, 5.3971800858565855,
            5.497168193855644, 9.631296164721292, 5.929885023156226, 3.4969159387270485, 3.123296730543715, 5.2414836381601955,
            7.738659525956443, 7.329831212331891, 5.539367070865469, 8.846867842872843, 9.224333102762714, 6.83895842907691
        ]);
        let xu = [-4.769677847426308, -3.5969155931370618, -1.9037195727875393, -1.9430412656953555, -3.076515775656527, -3.3371066538862233];
        let xo = [-3.7696778474263084, -2.5969155931370618, -0.9037195727875393, -0.9430412656953555, -2.076515775656527, -2.3371066538862233];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[-3.7696778474263084, -2.5969155931370618, -0.9037195727875393, -0.9430412656953555, -2.076515775656527, -2.3371066538862233]);

        assert_eq!(x, expected_x);
        assert_relative_eq!(fct, 371.3367348271027, epsilon = TOLERANCE);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_real_mistake_1() {
        let gam = -3.2661659570240418;
        let c = SVector::<f64, 6>::from_row_slice(&[0.011491952485028996, 0.10990155244417238, -0.5975771816968101, -0.8069326056544889, 1.8713998467574868, -1.4958051414638351]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            23.12798652584253, 0.08086473977917293, 1.7538162952525622, -1.9012829332291588, 1.7864612279290097, -0.7406818881433185,
            0.08086473977917293, 18.576721298566618, -0.5909985456367551, 0.8013573491818613, -0.9992079198191761, 0.1810561706642408,
            1.7538162952525622, -0.5909985456367551, 24.556579083791647, 3.371614208515673, -3.5009378170622605, 0.09958957165430643,
            -1.9012829332291588, 0.8013573491818613, 3.371614208515673, 48.67847201840808, -1.0333246379471976, 0.9233898437170295,
            1.7864612279290097, -0.9992079198191761, -3.5009378170622605, -1.0333246379471976, 89.37343113076405, 4.016171463395642,
            -0.7406818881433185, 0.1810561706642408, 0.09958957165430643, 0.9233898437170295, 4.016171463395642, 48.17000841044120 // accidentally deleted 1 digit
        ]);
        let xo = [0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837];
        let xu = [-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[0.0019995286865852144, -0.007427824716643584, 0.018821593308224843, 0.01439613293535691, -0.021623304847149496, 0.03259925177469269]);

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -3.3226086532809607);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }


    #[test]
    fn test_unbound() {
        let gam = 1.0;
        let c = SVector::<f64, 1>::from_row_slice(&[-1.0]);
        let mut G = SMatrix::<f64, 1, 1>::from_row_slice(&[0.0]);
        let xu = [f64::NEG_INFINITY];
        let xo = [f64::INFINITY];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, SVector::<f64, 1>::from_row_slice(&[1.0]));
        assert_eq!(fct, 1.0);
        assert_eq!(ier, IerEnum::UnboundedBelow);
    }

    #[test]
    fn test_global_return_0() {
        let gam = 0.0;
        let c = SVector::<f64, 2>::from_row_slice(&[-1.0, 20.0]);
        let mut G = SMatrix::<f64, 2, 2>::from_row_slice(&[123.0, 26.0, -0.3, -9.5]);
        let xu = [0.0, 0.0];
        let xo = [11.0, 22.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, SVector::<f64, 2>::from_row_slice(&[0.0, 22.0]));
        assert_eq!(fct, -1859.0000000000514);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
    #[test]
    fn test_munqsub_return() {
        let gam = 1.0;
        let c = SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]);
        let mut G = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 2.0, 3.0, 5.0]);
        let xu = [0.0, 0.0];
        let xo = [1.0, 2.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]));
        assert_eq!(fct, 1.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond() {
        let gam = 2.0;
        let c = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[1.0, 2.0, -4.0, 3.0, 5.0, -1.0, 0.0, -3.0, -10.0]);
        let xu = [-10.0, -10.0, -3.0];
        let xo = [1.0, 2.0, 4.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 3>::from_row_slice(&[1.0, -0.19999999999999557, 4.0]);

        assert_eq!(x, expected_x);
        assert_eq!(fct, 2.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond_2() {
        let gam = 200.0;
        let c = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[1.0, 2.0, 4.0, 3.0, 5.0, -1.0, 0.0, -3.0, -10.0]);
        let xu = [0.0, 0.0, -3.0];
        let xo = [1.0, 2.0, 4.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);
        let expected_x = SVector::<f64, 3>::from_row_slice(&[0.0, 0.39999999999999114, 4.0]);


        assert_eq!(x, expected_x);
        assert_eq!(fct, 200.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_global_return_4() {
        let gam = -200.0;
        let c = SVector::<f64, 2>::from_row_slice(&[-1.0, -2.0]);
        let mut G = SMatrix::<f64, 2, 2>::from_row_slice(&[-1.0, -2.0, 5.0, -1.0]);
        let xu = [0.0, 0.0];
        let xo = [10.0, 20.0];

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);
        let expected_x = SVector::<f64, 2>::from_row_slice(&[10.0, 0.0]);

        assert_eq!(x, expected_x);
        assert_eq!(fct, -260.00000000000114);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
}
