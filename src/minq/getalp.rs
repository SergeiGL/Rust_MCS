use crate::minq::IerEnum;

pub fn getalp(alpu: f64, alpo: f64, gTp: f64, pTGp: f64) -> (f64, bool, bool, IerEnum) {
    let mut lba = false;
    let mut uba = false;

    let mut ier = IerEnum::LocalMinimizerFound;

    if alpu == f64::NEG_INFINITY && (pTGp < 0.0 || (pTGp == 0.0 && gTp > 0.0)) {
        ier = IerEnum::UnboundedBelow;
        lba = true;
    }
    if alpo == f64::INFINITY && (pTGp < 0.0 || (pTGp == 0.0 && gTp < 0.0)) {
        ier = IerEnum::UnboundedBelow;
        uba = true;
    }

    if ier != IerEnum::LocalMinimizerFound {
        return (f64::NAN, lba, uba, ier);
    }

    // Activity determination
    let mut alp = f64::NAN;
    if pTGp == 0.0 && gTp == 0.0 {
        alp = 0.0;
    } else if pTGp <= 0.0 {
        // Concave case minimized at bound
        if alpu == f64::NEG_INFINITY {
            lba = false;
        } else if alpo == f64::INFINITY {
            lba = true;
        } else {
            lba = 2.0 * gTp + (alpu + alpo) * pTGp > 0.0;
        }
        uba = !lba;
    } else {
        alp = -gTp / pTGp;
        lba = alp <= alpu;
        uba = alp >= alpo;
    }

    // Apply bound if necessary
    if lba {
        alp = alpu;
    }
    if uba {
        alp = alpo;
    }


    (alp, lba, uba, ier)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinities() {
        let alpu = f64::NEG_INFINITY;
        let alpo = f64::INFINITY;
        let gTp = 1.0;
        let pTGp = -2.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp.is_nan(), lba, uba, ier), (true, true, true, IerEnum::UnboundedBelow));
    }

    #[test]
    fn test_finite_minimizer() {
        let alpu = -10.0;
        let alpo = 10.0;
        let gTp = 5.0;
        let pTGp = 1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (-5.0, false, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_unbounded_minimum_negative_p_tgp() {
        let alpu = f64::NEG_INFINITY;
        let alpo = f64::INFINITY;
        let gTp = 5.0;
        let pTGp = -1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert!((alp.is_nan(), lba, uba, ier) == (true, true, true, IerEnum::UnboundedBelow));
    }

    #[test]
    fn test_zero_gradient_and_quadratic() {
        let alpu = 0.0;
        let alpo = 10.0;
        let gTp = 0.0;
        let pTGp = 0.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (0.0, false, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_zero_quadratic() {
        let alpu = -10.0;
        let alpo = 10.0;
        let gTp = 5.0;
        let pTGp = 0.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (-10.0, true, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_active_lower_bound() {
        let alpu = -5.0;
        let alpo = 10.0;
        let gTp = 8.0;
        let pTGp = -1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (-5.0, true, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_active_upper_bound() {
        let alpu = -10.0;
        let alpo = 5.0;
        let gTp = -6.0;
        let pTGp = 1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (5.0, false, true, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_unbounded_minimum_infinite_lower_bound() {
        let alpu = f64::NEG_INFINITY;
        let alpo = 10.0;
        let gTp = 5.0;
        let pTGp = -1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert!((alp.is_nan(), lba, uba, ier) == (true, true, false, IerEnum::UnboundedBelow));
    }

    #[test]
    fn test_unbounded_minimum_infinite_upper_bound() {
        let alpu = -10.0;
        let alpo = f64::INFINITY;
        let gTp = -5.0;
        let pTGp = -1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert!((alp.is_nan(), lba, uba, ier) == (true, false, true, IerEnum::UnboundedBelow));
    }

    #[test]
    fn test_minimizer_at_lower_bound() {
        let alpu = 0.0;
        let alpo = 10.0;
        let gTp = 8.0;
        let pTGp = 0.5;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (0.0, true, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_minimizer_at_upper_bound() {
        let alpu = 0.0;
        let alpo = 10.0;
        let gTp = -8.0;
        let pTGp = 0.5;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (10.0, false, true, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_alpu_alpo_equal() {
        let alpu = 5.0;
        let alpo = 5.0;
        let gTp = -3.0;
        let pTGp = 1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (5.0, true, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_infinite_lower_bound() {
        let alpu = f64::NEG_INFINITY;
        let alpo = 10.0;
        let gTp = 4.0;
        let pTGp = 2.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (-2.0, false, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_infinite_upper_bound() {
        let alpu = -10.0;
        let alpo = f64::INFINITY;
        let gTp = -4.0;
        let pTGp = 2.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (2.0, false, false, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_infinite_bounds() {
        let alpu = f64::NEG_INFINITY;
        let alpo = f64::INFINITY;
        let gTp = 4.0;
        let pTGp = 1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (-4.0, false, false, IerEnum::LocalMinimizerFound));
    }
}