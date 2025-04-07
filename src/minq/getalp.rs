use crate::minq::IerEnum;

/// Computes the minimizer of a univariate quadratic function within given bounds.
///
/// Finds the value of `alp` that minimizes the quadratic function:
///
/// `q(alp) = alp * gTp + 0.5 * alp^2 * pTGp`
///
/// subject to the bounds: `alp ∈ [alpu, alpo]`
///
/// # Arguments
/// * `alpu` - Lower bound for alp. Can be `f64::NEG_INFINITY` for no lower bound.
/// * `alpo` - Upper bound for alp. Can be `f64::INFINITY` for no upper bound.
/// * `gTp` - Linear coefficient of the quadratic function (inner product of gradient and direction).
/// * `pTGp` - Quadratic coefficient (curvature term, inner product of direction and Hessian-direction).
///
/// # Returns
/// * `alp` - The computed minimizer within bounds, or `f64::NAN` if unbounded.
/// * `lba` - Boolean flag indicating if the lower bound is active (minimizer at lower bound).
/// * `uba` - Boolean flag indicating if the upper bound is active (minimizer at upper bound).
/// * `ier` - Error/status code as an `IerEnum` value:
///   - `LocalMinimizerFound` - A valid minimizer was found within the bounds
///   - `Unbounded` - The quadratic function is unbounded below (no finite minimizer exists)
///
/// # Mathematical Details
/// - If `pTGp > 0`, the function is strictly convex and has a unique unconstrained minimizer at `-gTp/pTGp`
/// - If `pTGp = 0` and `gTp = 0`, any point is a minimizer (constant function)
/// - If `pTGp = 0` and `gTp ≠ 0`, the function is linear and minimized at a bound
/// - If `pTGp < 0`, the function is concave and minimized at a bound (or unbounded if infinite bounds)
pub fn getalp(alpu: f64, alpo: f64, gTp: f64, pTGp: f64) -> (
    f64,     // alp
    bool,    // lba
    bool,    // uba
    IerEnum  // ier
)
{
    let (mut lba, mut uba, mut ier) = (false, false, IerEnum::LocalMinimizerFound);

    // Unboundedness determination
    if alpu == f64::NEG_INFINITY && (pTGp < 0.0 || (pTGp == 0.0 && gTp > 0.0)) {
        ier = IerEnum::Unbounded;
        lba = true;
    }
    if alpo == f64::INFINITY && (pTGp < 0.0 || (pTGp == 0.0 && gTp < 0.0)) {
        ier = IerEnum::Unbounded;
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
        alp = -gTp / pTGp;   // unconstrained optimal step
        lba = alp <= alpu;   // lower bound active
        uba = alp >= alpo;   // upper bound active
    }

    // Apply bound if necessary
    if lba { alp = alpu; }
    if uba { alp = alpo; }

    (alp, lba, uba, ier)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinities() {
        // Matlab equivalent test
        //
        // alpu = -inf;
        // alpo = inf;
        // gTp = 1.0;
        // pTGp = -2.0;
        // [alp_out, lba_out, uba_out, ier_out] = getalp(alpu, alpo, gTp, pTGp);
        //
        // format long g;
        // disp(alp_out);
        // disp(lba_out);
        // disp(uba_out);
        // disp(ier_out);

        let alpu = f64::NEG_INFINITY;
        let alpo = f64::INFINITY;
        let gTp = 1.0;
        let pTGp = -2.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp.is_nan(), lba, uba, ier), (true, true, true, IerEnum::Unbounded));
    }

    #[test]
    fn test_1() {
        let alpu = f64::NEG_INFINITY;
        let alpo = 1.0;
        let gTp = -1.0;
        let pTGp = 0.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (1.0, false, true, IerEnum::LocalMinimizerFound));
    }

    #[test]
    fn test_2() {
        let alpu = -1.0;
        let alpo = f64::INFINITY;
        let gTp = 1.0;
        let pTGp = 0.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert_eq!((alp, lba, uba, ier), (-1.0, true, false, IerEnum::LocalMinimizerFound));
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
        assert!((alp.is_nan(), lba, uba, ier) == (true, true, true, IerEnum::Unbounded));
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
        assert!((alp.is_nan(), lba, uba, ier) == (true, true, false, IerEnum::Unbounded));
    }

    #[test]
    fn test_unbounded_minimum_infinite_upper_bound() {
        let alpu = -10.0;
        let alpo = f64::INFINITY;
        let gTp = -5.0;
        let pTGp = -1.0;
        let (alp, lba, uba, ier) = getalp(alpu, alpo, gTp, pTGp);
        assert!((alp.is_nan(), lba, uba, ier) == (true, false, true, IerEnum::Unbounded));
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