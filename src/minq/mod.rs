mod getalp;
mod ldlrk1;
mod ldlup;
mod ldldown;
mod minqsub;

use crate::mcs_utils::helper_funcs::clamp_SVector_mut;
use crate::minq::{getalp::getalp, minqsub::minqsub};

use nalgebra::{Const, DimMin, SMatrix, SVector};

#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
pub enum IerEnum {
    LocalMinimizerFound, // 0
    Unbounded,           // 1
    MaxitExceeded,       // 99
    InputError,          // -1
}


const ONE_PLUS_HEPS: f64 = 1. + f64::EPSILON * 100.0;

/// Minimizes an affine quadratic form subject to simple bounds,
/// using coordinate searches and reduced subspace minimizations
/// using LDL^T factorization updates
///
///    min    `fct = gam + c^T x + 0.5 x^T G x`
///
///    s.t.   `x in [xu,xo]`    `xu<=xo` is assumed
///
/// # Arguments
/// * `gam` - number, representing `gam` term of fct
///
/// * `c` - 1 x N matrix, representing `c` term of fct
///
/// * `G` - Symmetric N x N matrix, not necessarily definite
/// (if G is indefinite, only a local minimum is found).
/// If G is sparse, it is assumed that the ordering is such that
/// a sparse modified Cholesky factorization is feasible
///
/// * `xu` - Lower bounds vector of size `N` defining the initial box
/// * `xo` - Upper bounds vector of size `N` defining the initial box
///
/// # Returns
/// * `x` - Minimizer (but unbounded direction if ier=1)
/// * `fct`	- Function value
/// * `ier` - IerEnum:
///     0  (local minimizer found);
/// 	1  (unbounded);
/// 	99 (maxit exceeded);
/// 	-1 (input error)
pub fn minq<const N: usize>(
    gam: f64,
    c: &SVector<f64, N>,
    G: &SMatrix<f64, N, N>,
    xu: &SVector<f64, N>,
    xo: &SVector<f64, N>,
) -> (
    SVector<f64, N>,  // x
    f64,              // fct
    IerEnum           // ier
)
where
    Const<N>: DimMin<Const<N>, Output=Const<N>>,
{
    // Maximal number of iterations this limits the work to about 1+4*maxit/n matrix multiplies; usually at most 2*n iterations are needed for convergence
    let maxit = 3 * N;
    // Maximal number of iterative refinement steps
    let nitrefmax: usize = 3;

    let mut x = SVector::<f64, N>::zeros();

    // Regularization for low rank problems // TODO: WTF
    // .clone() matrix with changed diag
    let G = SMatrix::<f64, N, N>::from_fn(|row, col| if row != col { G[(row, col)] } else { ONE_PLUS_HEPS * G[(row, col)] });
    
    let mut K = SVector::<bool, N>::repeat(false);
    // issparse() sparse matrices are used in MATLAB to efficiently store and operate on matrices that are mostly zeros -- no influence other than performance in Matlab
    let mut L = SMatrix::<f64, N, N>::identity();
    let mut d = SVector::<f64, N>::repeat(1.0);
    let mut free = SVector::<bool, N>::repeat(false);
    let mut nfree: usize = 0;
    let mut nfree_old_option: Option<usize> = None;
    let mut fct = f64::INFINITY; // best function value
    let mut nsub = 0_usize; // number of subspace steps
    let mut unfix = true; // allow variables to be freed in csearch?
    let mut nitref = 0_usize; // no iterative refinement steps so far
    let mut improvement = true; // improvement expected

    let mut ier = IerEnum::LocalMinimizerFound;
    let (mut alp, mut alpu, mut alpo, mut lba, mut uba) = (f64::NAN, f64::NAN, f64::NAN, true, true);

    clamp_SVector_mut(&mut x, &xu, &xo);

    let mut g: SVector<f64, N>;
    'main: loop {
        debug_assert!(!x.iter().any(|&val| val.is_infinite()));

        g = G * x + c;

        let fctnew = gam + 0.5 * x.dot(&(c + g)); // .dot() does transpose first (i.e self.transpose() * rhs)

        if !improvement ||
            nitref > nitrefmax ||
            (nitref > 0 && nfree_old_option.map_or(false, |nfree_old| nfree_old == nfree) && fctnew >= fct) {
            ier = IerEnum::LocalMinimizerFound;
            break 'main;
        }

        fct = if nitref == 0 { fct.min(fctnew) } else { fctnew };

        if nitref == 0 && nsub >= maxit {
            ier = IerEnum::MaxitExceeded;
            break 'main;
        }

        // Coordinate search
        let mut count = 0_usize;
        let (mut k, mut set_k) = (0, true); // set_k is a trick to avoid k=-1 initialization. set_k maps 0 -> 0 once, so the effect of incrementing k is the same

        'coord_search: loop {
            // Combined loop for coordinate search
            while count <= N {
                count += 1;
                k = if set_k {
                    set_k = false;
                    0
                } else if k + 1 == N {
                    0
                } else {
                    k + 1
                };

                if free[k] || unfix {
                    break;
                }
            }

            if count > N {
                break 'coord_search;
            }

            // Get column view once
            let q = G.column(k);
            (alpu, alpo) = (xu[k] - x[k], xo[k] - x[k]);

            // Find step size
            (alp, lba, uba, ier) = getalp(alpu, alpo, g[k], q[k]);
            if ier != IerEnum::LocalMinimizerFound {
                x = SVector::zeros();
                x[k] = if lba { -1.0 } else { 1.0 };
                return (x, fct, ier);
            }

            let xnew = x[k] + alp;

            // Optimized bounds handling
            if lba || xnew <= xu[k] {
                if alpu != 0.0 {
                    x[k] = xu[k];
                    g.axpy(alpu, &q, 1.0);
                    count = 0;
                }
                free[k] = false;
            } else if uba || xnew >= xo[k] {
                if alpo != 0.0 {
                    x[k] = xo[k];
                    g.axpy(alpo, &q, 1.0);
                    count = 0;
                }
                free[k] = false;
            } else if alp != 0.0 {
                x[k] = xnew;
                g.axpy(alp, &q, 1.0);
                free[k] = true;
            }
        }

        // Count free variables using SIMD-optimized operation
        nfree = free.iter().filter(|&&b| b).count();

        if unfix && nfree_old_option.map_or(false, |nfree_old| nfree_old == nfree) {
            g = G * x + c;
            nitref += 1;
        } else {
            nitref = 0;
        }
        nfree_old_option = Some(nfree);
        let gain_cs = fct - gam - 0.5 * x.dot(&(c + g));
        improvement = gain_cs > 0.0 || !unfix;

        match !improvement || nitref > nitrefmax {
            true => {}
            false => {
                if nfree == 0 {
                    unfix = true;
                } else {
                    minqsub(&mut nsub, &mut free, &mut L, &mut d, &mut K, &G, &mut g,
                            &mut x, xo, xu, &mut nfree, &mut unfix, &mut alp, &mut alpu, &mut alpo,
                            &mut lba, &mut uba, &mut ier);

                    if ier != IerEnum::LocalMinimizerFound {
                        return (x, fct, ier);
                    }
                }
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
    fn test_matlab_0() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 0.1;
        // c = [-1.0; -2.0; -3.0];
        // G = [
        //     [-1.0, 0.0, 0.0];
        //     [0.0, -2.0, 0.0];
        //     [0.0, 0.0, -1.0]
        //   ];
        // xu = [-5.0; -5.0; -5.0];
        // xo = [5.0; 5.0; 5.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 0.1;
        let c = SVector::<f64, 3>::from_row_slice(&[-1.0, -2.0, -3.0]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -1.0, 0.0, 0.0,
            0.0, -2.0, 0.0,
            0.0, 0.0, -1.0
        ]);
        let xu = SVector::<f64, 3>::from_row_slice(&[-5.0, -5.0, -5.0]);
        let xo = SVector::<f64, 3>::from_row_slice(&[5.0, 5.0, 5.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [5., 5., 5.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -79.90000000000111);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_matlab_1() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 5.0;
        // c = [0.01; 0.03; 0.04];
        // G = [
        //     [1.0, 0.0, 0.0];
        //     [0.0, -1.0, 0.0];
        //     [0.0, 0.0, 1.0]
        //   ];
        // xu = [-20.0; -20.0; -20.0];
        // xo = [20.0; 20.0; 20.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 5.0;
        let c = SVector::<f64, 3>::from_row_slice(&[0.01, 0.03, 0.04]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 1.0
        ]);
        let xu = SVector::<f64, 3>::from_row_slice(&[-20.0, -20.0, -20.0]);
        let xo = SVector::<f64, 3>::from_row_slice(&[20.0, 20.0, 20.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-0.009999999999999778, -20., -0.03999999999999911];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -195.60085000000444);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
    #[test]
    fn test_extreme_negative_curvature() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = -0.01;
        // c = [0.01; 0.012; 0.014; 0.015];
        // G = [
        //     [-1.0, -10.0, 30.0,-2.0];
        //     [1.1, -2.0, 0.0, 111.0];
        //     [2.0, -0.01, 3.0, 0.0];
        //     [-3.0, 30.0, 0.0, -4.0]
        //   ];
        // xu = [-100.0; -100.0; -100.0; -100.0];
        // xo = [100.0; 100.0; 100.0; 100.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = -0.01;
        let c = SVector::<f64, 4>::from_row_slice(&[0.01, 0.012, 0.014, 0.015]);
        let mut G = SMatrix::<f64, 4, 4>::from_row_slice(&[
            -1.0, -10.0, 30.0, -2.0,
            1.1, -2.0, 0.0, 111.0,
            2.0, -0.01, 3.0, 0.0,
            -3.0, 30.0, 0.0, -4.0
        ]);
        let xu = SVector::<f64, 4>::from_row_slice(&[-100.0, -100.0, -100.0, -100.0]);
        let xo = SVector::<f64, 4>::from_row_slice(&[100.0, 100.0, 100.0, 100.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-100., 100., 66.99533333333184, -100.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -820993.8410326652);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_tight_bounds() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 10.0;
        // c = [5.0; -3.0; 2.0; 1.0];
        // G = [
        //     [4.0, 0.5, 0.1, 0.2];
        //     [0.5, 3.0, 0.6, 0.3];
        //     [0.1, 0.6, 2.0, 0.4];
        //     [0.2, 0.3, 0.4, 5.0]
        //   ];
        // xu = [1.0; 1.0; 1.0; 1.0];
        // xo = [1.0; 1.0; 1.0; 1.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 10.0;
        let c = SVector::<f64, 4>::from_row_slice(&[5.0, -3.0, 2.0, 1.0]);
        let mut G = SMatrix::<f64, 4, 4>::from_row_slice(&[
            4.0, 0.5, 0.1, 0.2,
            0.5, 3.0, 0.6, 0.3,
            0.1, 0.6, 2.0, 0.4,
            0.2, 0.3, 0.4, 5.0
        ]);
        let xu = SVector::<f64, 4>::from_row_slice(&[1.0, 1.0, 1.0, 1.0]);
        let xo = SVector::<f64, 4>::from_row_slice(&[1.0, 1.0, 1.0, 1.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [1., 1., 1., 1.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, 24.100000000000158);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_large_scale() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 1.0;
        // n = 10;
        // c = ones(n, 1);
        // G = diag(2*ones(n, 1));
        // for i=1:n-1
        //   G(i,i+1) = -1;
        //   G(i+1,i) = -1;
        // end
        // xu = -5 * ones(n, 1);
        // xo = 5 * ones(n, 1);
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        const N: usize = 10;
        let gam = 1.0;
        let c = SVector::<f64, N>::from_element(1.0);

        let mut G = SMatrix::<f64, N, N>::from_element(0.0);
        // Set diagonal elements to 2.0
        for i in 0..N {
            G[(i, i)] = 2.0;
        }
        // Set off-diagonal elements
        for i in 0..N - 1 {
            G[(i, i + 1)] = -1.0;
            G[(i + 1, i)] = -1.0;
        }

        let xu = SVector::<f64, N>::from_element(-5.0);
        let xo = SVector::<f64, N>::from_element(5.0);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-2.666666466718111, -4.333333233358959, -5., -5., -5., -5., -5., -5., -4.333332963260136, -2.666666481630009, ];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -32.666666666661285);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_large_scale_2() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = -2.01;
        // c = ones(11, 1);
        // G = [
        //     [-5.0, 2.0, 3.0, -5.0, 9.0, 8.0, -5.0, -2.0, -4.0, 6.0, -9.0];
        //     [4.0, -5.0, 5.0, -8.0, 0.0, -4.0, 2.0, -3.0, 1.0, -9.0, 1.0];
        //     [-4.0, -9.0, -3.0, -3.0, 5.0, -3.0, 1.0, -9.0, -4.0, 5.0, 1.0];
        //     [-7.0, -8.0, -8.0, 3.0, -10.0, -2.0, -1.0, -4.0, -2.0, 3.0, -3.0];
        //     [-8.0, 6.0, -6.0, 3.0, -1.0, -8.0, -8.0, 5.0, -6.0, -6.0, -7.0];
        //     [-1.0, -7.0, -2.0, 4.0, 5.0, -6.0, -8.0, -10.0, 6.0, 1.0, 5.0];
        //     [8.0, 7.0, -6.0, -8.0, -6.0, -7.0, 5.0, -6.0, -8.0, -10.0, 7.0];
        //     [6.0, -7.0, 0.0, 5.0, 6.0, -2.0, -9.0, 5.0, -9.0, -10.0, 0.0];
        //     [9.0, -8.0, 6.0, -2.0, 6.0, 7.0, 7.0, 6.0, 1.0, -4.0, -7.0];
        //     [-1.0, -7.0, 8.0, 6.0, 0.0, -9.0, -6.0, 4.0, 1.0, -2.0, 2.0];
        //     [3.0, 7.0, -9.0, -7.0, -4.0, 9.0, -6.0, 4.0, 8.0, 7.0, -9.0]
        // ];
        // xu = -15 * ones(11, 1);
        // xo = 15 * ones(11, 1);
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        const N: usize = 11;
        let gam = -2.01;
        let c = SVector::<f64, N>::repeat(1.0);

        let mut G = SMatrix::<f64, N, N>::from_row_slice(&[
            -5.0, 2.0, 3.0, -5.0, 9.0, 8.0, -5.0, -2.0, -4.0, 6.0, -9.0,
            4.0, -5.0, 5.0, -8.0, 0.0, -4.0, 2.0, -3.0, 1.0, -9.0, 1.0,
            -4.0, -9.0, -3.0, -3.0, 5.0, -3.0, 1.0, -9.0, -4.0, 5.0, 1.0,
            -7.0, -8.0, -8.0, 3.0, -10.0, -2.0, -1.0, -4.0, -2.0, 3.0, -3.0,
            -8.0, 6.0, -6.0, 3.0, -1.0, -8.0, -8.0, 5.0, -6.0, -6.0, -7.0,
            -1.0, -7.0, -2.0, 4.0, 5.0, -6.0, -8.0, -10.0, 6.0, 1.0, 5.0,
            8.0, 7.0, -6.0, -8.0, -6.0, -7.0, 5.0, -6.0, -8.0, -10.0, 7.0,
            6.0, -7.0, 0.0, 5.0, 6.0, -2.0, -9.0, 5.0, -9.0, -10.0, 0.0,
            9.0, -8.0, 6.0, -2.0, 6.0, 7.0, 7.0, 6.0, 1.0, -4.0, -7.0,
            -1.0, -7.0, 8.0, 6.0, 0.0, -9.0, -6.0, 4.0, 1.0, -2.0, 2.0,
            3.0, 7.0, -9.0, -7.0, -4.0, 9.0, -6.0, 4.0, 8.0, 7.0, -9.,
        ]);

        let xu = SVector::<f64, N>::from_element(-15.0);
        let xo = SVector::<f64, N>::from_element(15.0);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-15., 15., 15., 15., 15., 15., 15., 15., -15., 15., 15., ];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -17672.010000000042);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_large_scale_3() {
        //         Matlab Equivalent test:
        //
        //         format long g;
        //         prt = 0;
        //         gam = 0.01;
        //         c = zeros(11, 1);
        //         G = [
        // [3.0, -8.0, 2.0, 5.0, 7.0, -6.0, -9.0, -3.0, 6.0, 2.0, -4.0];
        //  [8.0, 6.0, 6.0, 0.0, 4.0, -10.0, -1.0, -7.0, -8.0, -7.0, -3.0];
        //  [-2.0, 9.0, 4.0, 5.0, -3.0, 6.0, 3.0, -6.0, -10.0, 4.0, -9.0];
        //  [5.0, 1.0, 5.0, -9.0, -7.0, -2.0, -10.0, 8.0, -5.0, -4.0, 7.0];
        //  [1.0, 7.0, 9.0, 0.0, -5.0, 1.0, 6.0, 7.0, 0.0, 5.0, 3.0];
        //  [1.0, -8.0, -5.0, 5.0, -6.0, -10.0, -6.0, -2.0, -2.0, -1.0, -1.0];
        //  [-6.0, -9.0, 6.0, 1.0, 6.0, -3.0, 0.0, -6.0, -9.0, 2.0, 7.0];
        //  [-2.0, -6.0, -6.0, 6.0, 7.0, -5.0, 7.0, -5.0, 3.0, -4.0, 1.0];
        //  [-10.0, 8.0, 8.0, 5.0, 1.0, 4.0, 5.0, 8.0, -5.0, -2.0, 2.0];
        //  [-4.0, -6.0, 8.0, 5.0, -1.0, -10.0, -4.0, 7.0, 6.0, 2.0, 1.0];
        //  [1.0, -3.0, 8.0, -4.0, 5.0, 1.0, 2.0, -2.0, -7.0, -7.0, -10.0]
        //      ];
        //         xu = -10 * ones(11, 1);
        //         xo = 10 * ones(11, 1);
        //
        //         [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        //         x = sprintf('%.16g,', x);
        //         disp(x);
        //         disp(fct);
        //         disp(ier);

        const N: usize = 11;
        let gam = 0.01;
        let c = SVector::<f64, N>::repeat(0.0);

        let mut G = SMatrix::<f64, N, N>::from_row_slice(&[
            3.0, -8.0, 2.0, 5.0, 7.0, -6.0, -9.0, -3.0, 6.0, 2.0, -4.0,
            8.0, 6.0, 6.0, 0.0, 4.0, -10.0, -1.0, -7.0, -8.0, -7.0, -3.0,
            -2.0, 9.0, 4.0, 5.0, -3.0, 6.0, 3.0, -6.0, -10.0, 4.0, -9.0,
            5.0, 1.0, 5.0, -9.0, -7.0, -2.0, -10.0, 8.0, -5.0, -4.0, 7.0,
            1.0, 7.0, 9.0, 0.0, -5.0, 1.0, 6.0, 7.0, 0.0, 5.0, 3.0,
            1.0, -8.0, -5.0, 5.0, -6.0, -10.0, -6.0, -2.0, -2.0, -1.0, -1.0,
            -6.0, -9.0, 6.0, 1.0, 6.0, -3.0, 0.0, -6.0, -9.0, 2.0, 7.0,
            -2.0, -6.0, -6.0, 6.0, 7.0, -5.0, 7.0, -5.0, 3.0, -4.0, 1.0,
            -10.0, 8.0, 8.0, 5.0, 1.0, 4.0, 5.0, 8.0, -5.0, -2.0, 2.0,
            -4.0, -6.0, 8.0, 5.0, -1.0, -10.0, -4.0, 7.0, 6.0, 2.0, 1.0,
            1.0, -3.0, 8.0, -4.0, 5.0, 1.0, 2.0, -2.0, -7.0, -7.0, -10.,
        ]);

        let xu = SVector::<f64, N>::from_element(-10.0);
        let xo = SVector::<f64, N>::from_element(10.0);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-10., -10., 10., 10., 10., -10., -10., -10., 10., -10., -10.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -11849.990000000033);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_indefinite_matrix() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 0.0;
        // c = [1.0; -2.0; 3.0];
        // G = [
        //     [2.0, 1.0, 0.5];
        //     [1.0, -3.0, 1.0];
        //     [0.5, 1.0, 2.0]
        //   ];
        // xu = [-1.0; -1.0; -1.0];
        // xo = [1.0; 1.0; 1.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 0.0;
        let c = SVector::<f64, 3>::from_row_slice(&[1.0, -2.0, 3.0]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[
            2.0, 1.0, 0.5,
            1.0, -3.0, 1.0,
            0.5, 1.0, 2.0
        ]);
        let xu = SVector::<f64, 3>::from_row_slice(&[-1.0, -1.0, -1.0]);
        let xo = SVector::<f64, 3>::from_row_slice(&[1.0, 1.0, 1.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-0.7499999999999833, 1., -1.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -7.062499999999999);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_asymmetric_bounds() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 5.0;
        // c = [2.0; 3.0; -1.0; 4.0];
        // G = [
        //     [4.0, 0.0, 0.0, 0.0];
        //     [0.0, 6.0, 0.0, 0.0];
        //     [0.0, 0.0, 8.0, 0.0];
        //     [0.0, 0.0, 0.0, 10.0]
        //   ];
        // xu = [-10.0; -5.0; 0.0; 2.0];
        // xo = [-5.0; 0.0; 5.0; 10.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 5.0;
        let c = SVector::<f64, 4>::from_row_slice(&[2.0, 3.0, -1.0, 4.0]);
        let mut G = SMatrix::<f64, 4, 4>::from_row_slice(&[
            4.0, 0.0, 0.0, 0.0,
            0.0, 6.0, 0.0, 0.0,
            0.0, 0.0, 8.0, 0.0,
            0.0, 0.0, 0.0, 10.0
        ]);
        let xu = SVector::<f64, 4>::from_row_slice(&[-10.0, -5.0, 0.0, 2.0]);
        let xo = SVector::<f64, 4>::from_row_slice(&[-5.0, 0.0, 5.0, 10.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-5., -0.4999999999999889, 0.1249999999999972, 2.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, 72.18750000000156);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_zero_diagonal() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 3.0;
        // c = [1.0; 2.0; 3.0];
        // G = [
        //     [0.0, 1.0, 2.0];
        //     [1.0, 0.0, 3.0];
        //     [2.0, 3.0, 0.0]
        //   ];
        // xu = [-2.0; -2.0; -2.0];
        // xo = [2.0; 2.0; 2.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 3.0;
        let c = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[
            0.0, 1.0, 2.0,
            1.0, 0.0, 3.0,
            2.0, 3.0, 0.0
        ]);
        let xu = SVector::<f64, 3>::from_row_slice(&[-2.0, -2.0, -2.0]);
        let xo = SVector::<f64, 3>::from_row_slice(&[2.0, 2.0, 2.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-2., -2., 2.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -13.);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_mixed_bounds() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 7.5;
        // c = [1.0; -1.0; 0.5; -0.5; 2.0];
        // G = eye(5);
        // for i=1:4
        //   G(i,i+1) = 0.1;
        //   G(i+1,i) = 0.1;
        // end
        // xu = [-Inf; -Inf; 0.0; -Inf; -Inf]; % Some unbounded below
        // xo = [Inf; 1.0; Inf; 0.0; Inf]; % Some unbounded above
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 7.5;
        let c = SVector::<f64, 5>::from_row_slice(&[1.0, -1.0, 0.5, -0.5, 2.0]);

        let mut G = SMatrix::<f64, 5, 5>::identity();
        for i in 0..4 {
            G[(i, i + 1)] = 0.1;
            G[(i + 1, i)] = 0.1;
        }

        let xu = SVector::<f64, 5>::from_row_slice(&[f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY, f64::NEG_INFINITY]);
        let xo = SVector::<f64, 5>::from_row_slice(&[f64::INFINITY, 1.0, f64::INFINITY, 0.0, f64::INFINITY]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [-1.099999999999976, 1., 0., 0., -1.999999999999956];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, 4.395000000000069);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }


    #[test]
    fn test_matlab_random_0() {
        // Matlab Equivalent test:
        //
        // prt = 0;
        // gam = -2.;
        // c = [2.; -1.; 0.; -4.; 0.; 0.];
        // G = [
        //     [95., 1., 0., 4., 0., 0.];
        //     [1., 45., 0., -3., 0., 0.];
        //     [-0., -0., 0., -0., 0., 0.];
        //     [4., -3., 0., 50., 0., 0.];
        //     [0., 0., 0., 0., 0., 0.];
        //     [0., 0., 0., 0., 0., 80.]
        //   ];
        // xu = [0.;0.;-0.5;-0.1;-0.07;-0.019];
        // xo = [0.25;0.;0.1;0.1;0.07;0.1];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = -2.;
        let c = SVector::<f64, 6>::from_row_slice(&[2., -1., 0., -4., 0., 0.]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            95., 1., -0., 4., 0., 0.,
            1., 45., -0., -3., 0., -0.,
            -0., -0., 0., -0., 0., 0.,
            4., -3., -0., 50., 0., -0.,
            0., 0., 0., 0., 0., -0.,
            0., -0., 0., -0., -0., 80.
        ]);
        let xu = SVector::<f64, 6>::from_row_slice(&[0., 0., -0.5, -0.1, -0.07, -0.019]);
        let xo = SVector::<f64, 6>::from_row_slice(&[0.25, 0., 0.1, 0.1, 0.07, 0.1]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = [0., 0., 0., 0.07999999999999823, 0., 0.];

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -2.1599999999999966);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_coverage_3() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = -5.0;
        // c = [0.01;0.1;0.7;0.3;0.9;-0.5];
        // G = [
        //     [13., 0.08, 0.105, -0.9, 0.7, -0.4];
        //     [1., 0.08, 0.105, 0.9, 0.7, -0.4];
        //     [0.1, 0.08, 0.105, -0.9, 0.7, -0.4];
        //     [0.2, 0.08, 0.105, 0.9, 0.7, -0.4];
        //     [-0.3, 0.08, 0.105, 0.9, 0.7, -0.4];
        //     [4., 0.08, 0.105, -0.9, 0.7, -0.4]
        //   ];
        // xu = [-3.;-3.;-3.;-3.;-3.;-3.];
        // xo = [3.;3.;3.;3.;3.;3.];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

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
        let xu = SVector::<f64, 6>::from_row_slice(&[-3.; 6]);
        let xo = SVector::<f64, 6>::from_row_slice(&[3.; 6]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[0.4549999999999899, -3., 3., 3., -3., 3.]);

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(fct, -21.028224999999914, epsilon = TOLERANCE);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 2.0;
        // c = [1.0; 0.1; 0.2; 0.3; 0.4; 0.6];
        // G = [
        //     [1., 0.08, 0.1, -0.3, 0.7, -0.4];
        //     [1., 0.1, -0.1, 0.1, 0.7, -0.4];
        //     [0.1, 0.04, 0.5, -10., 0.6, -0.2];
        //     [0.2, 0.08, -0.1, 0.4, 0.6, -0.1];
        //     [-0.3, -0.3, 0.1, 0.2, 0.0, -0.3];
        //     [4., 0.4, 0.05, -0.9, 0.0, -0.4];
        //   ];
        // xu = [-10.; -9.; -8.; -7.; -6.; -5.];
        // xo = [10.; 9.; 8.; 7.; 6.; 5.];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 2.0;
        let c = SVector::<f64, 6>::from_row_slice(&[1.0, 0.1, 0.2, 0.3, 0.4, 0.6]);
        let mut G = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1., 0.08, 0.1, -0.3, 0.7, -0.4,
            1., 0.1, -0.1, 0.1, 0.7, -0.4,
            0.1, 0.04, 0.5, -10., 0.6, -0.2,
            0.2, 0.08, -0.1, 0.4, 0.6, -0.1,
            -0.3, -0.3, 0.1, 0.2, 0.0, -0.3,
            4., 0.4, 0.05, -0.9, 0.0, -0.4,
        ]);
        let xu = SVector::<f64, 6>::from_row_slice(&[-10.0, -9.0, -8.0, -7.0, -6.0, -5.0]);
        let xo = SVector::<f64, 6>::from_row_slice(&[10.0, 9.0, 8.0, 7.0, 6.0, 5.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[-5.219999999999884, 9., -8., -7., 6., 5.]);

        assert_eq!(x, expected_x);
        assert_eq!(fct, -331.51999999999794);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_difficult_cond_2() {
        // Matlab Equivalent test:
        //
        // format long g;
        // prt = 0;
        // gam = 200.0;
        // c = [1.0; 2.0; 3.0];
        // G = [
        //     [1.0, 2.0, 4.0];
        //     [3.0, 5.0, -1.0];
        //     [0.0, -3.0, -10.0];
        //   ];
        // xu = [0.0; 0.0; -3.0];
        // xo = [1.0; 2.0; 4.0];
        //
        // [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt);
        //
        // x = sprintf('%.16g,', x);
        // disp(x);
        // disp(fct);
        // disp(ier);

        let gam = 200.0;
        let c = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        let mut G = SMatrix::<f64, 3, 3>::from_row_slice(&[
            1.0, 2.0, 4.0,
            3.0, 5.0, -1.0,
            0.0, -3.0, -10.0
        ]);
        let xu = SVector::<f64, 3>::from_row_slice(&[0.0, 0.0, -3.0]);
        let xo = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 4.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 3>::from_row_slice(&[0., 0.39999999999999114, 4.]);

        assert_eq!(x, expected_x);
        assert_eq!(fct, 129.99999999999827);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

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
        let xu = SVector::<f64, 6>::from_row_slice(&[-0.125, -0.06239714070289448, -0.11155191538152609, -0.1875, -0.07153472077076538, -0.019673386763705048]);
        let xo = SVector::<f64, 6>::from_row_slice(&[0.125, 0.06239714070289448, 0.11155191538152609, 0.1875, 0.07153472077076538, 0.019673386763705048]);

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
        let xu = SVector::<f64, 6>::from_row_slice(&[-1.; 6]);
        let xo = SVector::<f64, 6>::from_row_slice(&[1.; 6]);

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
        let xu = SVector::<f64, 6>::from_row_slice(&[-1.; 6]);
        let xo = SVector::<f64, 6>::from_row_slice(&[1.; 6]);

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
        let xu = SVector::<f64, 6>::from_row_slice(&[-2.; 6]);
        let xo = SVector::<f64, 6>::from_row_slice(&[2.; 6]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        let expected_x = SVector::<f64, 6>::from_row_slice(&[0.302259618771836, -2.0, 2.0, 1.9883867513839923, -2.0, 2.0]);

        assert_relative_eq!(x.as_slice(), expected_x.as_slice(), epsilon = TOLERANCE);
        assert_eq!(fct, -14.442743529914747);
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
        let xu = SVector::<f64, 6>::from_row_slice(&[-3.; 6]);
        let xo = SVector::<f64, 6>::from_row_slice(&[3.; 6]);

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
        let xu = SVector::<f64, 3>::from_row_slice(&[0.0, -2.0, 1.0]);
        let xo = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 5.0]);

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
        let xu = SVector::<f64, 6>::from_row_slice(&[0.9558285681648798, 0.5194757464362438, 0.3015417041472618, 0.8101331118648574, 0.061618326392057776, 0.5421792878798574]);
        let xo = SVector::<f64, 6>::from_row_slice(&[2.172274965654365, 2.8179460480833494, 2.1123039892888325, 2.4283704885558492, 2.500248662862857, 2.874993639004977]);

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
        let xu = SVector::<f64, 6>::from_row_slice(&[-4.769677847426308, -3.5969155931370618, -1.9037195727875393, -1.9430412656953555, -3.076515775656527, -3.3371066538862233]);
        let xo = SVector::<f64, 6>::from_row_slice(&[-3.7696778474263084, -2.5969155931370618, -0.9037195727875393, -0.9430412656953555, -2.076515775656527, -2.3371066538862233]);

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
        let xo = SVector::<f64, 6>::from_row_slice(&[0.20094711239564478, 0.1495167421889697, 0.3647846775, 0.2559626362565812, 0.331602309105488, 0.3724789161602837]);
        let xu = SVector::<f64, 6>::from_row_slice(&[-0.20094711239564478, -0.1495167421889697, -0.3647846775, -0.2559626362565812, -0.331602309105488, -0.3724789161602837]);

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
        let xu = SVector::<f64, 1>::from_row_slice(&[f64::NEG_INFINITY]);
        let xo = SVector::<f64, 1>::from_row_slice(&[f64::INFINITY]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, SVector::<f64, 1>::from_row_slice(&[1.0]));
        assert_eq!(fct, 1.0);
        assert_eq!(ier, IerEnum::Unbounded);
    }

    #[test]
    fn test_global_return_0() {
        let gam = 0.0;
        let c = SVector::<f64, 2>::from_row_slice(&[-1.0, 20.0]);
        let mut G = SMatrix::<f64, 2, 2>::from_row_slice(&[123.0, 26.0, -0.3, -9.5]);
        let xu = SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]);
        let xo = SVector::<f64, 2>::from_row_slice(&[11.0, 22.0]);

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
        let xu = SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]);
        let xo = SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);

        assert_eq!(x, SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]));
        assert_eq!(fct, 1.0);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }

    #[test]
    fn test_global_return_4() {
        let gam = -200.0;
        let c = SVector::<f64, 2>::from_row_slice(&[-1.0, -2.0]);
        let mut G = SMatrix::<f64, 2, 2>::from_row_slice(&[-1.0, -2.0, 5.0, -1.0]);
        let xu = SVector::<f64, 2>::from_row_slice(&[0.0, 0.0]);
        let xo = SVector::<f64, 2>::from_row_slice(&[10.0, 20.0]);

        let (x, fct, ier) = minq(gam, &c, &mut G, &xu, &xo);
        let expected_x = SVector::<f64, 2>::from_row_slice(&[10.0, 0.0]);

        assert_eq!(x, expected_x);
        assert_eq!(fct, -260.00000000000114);
        assert_eq!(ier, IerEnum::LocalMinimizerFound);
    }
}
