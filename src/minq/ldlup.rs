use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{SMatrix, SVector};


/// Updates an LDL^T factorization by replacing the j-th row and column with a new vector `g`
///
/// This function updates the LDL^T factorization of a matrix in-place when the j-th row and
/// column (originally represented by a unit vector) are replaced by the vector `g`. The updated
/// factors (L and d) correspond to the new matrix A' where the j-th row and column have been modified.
///
/// The algorithm computes intermediate quantities (such as vectors `u` and `v`) to evaluate a pivot
/// value (`delta`). If `delta` is sufficiently positive, the update is performed directly on L and d,
/// ensuring that A' remains positive definite, and the function returns `None`. If `delta` is non-positive
/// (up to a tolerance scaled by N), a curvature direction is identified and returned as a vector `p`,
/// while leaving L and d unchanged. This curvature direction may indicate a null or negative curvature,
/// which is important in optimization contexts.
///
/// # Arguments
/// * `L` - A mutable lower triangular matrix with unit diagonal from the LDL^T factorization,
///         which will be updated in-place.
/// * `d` - A mutable vector containing the diagonal elements of D (must be positive), also updated in-place.
/// * `j` - The index (0-based) of the row/column to be replaced.
/// * `g` - The column vector that replaces the j-th unit vector. **Note:** `g` must have zeros in positions
///         corresponding to other unit rows to ensure the correctness of the update.
///
/// # Returns
/// * `None` if the updated matrix A' is positive definite, meaning the in-place update was successful.
/// * `Some(p)` where `p` is a vector representing a direction of null or negative curvature, indicating that
///    A' is not positive definite.
///
/// # Mathematical Background
/// Given an original factorization A = LDL^T, replacing the j-th row and column by `g` yields a new matrix A'.
/// The function computes the factors L' and d' such that A' = L' D' (L')^T. A key step involves computing
/// a pivot value `delta = g[j] - u.dot(v)` (with appropriate intermediate solves), which determines whether the
/// update maintains positive definiteness. In cases where `delta` is non-positive (within a tolerance scaled by N),
/// the function identifies and returns a curvature direction `p` instead of updating the factorization.
///
/// # Note
/// Ensure that the vector `g` is properly constructed with zeros in other unit rows (except possibly at index j)
/// to satisfy the assumptions of the algorithm.
pub fn ldlup<const N: usize>(
    L: &mut SMatrix<f64, N, N>,
    d: &mut SVector<f64, N>,
    j: usize,
    g: &SVector<f64, N>,
) ->
    Option<SVector<f64, N>> // p
{
    let N_times_EPS = (N as f64) * f64::EPSILON;

    // Special case for j == 0
    if j == 0 {
        if g[0] <= N_times_EPS {
            let mut p = SVector::<f64, N>::zeros();
            p[0] = 1.0;
            return Some(p);
        }
        // w=g(K)/del; // will never be used afterwards
        // L(j,I)=v'; // as I=1:j-1; j==1; the range is 1:0 - empty so no changes after L(j,I)=v' - tested; consistent with range(0,0) in python version
        d[0] = g[0];
        return None;
    }

    let LII = L.view((0, 0), (j, j));
    let LII_T = LII.transpose(); // needed to satisfy the borrow checker later

    // u=LII\g(I);
    let u = LII.lu().solve(&g.rows(0, j)).unwrap();
    // v=u./d(I);
    let v = u.component_div(&d.rows(0, j));
    // del=g(j)-u'*v;
    let del = g[j] - u.dot(&v); // .dot(rhs) equal to self.transpose() * rhs

    if del <= N_times_EPS {
        let mut p = SVector::<f64, N>::zeros();
        // LII'\v
        let p_head = LII_T.lu().solve(&v).unwrap();

        p.rows_mut(0, j).copy_from(&p_head);
        p[j] = -1.0;
        return Some(p);
    }

    let K_size = N - (j + 1);
    let LKI = L.view((j + 1, 0), (K_size, j));
    let LKI_T = LKI.transpose(); // needed to satisfy the borrow checker later and to save L state before ldlrk1's L modification
    // w=(g(K)-LKI*u)/del;
    let w = (g.rows(j + 1, K_size) - &(LKI * &u)).scale(1.0 / del);

    let mut LKK = L.view_mut((j + 1, j + 1), (K_size, K_size));
    let mut dK = d.rows_mut(j + 1, K_size);

    let q = ldlrk1(&mut LKK, &mut dK, -del, w.clone());

    return match q.is_empty() {
        true => {
            // Update row j: replace its left part with v transposed, and set the diagonal element to 1.
            // The right part L[(j, j+1)..] remains unchanged (same as L(j, K) in MATLAB).
            L.view_mut((j, 0), (1, j)).copy_from_slice(v.as_slice());
            L[(j, j)] = 1.0;

            // Update the lower block: for rows j+1 through N-1, replace the j-th column with w.
            // This corresponds to replacing the middle column of the lower block [LKI, w, LKK].
            L.view_mut((j + 1, j), (K_size, 1)).copy_from(&w);

            d[j] = del;
            None
        }
        false => {
            // pi=w'*q;
            let pi = w.dot(&q); // .dot(rhs) equal to self.transpose() * rhs
            let piv_lki_q = pi * v - (LKI_T * &q);

            // LII'\(pi*v-LKI'*q)
            let p_head = LII_T.lu().solve(&piv_lki_q).unwrap();

            let mut p = SVector::<f64, N>::zeros();
            p.rows_mut(0, j).copy_from(&p_head);
            p[j] = -pi;
            p.rows_mut(j + 1, K_size).copy_from(&q);

            Some(p)
        }
    };
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ldlup_case_5() {
        // Matlab code for test_ldlup_case_5:
        // -----------------------------------
        // function [L_out,d_out,p] = ldlup(L,d,j,g)
        // L = eye(4);
        // d = [2.0; 3.0; 4.0; 5.0];
        // j = 3; % Matlab is 1-indexed; corresponds to Rust j = 2 (0-indexed)
        // g = [1.0; 0.0; 3.0; 0.0];
        // [L_out,d_out,p] = ldlup(L,d,j,g)
        // disp('L_out:'); disp(L_out)
        // disp('d_out:'); disp(d_out)
        // disp('p:'); disp(p)
        // -----------------------------------
        
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::identity();
        let mut d = SVector::<f64, N>::from_row_slice(&[2.0, 3.0, 4.0, 5.0]);
        let j = 2; // Rust index (0-indexed) corresponding to Matlab's j=3
        let g = SVector::<f64, N>::from_row_slice(&[1.0, 0.0, 3.0, 0.0]);
        let p = ldlup(&mut L, &mut d, j, &g);

        // Expected values from Matlab (replace these placeholders with actual values):
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.5, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[2.0, 3.0, 2.5, 5.0]);
        let expected_p = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_ldlup_real_mistake() {
        // Matlab code for test_ldlup_case_6:
        // -----------------------------------
        // function [L_out,d_out,p] = ldlup(L,d,j,g)
        // L = [ 1.0, 0.0, 0.0, 0.0;
        //     2.0, 3.0, 0.0, 0.0;
        //     3.0, 4.0, 2.0, 0.0;
        //     5.0, 6.0, 7.0, 4.0];
        // d = [10; 20; 30; 40];
        // j = 2; % Matlab indexing: j=2 corresponds to Rust j = 1
        // g = [0.5; 0.2; 0.8; 0.1];
        // [L_out,d_out,p] = ldlup(L,d,j,g)
        // disp('L_out:'); disp(L_out)
        // disp('d_out:'); disp(d_out)
        // disp('p:'); disp(p)
        // -----------------------------------
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            2.0, 3.0, 0.0, 0.0,
            3.0, 4.0, 2.0, 0.0,
            5.0, 6.0, 7.0, 4.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[10.0, 20.0, 30.0, 40.0]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[0.5, 0.2, 0.8, 0.1]);
        let p = ldlup(&mut L, &mut d, j, &g);

        // Expected values from Matlab (replace these placeholders with actual values):
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1., 0., 0., 0.,
            0.050000000000000, 1., 0., 0.,
            3., -3.999999999999999, 2., 0.,
            5., -13.714285714285712, 7.367647058823529, 4.,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[10., 0.17500000000000002, 27.200000000000003, 0.6092436974790232]);
        let expected_p = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_ldlup_real_del_lower_or_eq_tham_N_times_EPS() {
        // Matlab code for test_ldlup_case_6:
        // -----------------------------------
        // function [L_out,d_out,p] = ldlup(L,d,j,g)
        // L = [ 1.0, 1.1, 2.2, 3.3;
        //     2.0, 3.0, 7.0, 4.0;
        //     3.0, 4.0, 2.0, 5.5;
        //     5.0, 6.0, 7.0, 4.1];
        // d = [11; -20; 3; -40];
        // j = 3; % Matlab indexing: j=3 corresponds to Rust j = 2
        // g = [-1.5; 0.2; -0.8; -0.1];
        // [L_out,d_out,p] = ldlup(L,d,j,g)
        // disp('L_out:'); disp(L_out)
        // disp('d_out:'); disp(d_out)
        // disp('p:'); disp(p)
        // -----------------------------------
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 1.1, 2.2, 3.3,
            2.0, 3.0, 7.0, 4.0,
            3.0, 4.0, 2.0, 5.5,
            5.0, 6.0, 7.0, 4.1,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[11., -20.0, 3., -40.0]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[-1.5, 0.2, -0.8, -0.1]);
        let p = ldlup(&mut L, &mut d, j, &g);

        // Expected values from Matlab (replace these placeholders with actual values):
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 1.1, 2.2, 3.3,
            2.0, 3.0, 7.0, 4.0,
            3.0, 4.0, 2.0, 5.5,
            5.0, 6.0, 7.0, 4.1,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[11., -20., 3., -40.]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[-1.5113636363636374, 0.4875000000000004, -1.000000000000000, 0.]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_ldlup_match_false() {
        // Matlab code for test_ldlup_case_6:
        // -----------------------------------
        // function [L_out,d_out,p] = ldlup(L,d,j,g)
        // L = [ -1.0, 1.1, 2.2, 3.3;
        //     -2.0, -3.0, 7.0, 4.0;
        //     -3.0, -4.0, 2.0, 5.5;
        //     5.0, -6.0, 7.0, 4.1];
        // d = [100; 40; 30; 20];
        // j = 3;
        // g = [150; 100; 200; 400];
        // [L_out,d_out,p] = ldlup(L,d,j,g);
        // disp('L_out:'); disp(L_out)
        // disp('d_out:'); disp(d_out)
        // disp('p:'); disp(p)
        // -----------------------------------
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            -1.0, 1.1, 2.2, 3.3,
            -2.0, -3.0, 7.0, 4.0,
            -3.0, -4.0, 2.0, 5.5,
            5.0, -6.0, 7.0, 4.1,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[100., 40., 30., 20.]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[150., 100., 200., 400.]);
        let p = ldlup(&mut L, &mut d, j, &g);

        // Expected values from Matlab (replace these placeholders with actual values):
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            -1., 1.1, 2.2, 3.3,
            -2., -3., 7., 4.,
            -3., -4., 2., 5.5,
            5., -6., 7., 4.1
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[100., 40., 30., 20.]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[7.274888786623714, 0.23661604540573702, -6.062279490719435, 0.24390243902439027]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_ldlup_match_false_2() {
        // Matlab code for test_ldlup_case_6:
        // -----------------------------------
        // function [L_out,d_out,p] = ldlup(L,d,j,g)
        // L = [ -1.0, 1.1, 2.2, 3.3;
        //     -2.0, -3.0, 7.0, 4.0;
        //     -3.0, -4.0, 2.0, 5.5;
        //     -5.0, -6.0, 7.0, 4.1];
        // d = [40; 40; 30; 20];
        // j = 2;
        // g = [10; 100; 200; 400];
        // [L_out,d_out,p] = ldlup(L,d,j,g);
        // disp('L_out:'); disp(L_out)
        // disp('d_out:'); disp(d_out)
        // disp('p:'); disp(p)
        // -----------------------------------
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            -1.0, 1.1, 2.2, 3.3,
            -2.0, -3.0, 7.0, 4.0,
            -3.0, -4.0, 2.0, 5.5,
            -5.0, -6.0, 7.0, 4.1,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[40., 40., 30., 20.]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[10., 100., 200., 400.]);
        let p = ldlup(&mut L, &mut d, j, &g);

        // Expected values from Matlab (replace these placeholders with actual values):
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            -1., 1.1, 2.2, 3.3,
            -2., -3., 7., 4.,
            -3., -4., 2., 5.5,
            -5., -6., 7., 4.1
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[40., 40., 30., 20.]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[-1.282051282051282, -0.8717948717948718, 0.500000000000000, 0.]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_ldlup_match_false_3() {
        // Matlab code for test_ldlup_case_6:
        // -----------------------------------
        // function [L_out,d_out,p] = ldlup(L,d,j,g)
        // L = [ 1.0, 1.1, 1.2, 1.3;
        //     12.0, -3.0, 7.0, 0.0;
        //     3.1, 3.0, 23.0, 5.;
        //     -25.0, -36.0, 17.0, 4.1];
        // d = [25; 10; 30; 20];
        // j = 2;
        // g = [13; 101; 220; 430];
        // [L_out,d_out,p] = ldlup(L,d,j,g);
        // disp('L_out:'); disp(L_out)
        // disp('d_out:'); disp(d_out)
        // disp('p:'); disp(p)
        // -----------------------------------
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 1.1, 1.2, 1.3,
            12.0, -3.0, 7.0, 0.0,
            3.1, 3.0, 23.0, 5.,
            -25.0, -36.0, 17.0, 4.1,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[25., 10., 30., 20.]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[13., 101., 220., 430.]);
        let p = ldlup(&mut L, &mut d, j, &g);

        // Expected values from Matlab (replace these placeholders with actual values):
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 1.1, 1.2, 1.3,
            12.0, -3.0, 7.0, 0.0,
            3.1, 3.0, 23.0, 5.,
            -25.0, -36.0, 17.0, 4.1,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[25., 10., 30., 20.]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[-0.09167158780541818, -0.08290580940429615, 0.043478260869565216, 0.]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_coverage_0() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.0, 0.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            1.23, -1.0, 1.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.0, 0.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            f64::INFINITY, f64::NEG_INFINITY, -1.0,
        ]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 2);
    }

    #[test]
    fn test_coverage_1() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.2,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.3, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.2,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            2.1111111111111103, -6.666666666666663, 0.1111111111111111,
        ]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_coverage_2() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -7.0, 8.0, 9.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.2,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.3, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -2.0, 3.0,
            4.0, -5.0, 6.0,
            -0.28888888888888886, -0.08943089430894309, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 4.1, 0.1420234869015357,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 2);
    }

    #[test]
    fn test_coverage_3() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 3.0,
            4.0, -1.0, 6.0,
            -8.0, 8.0, 8.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 0.0, 0.2,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 3.0,
            4.0, -1.0, 6.0,
            -8.0, 8.0, 8.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 0.0, 0.2,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            1.0, -0.375, 0.125,
        ]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_coverage_4() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let j = 0;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            1.0, 0.0, 0.0,
        ]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 0);
    }
    #[test]
    fn test_coverage_5() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            -0.3333333333333333, -1.0, 0.3333333333333333,
        ]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_coverage_6() {
        const N: usize = 3;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            1.0, 0.0, 3.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.1, 0.3,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, -1.0, 0.0,
            2.0, -1.0, -1.0,
            0.06666666666666667, 0.1, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.5, 1.0, 0.2833333333333333,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 2);
    }

    #[test]
    fn test_mistake_0() {
        const N: usize = 6;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            23.127986525843045, 18.57643856292412, 1.0, 1.0, 1.0, 1.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            1.7538162952525622, -0.5909985456367551, 24.55657908379219, 0.0, 0.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0034964020620132778, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.07583091132005203, -0.03214451416643742, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            23.127986525843045, 18.57643856292412, 24.40439112304386, 1.0, 1.0, 1.0,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_j_start_row() {
        const N: usize = 2;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0,
        ]);
        let j = 0;
        let g = SVector::<f64, N>::from_row_slice(&[
            1.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(j, 0);
    }

    #[test]
    fn test_j_last_row() {
        const N: usize = 2;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0,
        ]);
        let j = 1;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.5, 2.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0,
            0.5, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 1.75,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }

    #[test]
    fn test_large_matrix_size() {
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.3, 0.6, 1.0, 0.0,
            0.4, 0.7, 0.8, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            0.0, 0.0, 3.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.4, 0.7, 0.0, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let expected_p: Option<SVector<f64, N>> = None;

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
        assert_eq!(g, SVector::<f64, N>::from_row_slice(&[0.0, 0.0, 3.0, 0.0])); // just in case check
    }

    #[test]
    fn test_3() {
        const N: usize = 4;
        let mut L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.3, 0.6, 1.0, 0.0,
            0.4, 0.7, 0.8, 1.0,
        ]);
        let mut d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let j = 2;
        let g = SVector::<f64, N>::from_row_slice(&[
            -100.0, 0.0, 3.0, 0.0,
        ]);
        let p = ldlup(&mut L, &mut d, j, &g);
        let expected_L = SMatrix::<f64, N, N>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.2, 1.0, 0.0, 0.0,
            0.3, 0.6, 1.0, 0.0,
            0.4, 0.7, 0.8, 1.0,
        ]);
        let expected_d = SVector::<f64, N>::from_row_slice(&[
            1.0, 2.0, 3.0, 4.0,
        ]);
        let expected_p = Some(SVector::<f64, N>::from_row_slice(&[
            -102.0, 10.0, -1.0, 0.0,
        ]));

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
        assert_eq!(p, expected_p);
    }
}