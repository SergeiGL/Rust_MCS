use nalgebra::SVector;

#[inline]
pub(crate) fn clamp_SVector_mut<const N: usize>(svec: &mut SVector<f64, N>, min: &SVector<f64, N>, max: &SVector<f64, N>) {
    *svec = svec.sup(min).inf(max);
}

#[inline]
pub(super) fn clamp_SVector<const N: usize>(svec: &SVector<f64, N>, min: &SVector<f64, N>, max: &SVector<f64, N>) -> SVector<f64, N> {
    svec.sup(min).inf(max)
}

// used in basket and basket1
#[inline]
pub(super) fn update_fbest_xbest_nsweepbest<const N: usize>(
    fbest: &mut f64, xbest: &mut SVector<f64, N>, nsweepbest: &mut usize,
    fbest_new: f64, xbest_new: &SVector<f64, N>, nsweepbest_new: usize,
) {
    *fbest = fbest_new;
    *xbest = *xbest_new;
    *nsweepbest = nsweepbest_new;
}

// used in basket and basket1
pub(super) fn get_sorted_indices<const N: usize>(nbasket: usize, x: &SVector<f64, N>, xmin: &[SVector<f64, N>]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..nbasket).collect();
    debug_assert!(nbasket <= xmin.len());
    indices.sort_unstable_by(|&i, &j|
        // Norm norm_squared() is used for performance as just norm() is self.norm_squared().simd_sqrt()
        (x - xmin[i]).norm_squared().total_cmp(&(x - xmin[j]).norm_squared())
    );
    indices
}

