use nalgebra::SVector;

#[inline]
pub fn clamp_SVector<const N: usize>(svec: &SVector<f64, N>, min: &SVector<f64, N>, max: &SVector<f64, N>) -> SVector<f64, N> {
    svec.sup(min).inf(max)
}
#[inline]
pub fn clamp_SVector_mut<const N: usize>(svec: &mut SVector<f64, N>, min: &SVector<f64, N>, max: &SVector<f64, N>) {
    *svec = svec.sup(min).inf(max);
}

#[inline]
pub fn update_flag(flag: &mut bool, fbest: f64, nsweeps: usize, freach: f64) {
    // if stop[0] > 0.0 && stop[0] < 1.0 {
    //     *flag = {
    //         let fglob = stop[1];
    //         if fbest - fglob <= (stop[0] * fglob.abs()).max(stop[2]) {
    //             false
    //         } else {
    //             true
    //         }
    //     };
    // } else if stop[0] == 0.0 {
    //     *flag = if fbest <= stop[cvtr_vtr] { false } else { true };
    // }
    
    // stop(1) in matlab and stop[0] here is nsweeps
    if nsweeps == 0 {
        *flag = if fbest <= freach { false } else { true };
    }
}

