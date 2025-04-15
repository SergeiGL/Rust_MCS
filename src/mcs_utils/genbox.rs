use crate::mcs_utils::updtrec::updtrec;
use nalgebra::Matrix2xX;

#[inline]
pub(super) fn genbox<const SMAX: usize>(
    nboxes: &mut usize,
    ipar: &mut Vec<Option<usize>>,
    level: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    isplit: &mut Vec<isize>,
    nogain: &mut Vec<bool>,
    f: &mut Matrix2xX<f64>,
    z: &mut Matrix2xX<f64>,
    ipar_upd: usize,  // par; -1 from Matlab
    level_upd: usize, // s + smth
    ichild_upd: isize,
    f_upd: f64,
    record: &mut [Option<usize>; SMAX],
) {
    // Do [*nboxes] before *nboxes += 1
    // nboxes: -1 from Matlab
    if *nboxes <= ipar.len() - 1 {
        ipar[*nboxes] = Some(ipar_upd + 1); // ipar_upd -1 from Matlab as comes from record, ipar as in Matlab
        level[*nboxes] = level_upd;
        ichild[*nboxes] = ichild_upd;
        f[(0, *nboxes)] = f_upd;
    } else { // perform allocation; by default *2 from current
        ipar.push(Some(ipar_upd + 1)); // ipar_upd -1 from Matlab as comes from record, ipar as in Matlab
        level.push(level_upd);
        ichild.push(ichild_upd);

        let new_cap = ipar.capacity();
        nogain.resize(new_cap, false);
        isplit.resize(new_cap, 0_isize);

        f.resize_horizontally_mut(new_cap, 1.0);
        z.resize_horizontally_mut(new_cap, 1.0);

        f[(0, *nboxes)] = f_upd;
    }

    updtrec(*nboxes, level[*nboxes], f.row(0), record);

    *nboxes += 1; // nboxes: consistent with matlab again
}
