use crate::mcs_utils::updtrec::updtrec;

pub(super) fn genbox<const SMAX: usize>(
    nboxes: &mut usize,
    ipar: &mut Vec<usize>,
    level: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    isplit: &mut Vec<isize>,
    nogain: &mut Vec<bool>,
    f: &mut [Vec<f64>; 2],
    z: &mut [Vec<f64>; 2],
    ipar_upd: usize,  // par; -1 from Matlab
    level_upd: usize, // s + smth
    ichild_upd: isize,
    f_upd: f64,
    record: &mut [usize; SMAX],
) {
    // Do [*nboxes] before *nboxes += 1
    // nboxes: -1 from Matlab
    if *nboxes >= level.capacity() {
        let new_capacity = 2 * level.capacity();

        level.resize(new_capacity, 0_usize);
        ipar.resize(new_capacity, 0_usize);
        isplit.resize(new_capacity, 0isize);
        ichild.resize(new_capacity, 0_isize);
        nogain.resize(new_capacity, false);

        f[0].resize(new_capacity, 0_f64);
        f[1].resize(new_capacity, 0_f64);

        z[0].resize(new_capacity, 0_f64);
        z[1].resize(new_capacity, 0_f64);
    }

    ipar[*nboxes] = ipar_upd + 1; // ipar_upd -1 from Matlab as comes from record, ipar as in Matlab
    level[*nboxes] = level_upd;
    ichild[*nboxes] = ichild_upd;
    f[0][*nboxes] = f_upd;

    updtrec(*nboxes, level[*nboxes], &f[0], record);

    *nboxes += 1; // nboxes: consistent with matlab again
}
