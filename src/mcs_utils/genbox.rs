use crate::mcs_utils::updtrec::updtrec;
use nalgebra::Matrix2xX;
use std::cmp::Ordering;

#[inline]
pub fn genbox<const SMAX: usize>(
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
    match (*nboxes + 1).cmp(&level.capacity()) {
        Ordering::Less => {}
        Ordering::Equal | Ordering::Greater => {
            let new_capacity = level.capacity() * 2;

            level.resize(new_capacity, 0_usize);
            ipar.resize(new_capacity, Some(0));
            isplit.resize(new_capacity, 0isize);
            ichild.resize(new_capacity, 0_isize);
            nogain.resize(new_capacity, false);

            f.resize_horizontally_mut(new_capacity, 1.0);
            z.resize_horizontally_mut(new_capacity, 1.0);
        }
    }
    
    // Do [] before incrementing *nboxes += 1; so at this point rust nboxes = matlab nboxes -1 ;
    ipar[*nboxes] = Some(ipar_upd + 1); // ipar_upd -1 from Matlab as comes from record, ipar as in Matlab
    level[*nboxes] = level_upd;
    ichild[*nboxes] = ichild_upd;
    f[(0, *nboxes)] = f_upd;

    updtrec(*nboxes, level[*nboxes], f.row(0), record);

    *nboxes += 1;
}
