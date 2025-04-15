// Calculate up and down vectors
#[inline]
pub(super) fn clear_and_calc_up_down(up: &mut Vec<bool>, down: &mut Vec<bool>, flist: &Vec<f64>) {
    up.clear();
    down.clear();
    for i in 0..flist.len() - 1 {
        let up_bool = flist[i + 1] > flist[i];
        up.push(up_bool);
        down.push(!up_bool);
    };
}