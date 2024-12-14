pub fn update_flag(flag: &mut bool, stop: &[f64], fbest: f64, cvtr_vtr: usize) {
    if stop[0] > 0.0 && stop[0] < 1.0 {
        *flag = {
            let fglob = stop[1];
            if fbest - fglob <= (stop[0] * fglob.abs()).max(stop[2]) {
                false
            } else {
                true
            }
        };
    } else if stop[0] == 0.0 {
        *flag = if fbest <= stop[cvtr_vtr] { false } else { true };
    }
}

