use nalgebra::SVector;

pub fn add_basket<const N: usize>(nbasket_option: &mut Option<usize>, xmin: &mut Vec<SVector<f64, N>>, fmi: &mut Vec<f64>, x_val: &SVector<f64, N>, f_val: f64) {
    let nbasket = match nbasket_option {
        Some(val) => {
            *val += 1;
            *val
        }
        None => {
            *nbasket_option = Some(0);
            0
        }
    };
    if xmin.len() == nbasket {
        xmin.push(*x_val);
        fmi.push(f_val);
    } else {
        xmin[nbasket] = *x_val;
        fmi[nbasket] = f_val;
    }
}
