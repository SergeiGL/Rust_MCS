use nalgebra::SVector;

pub(crate) fn add_basket<const N: usize>(nbasket: &mut usize, xmin: &mut Vec<SVector<f64, N>>, fmi: &mut Vec<f64>, x_val: &SVector<f64, N>, f_val: f64) {
    // nbasket: as in Matlab => first use as [nbasket], then do +=1
    if xmin.len() == *nbasket {
        xmin.push(*x_val);
        fmi.push(f_val);
    } else {
        xmin[*nbasket] = *x_val;
        fmi[*nbasket] = f_val;
    }
    *nbasket += 1;
}
