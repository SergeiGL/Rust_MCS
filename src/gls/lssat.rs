pub fn lssat(
    small: f64,
    alist: &Vec<f64>,
    flist: &Vec<f64>,
    mut alp: f64,
    amin: f64,
    amax: f64,
    s: usize,
    mut saturated: bool,
) -> (
    f64, // alp
    bool // saturated
) {
    let cont = saturated;

    if cont {
        let i = flist
            .iter()
            .position(|&val| val == *flist.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap())
            .unwrap();
        if !(i == 0 || i == s - 1) {
            // Select 3 points for parabolic interpolation
            let aa = &alist[(i - 1)..=(i + 1)];
            let ff = &flist[(i - 1)..=(i + 1)];

            let f12 = (ff[1] - ff[0]) / (aa[1] - aa[0]);
            let f23 = (ff[2] - ff[1]) / (aa[2] - aa[1]);
            let f123 = (f23 - f12) / (aa[2] - aa[0]);

            if f123 > 0.0 {
                alp = 0.5 * (aa[1] + aa[2] - f23 / f123);
                alp = alp.clamp(amin, amax);

                let alptol = small * (aa[2] - aa[0]);
                saturated = (alist[i] - alp).abs() <= alptol;
            } else {
                saturated = false;
            }
        }
    }

    (alp, saturated)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturated_false() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 1.5, 2.1];
        let alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = 3;
        let saturated = false;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (0.75, false));
    }

    #[test]
    fn test_saturation_check_at_lower_boundary() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![1.0, 1.5, 2.0];
        let alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = 3;
        let saturated = true;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (0.75, true));
    }

    #[test]
    fn test_saturation_check_at_upper_boundary() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 2.1, 1.0];
        let alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = 3;
        let saturated = true;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (0.75, true));
    }

    #[test]
    fn test_interpolation_positive_f123() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5, 2.0];
        let flist = vec![2.0, 1.5, 1.8, 2.1];
        let alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = 4;
        let saturated = true;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (1.0625, false));
    }

    #[test]
    fn test_interpolation_negative_f123() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5, 2.0];
        let flist = vec![2.0, 1.5, 1.2, 2.1];
        let alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = 4;
        let saturated = true;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (1.375, false));
    }

    #[test]
    fn test_alp_clamped_to_amin_amax() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 1.5, 2.1];
        let alp = 0.75;
        let amin = 0.8;
        let amax = 1.2;
        let s = 3;
        let saturated = true;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (0.9772727272727273, false));
    }

    #[test]
    fn test_saturation_tolerance() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 1.5, 2.1];
        let alp = 1.0;
        let amin = 0.1;
        let amax = 2.0;
        let s = 3;
        let saturated = true;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (0.9772727272727273, false));
    }

    #[test]
    fn test_single_element_list() {
        let small = 1e-6;
        let alist = vec![1.0];
        let flist = vec![2.0];
        let alp = 1.0;
        let amin = 0.1;
        let amax = 2.0;
        let s = 1;
        let saturated = true;

        let result = lssat(small, &alist, &flist, alp, amin, amax, s, saturated);
        assert_eq!(result, (1.0, true));
    }
}
