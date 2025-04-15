#[inline]
pub(super) fn lssat(
    small: f64,
    alist: &[f64],
    flist: &[f64],
    alp: &mut f64,
    amin: f64,
    amax: f64,
    s: usize,
    saturated: &mut bool,
) {
    if !(*saturated) {
        return;
    }

    // Find the index of the minimum element
    let (i, _) = flist
        .iter()
        .enumerate()
        .min_by(|&(_, f_i), (_, f_j)| f_i.total_cmp(f_j))
        .unwrap();

    if i != 0 && i != s - 1 {
        // Select points for parabolic interpolation
        let (aa0, aa1, aa2) = (alist[i - 1], alist[i], alist[i + 1]);
        let (ff0, ff1, ff2) = (flist[i - 1], flist[i], flist[i + 1]);

        let f12 = (ff1 - ff0) / (aa1 - aa0);
        let f23 = (ff2 - ff1) / (aa2 - aa1);
        let f123 = (f23 - f12) / (aa2 - aa0);

        if f123 > 0.0 {
            // Parabolic minimizer
            *alp = 0.5 * (aa1 + aa2 - f23 / f123);
            *alp = alp.clamp(amin, amax);

            let alptol = small * (aa2 - aa0);
            *saturated = (alist[i] - *alp).abs() <= alptol;
        } else {
            *saturated = false;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturated_false() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 1.5, 2.1];
        let mut alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = alist.len();
        let mut saturated = false;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (0.75, false));
    }

    #[test]
    fn test_saturation_check_at_lower_boundary() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![1.0, 1.5, 2.0];
        let mut alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (0.75, true));
    }

    #[test]
    fn test_saturation_check_at_upper_boundary() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 2.1, 1.0];
        let mut alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (0.75, true));
    }

    #[test]
    fn test_interpolation_positive_f123() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5, 2.0];
        let flist = vec![2.0, 1.5, 1.8, 2.1];
        let mut alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (1.0625, false));
    }

    #[test]
    fn test_interpolation_negative_f123() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5, 2.0];
        let flist = vec![2.0, 1.5, 1.2, 2.1];
        let mut alp = 0.75;
        let amin = 0.1;
        let amax = 2.0;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (1.375, false));
    }

    #[test]
    fn test_alp_clamped_to_amin_amax() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 1.5, 2.1];
        let mut alp = 0.75;
        let amin = 0.8;
        let amax = 1.2;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (0.9772727272727273, false));
    }

    #[test]
    fn test_saturation_tolerance() {
        let small = 1e-6;
        let alist = vec![0.5, 1.0, 1.5];
        let flist = vec![2.0, 1.5, 2.1];
        let mut alp = 1.0;
        let amin = 0.1;
        let amax = 2.0;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (0.9772727272727273, false));
    }

    #[test]
    fn test_single_element_list() {
        let small = 1e-6;
        let alist = vec![1.0];
        let flist = vec![2.0];
        let mut alp = 1.0;
        let amin = 0.1;
        let amax = 2.0;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (1.0, true));
    }

    #[test]
    fn test_0() {
        // Matlab equivalent test
        // small = 1e-10;
        // alist = [1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        // flist = [1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        // alp = 2.0;
        // amin = 0.1;
        // amax = 0.11;
        // s = 10;
        // saturated = true;
        //
        // lssat;
        //
        // disp(alp);
        // disp(saturated);

        let small = 1e-10;
        let alist = vec![1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let flist = vec![1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        let mut alp = 2.0;
        let amin = 0.1;
        let amax = 0.11;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (0.11, false));
    }

    #[test]
    fn test_1() {
        // Matlab equivalent test
        // small = 1e-10;
        // alist = [1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        // flist = [1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        // alp = 2.0;
        // amin = 100.;
        // amax = 200.;
        // s = 10;
        // saturated = true;
        //
        // lssat;
        //
        // disp(alp);
        // disp(saturated);

        let small = 1e-10;
        let alist = vec![1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let flist = vec![1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        let mut alp = 2.0;
        let amin = 100.;
        let amax = 200.;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (100., false));
    }

    #[test]
    fn test_2() {
        // Matlab equivalent test
        // small = 1e-10;
        // alist = [1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        // flist = [1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        // alp = 2.0;
        // amin = -100.;
        // amax = 200.;
        // s = 10;
        // saturated = true;
        //
        // lssat;
        //
        // disp(alp);
        // disp(saturated);

        let small = 1e-10;
        let alist = vec![1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let flist = vec![1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        let mut alp = 2.0;
        let amin = -100.;
        let amax = 200.;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (5.571428571428571, false));
    }

    #[test]
    fn test_3() {
        // Matlab equivalent test
        // small = 0.4;
        // alist = [1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        // flist = [1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        // alp = 2.0;
        // amin = -100.;
        // amax = 200.;
        // s = 10;
        // saturated = true;
        //
        // lssat;
        //
        // disp(alp);
        // disp(saturated);

        let small = 0.4;
        let alist = vec![1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let flist = vec![1.0, 2., 3., 4., -5., -6., 7., 8., 9., 10.];
        let mut alp = 2.0;
        let amin = -100.;
        let amax = 200.;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (5.571428571428571, true));
    }

    #[test]
    fn test_4() {
        // Matlab equivalent test
        // small = 0.1;
        // alist = [1.0, -2., 3.1, 4.3, -5., 16., 47., -8.1, 9.4, -10.4];
        // flist = [1.3, 2.1, -3.2, 4.1, -5.3, -6.4, 7.1, -8.6, 9.9, 10.1];
        // alp = 1.01;
        // amin = -100.;
        // amax = 200.;
        // s = 10;
        // saturated = true;
        //
        // lssat;
        //
        // disp(alp);
        // disp(saturated);

        let small = 0.1;
        let alist = vec![1.0, -2., 3.1, 4.3, -5., 16., 47., -8.1, 9.4, -10.4];
        let flist = vec![1.3, 2.1, -3.2, 4.1, -5.3, -6.4, 7.1, -8.6, 9.9, 10.1];
        let mut alp = 1.01;
        let amin = -100.;
        let amax = 200.;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (1.01, false));
    }


    #[test]
    fn test_5() {
        // Matlab equivalent test
        // small = 0.1;
        // alist = [1.0, -2., 3.1, 4.3, -5., 16., 47., -8.1, 9.4, -10.4];
        // flist = [1.3, -2.1, -3.2, -14.1, -15.3, -6.4, -7.1, -8.6, 9.9, 10.1];
        // alp = -1.01;
        // amin = -100.;
        // amax = 200.;
        // s = 10;
        // saturated = true;
        //
        // lssat;
        //
        // disp(alp);
        // disp(saturated);

        let small = 0.1;
        let alist = vec![1.0, -2., 3.1, 4.3, -5., 16., 47., -8.1, 9.4, -10.4];
        let flist = vec![1.3, -2.1, -3.2, -14.1, -15.3, -6.4, -7.1, -8.6, 9.9, 10.1];
        let mut alp = -1.01;
        let amin = -100.;
        let amax = 200.;
        let s = alist.len();
        let mut saturated = true;

        lssat(small, &alist, &flist, &mut alp, amin, amax, s, &mut saturated);
        assert_eq!((alp, saturated), (-2.9107087024491953, false));
    }
}
