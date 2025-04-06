use crate::gls::lsguard::lsguard;
use crate::gls::lslocal::lslocal;
use crate::gls::lssort::lssort;
use crate::gls::quartic::quartic;
use nalgebra::{Matrix3, SVector};

#[inline]
pub fn lsquart<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    nloc: usize,
    small: f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: &mut f64,
    amax: &mut f64,
    alp: &mut f64,
    abest: &mut f64,
    fbest: &mut f64,
    fmed: &mut f64,
    up: &mut Vec<bool>,
    down: &mut Vec<bool>,
    monotone: &mut bool,
    minima: &mut Vec<bool>,
    nmin: &mut usize,
    unitlen: &mut f64,
    s: &mut usize,
    saturated: &mut bool,
) {
    debug_assert!(*s - 1 == up.len());
    debug_assert!(*s - 1 == down.len());

    let f12 = if alist[0] == alist[1] { 0.0 } else { (flist[1] - flist[0]) / (alist[1] - alist[0]) };
    let f23 = if alist[1] == alist[2] { 0.0 } else { (flist[2] - flist[1]) / (alist[2] - alist[1]) };
    let f34 = if alist[2] == alist[3] { 0.0 } else { (flist[3] - flist[2]) / (alist[3] - alist[2]) };
    let f45 = if alist[3] == alist[4] { 0.0 } else { (flist[4] - flist[3]) / (alist[4] - alist[3]) };

    let f123 = (f23 - f12) / (alist[2] - alist[0]);
    let f234 = (f34 - f23) / (alist[3] - alist[1]);
    let f345 = (f45 - f34) / (alist[4] - alist[2]);
    let f1234 = (f234 - f123) / (alist[3] - alist[0]);
    let f2345 = (f345 - f234) / (alist[4] - alist[1]);
    let f12345 = (f2345 - f1234) / (alist[4] - alist[0]);

    if f12345 <= 0.0 { // quart false
        // quartic not bounded below
        lslocal(func, nloc, small, x, p, alist, flist, *amin, *amax, alp, abest, fbest, fmed,
                up, down, monotone, minima, nmin, unitlen, s, saturated);
    } else { // quart true
        // Expanding around alist[2]
        let mut c: SVector<f64, 5> = SVector::zeros();
        c[0] = f12345;
        c[1] = f1234 + c[0] * (alist[2] - alist[0]);
        c[2] = f234 + c[1] * (alist[2] - alist[3]);
        c[1] += c[0] * (alist[2] - alist[3]);
        c[3] = f23 + c[2] * (alist[2] - alist[1]);
        c[2] += c[1] * (alist[2] - alist[1]);
        c[1] += c[0] * (alist[2] - alist[1]);
        c[4] = flist[2];

        let cmax = c.max();
        // Built-in normalize func does not work as it computes cmax not as c.max() but as c.max()-c.min()
        c.scale_mut(1. / cmax);

        let hk = 4.0 * c[0];
        let compmat = Matrix3::<f64>::new(
            0.0, 0.0, -c[3],
            hk, 0.0, -2.0 * c[2],
            0.0, hk, -3.0 * c[1],
        );

        // Calculate eigenvalues (complex)
        let mut ev = compmat.complex_eigenvalues();
        ev.scale_mut(1. / hk);

        let n_real_roots = ev.iter().filter(|ev_i| ev_i.im == 0.0).count();

        if n_real_roots == 1 {
            *alp = alist[2] + ev[0].re; // Img part is 0
        } else {
            let mut alp1 = alist[2] + ev[0].re.min(ev[1].re.min(ev[2].re));
            lsguard(&mut alp1, alist, *amax, *amin, small);

            let mut alp2 = alist[2] + ev[0].re.max(ev[1].re.max(ev[2].re));
            lsguard(&mut alp2, alist, *amax, *amin, small);

            let f1 = cmax * quartic(&c, alp1 - alist[2]);
            let f2 = cmax * quartic(&c, alp2 - alist[2]);

            *alp = if alp2 > alist[4] && f2 < *flist.iter().max_by(|a, b| a.total_cmp(b)).unwrap() {
                alp2
            } else if alp1 < alist[0] && f1 < *flist.iter().max_by(|a, b| a.total_cmp(b)).unwrap() {
                alp1
            } else if f2 <= f1 {
                alp2
            } else {
                alp1
            };
        }

        if !alist.iter().any(|&v| v == *alp) { // opposite to if max(alist==alp);
            lsguard(alp, alist, *amax, *amin, small);
            let falp = func(&(x + p.scale(*alp)));
            alist.push(*alp);
            flist.push(falp);
            (*abest, *fbest, *fmed, *up, *down, *monotone, *minima, *nmin, *unitlen, *s) = lssort(alist, flist);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_real_mistake_0() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // nloc = 1;
        // small = 0.1;
        // x = [0.190983, 0.6, 0.7, 0.8, 0.9, 1.0];
        // p = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        // alist = [-1.6, -1.1, -0.6, -0.40767007775845987, 0.0];
        // flist = [-0.00033007085742366247, -0.005222762849281021, -0.019362461793975085, -0.02327501776110989, -0.01523227097945881];
        // amin = -1.6;
        // amax = 0.4;
        // alp = -0.40767007775845987;
        // abest = -0.40767007775845987;
        // fbest = -0.02327501776110989;
        // fmed = -0.01523227097945881;
        // up = [false, false, false, true];
        // down = [true, true, true, false];
        // monotone = false;
        // minima = [false, false, false, true, false];
        // nmin = 1;
        // unitlen = 1.1923299222415402;
        // s = 5;
        // saturated = false;
        //
        // lsquart;
        //
        // disp(alist);
        // disp(flist);
        // disp(amin);
        // disp(amax);
        // disp(alp);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);
        // disp(saturated);

        let nloc = 1;
        let small = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[0.190983, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-1.6, -1.1, -0.6, -0.40767007775845987, 0.0];
        let mut flist = vec![-0.00033007085742366247, -0.005222762849281021, -0.019362461793975085, -0.02327501776110989, -0.01523227097945881];
        let mut amin = -1.6;
        let mut amax = 0.4;
        let mut alp = -0.40767007775845987;
        let mut abest = -0.40767007775845987;
        let mut fbest = -0.02327501776110989;
        let mut fmed = -0.01523227097945881;
        let mut up = vec![false, false, false, true];
        let mut down = vec![true, true, true, false];
        let mut monotone = false;
        let mut minima = vec![false, false, false, true, false];
        let mut nmin = 1;
        let mut unitlen = 1.1923299222415402;
        let mut s = up.len() + 1;
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alist, [-1.6, -1.1, -0.6, -0.40767007775845987, -0.30750741391479774, 0.0]);
        assert_eq!(flist, [-0.00033007085742366247, -0.005222762849281021, -0.019362461793975085, -0.02327501776110989, -0.023740401281451076, -0.01523227097945881]);
        assert_eq!(amin, -1.6);
        assert_eq!(amax, 0.4);
        assert_eq!(alp, -0.30750741391479774);
        assert_eq!(abest, -0.30750741391479774);
        assert_eq!(fbest, -0.023740401281451076);
        assert_eq!(fmed, -0.017297366386716948);
        assert_eq!(up, vec![false, false, false, false, true, ]);
        assert_eq!(down, vec![true, true, true, true, false, ]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, false, false, false, true, false, ]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 1.2924925860852023);
        assert_eq!(s, 6);
        assert_eq!(saturated, false);
    }

    #[test]
    fn test_unbounded_case() {
        // MATLAB equivalent (quartic false branch):
        // ---------------------------------------------
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // nloc = 1;
        // small = 0.1;
        // x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        // p = [1, 0, 0, 0, 0, 0];
        // alist = [-1.0, -0.5, 0.0, 0.5, 1.0];
        // flist = [0, 0, 0, 0, 0];   % Zero differences force f12345 == 0 (<= 0)
        // amin = -1.0;
        // amax = 1.0;
        // alp = 0.0;
        // abest = 0.0;
        // fbest = 0.0;
        // fmed = 0.0;
        // up = [true, false, true, false];
        // down = [false, true, false, true];
        // monotone = false;
        // minima = [false, true, false, true, false];
        // nmin = 1;
        // unitlen = 1.0;
        // s = length(up)+1;
        // saturated = false;
        //
        // lsquart();
        //
        // disp(alist);
        // disp(flist);
        // disp(amin);
        // disp(amax);
        // disp(alp);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);
        // disp(saturated);

        let nloc = 1;
        let small = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut flist = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let mut amin = -1.0;
        let mut amax = 1.0;
        let mut alp = 0.0;
        let mut abest = 0.0;
        let mut fbest = 0.0;
        let mut fmed = 0.0;
        let mut up = vec![true, false, true, false];
        let mut down = vec![false, true, false, true];
        let mut monotone = false;
        let mut minima = vec![false, true, false, true, false];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = up.len() + 1;
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alist, [-1.0, -0.5, 0.0, 0.5, 1.0]);
        assert_eq!(flist, [0., 0., 0., 0., 0.]);
        assert_eq!(amin, -1.);
        assert_eq!(amax, 1.);
        assert_eq!(alp, 0.0);
        assert_eq!(abest, 0.0);
        assert_eq!(fbest, 0.0);
        assert_eq!(fmed, 0.0);
        assert_eq!(up, vec![false, false, false, false]);
        assert_eq!(down, vec![true, true, true, false]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, false, false, false, false, ]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 1.);
        assert_eq!(s, 5);
        assert_eq!(saturated, false);
    }

    #[test]
    fn test_bounded_case() {
        // MATLAB equivalent (quartic true everywhere branch):
        // ---------------------------------------------
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // nloc = 1;
        // small = 0.1;
        // x = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        // p = [0, 1, 0, 0, 0, 0];
        // alist = [-1.0, -0.5, 0.0, 0.5, 1.0];
        // flist = [0.0, -0.5, -1.0, -1.5, -1.0];
        // amin = -1.0;
        // amax = 1.0;
        // alp = 1.4;
        // abest = 21.4;
        // fbest = -1.45;
        // fmed = 3.;
        // up = [false, true, false, true];
        // down = [true, false, true, false];
        // monotone = false;
        // minima = [true, false, true, false, true];
        // nmin = 1;
        // unitlen = 1.0;
        // s = length(up) + 1;
        // saturated = false;
        //
        // lsquart();
        //
        // disp(alist);
        // disp(flist);
        // disp(amin);
        // disp(amax);
        // disp(alp);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);
        // disp(saturated);

        let nloc = 1;
        let small = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 0.4, 0.3, 0.2, 0.1, 0.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut flist = vec![0.0, -0.5, -1.0, -1.5, -1.0];
        let mut amin = -1.0;
        let mut amax = 1.0;
        let mut alp = 1.4;
        let mut abest = 21.4;
        let mut fbest = -1.45;
        let mut fmed = 3.;
        let mut up = vec![false, true, false, true];
        let mut down = vec![true, false, true, false];
        let mut monotone = false;
        let mut minima = vec![true, false, true, false, true];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = up.len() + 1;
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alist, [-1.0, -0.5, 0.0, 0.5, 0.6140142478696381, 1.0]);
        assert_eq!(flist, [0., -0.5, -1., -1.5, -0.5698604198029456, -1.00]);
        assert_eq!(amin, -1.);
        assert_eq!(amax, 1.);
        assert_eq!(alp, 0.6140142478696381);
        assert_eq!(abest, 0.5);
        assert_eq!(fbest, -1.5);
        assert_eq!(fmed, -0.7849302099014728);
        assert_eq!(up, vec![false, false, false, true, false]);
        assert_eq!(down, vec![true, true, true, false, true]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, false, false, true, false, true]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 0.5);
        assert_eq!(s, 6);
        assert_eq!(saturated, false);
    }

    #[test]
    fn test_edge_case_equal_alist() {
        // MATLAB equivalent (testing divisions when consecutive alist values are equal):
        // ---------------------------------------------
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // nloc = 1;
        // small = 0.1;
        // x = [1, 2, 3, 4, 5, 6];
        // p = [0, 0, 1, 0, 0, 0];
        // alist = [0.0, 0.0, 1.0, 2.0, 3.0];   % First two elements equal
        // flist = [1.0, 1.0, 2.0, 3.0, 4.0];
        // amin = 0.0;
        // amax = 3.0;
        // alp = 0.0;
        // abest = 0.1;
        // fbest = 0.2;
        // fmed = 0.3;
        // up = [true, false, true, false];
        // down = [false, true, false, true];
        // monotone = false;
        // minima = [false, false, true, false, true];
        // nmin = 1;
        // unitlen = 1.0;
        // s = length(up)+1;  saturated = false;
        //
        // lsquart();
        //
        // disp(alist);
        // disp(flist);
        // disp(amin);
        // disp(amax);
        // disp(alp);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);
        // disp(saturated);

        let nloc = 1;
        let small = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![0.0, 0.0, 1.0, 2.0, 3.0];
        let mut flist = vec![1.0, 1.0, 2.0, 3.0, 4.0];
        let mut amin = 0.0;
        let mut amax = 3.0;
        let mut alp = 0.0;
        let mut abest = 0.1;
        let mut fbest = 0.2;
        let mut fmed = 0.3;
        let mut up = vec![true, false, true, false];
        let mut down = vec![false, true, false, true];
        let mut monotone = false;
        let mut minima = vec![false, false, true, false, true];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = up.len() + 1;
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alist, vec![0., 0., 0.0, 1.0, 2.0, 3., ]);
        assert_eq!(flist, vec![1., 1., -3.391970967769076e-191, 2.0, 3., 4.0]);
        assert_eq!(amin, 0.);
        assert_eq!(amax, 3.);
        assert_eq!(alp, 0.0);
        assert_eq!(abest, 0.0);
        assert_eq!(fbest, -3.391970967769076e-191);
        assert_eq!(fmed, 1.5);
        assert_eq!(up, vec![false, false, true, true, true, ]);
        assert_eq!(down, vec![true, true, false, false, false, ]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, false, true, false, false, false, ]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 3.);
        assert_eq!(s, 6);
        assert_eq!(saturated, false);
    }

    #[test]
    fn test_real_mistake_negative_direction_() {
        // MATLAB equivalent:
        // --------------------
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // nloc = 1;
        // small = 0.1;
        // x = [1.0, 0.5, -0.5, -1.0, -1.5, -2.0];
        // p = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        // alist = [0.1, 0.5, -1.0, 1.5, -2.0];
        // flist = [0.0, -0.2, -0.4, -0.3, -0.1];
        // amin = 0.0;
        // amax = 2.0;
        // alp = 1.0;
        // abest = 1.0;
        // fbest = -0.4;
        // fmed = -0.3;
        // up = [true, false, true, false];
        // down = [false, true, false, true];
        // monotone = true;
        // minima = [false, true, false, true, false];
        // nmin = 1;
        // unitlen = 1.0;
        // s = length(up) + 1;
        // saturated = false;
        //
        // lsquart();
        //
        // disp(alist);
        // disp(flist);
        // disp(amin);
        // disp(amax);
        // disp(alp);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);
        // disp(saturated);

        let nloc = 1;
        let small = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 0.5, -0.5, -1.0, -1.5, -2.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]);
        let mut alist = vec![0.1, 0.5, -1.0, 1.5, -2.0];
        let mut flist = vec![0.0, -0.2, -0.4, -0.3, -0.1];
        let mut amin = 0.0;
        let mut amax = 2.0;
        let mut alp = 1.0;
        let mut abest = 1.0;
        let mut fbest = -0.4;
        let mut fmed = -0.3;
        let mut up = vec![true, false, true, false];
        let mut down = vec![false, true, false, true];
        let mut monotone = true;
        let mut minima = vec![false, true, false, true, false];
        let mut nmin = 1;
        let mut unitlen = 1.0;
        let mut s = up.len() + 1;
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alist, vec![-2., -1., 0.1, 0.5, 1.1478848440118115, 1.5]);
        assert_eq!(flist, vec![-0.1, -0.4, 0., -0.2, -1.6484456831893024e-9, -0.3]);
        assert_eq!(amin, 0.);
        assert_eq!(amax, 2.);
        assert_eq!(alp, 1.1478848440118115);
        assert_eq!(abest, -1.0);
        assert_eq!(fbest, -0.4);
        assert_eq!(fmed, -0.15000000000000002);
        assert_eq!(up, vec![false, true, false, true, false, ]);
        assert_eq!(down, vec![true, false, true, false, true, ]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![false, true, false, true, false, true, ]);
        assert_eq!(nmin, 3);
        assert_eq!(unitlen, 1.5);
        assert_eq!(s, 6);
        assert_eq!(saturated, false);
    }

    #[test]
    fn test_non_monotone_input() {
        // MATLAB equivalent:
        // --------------------
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // nloc = 1;
        // small = 0.1;
        // x = [2, 4, 6, 8, 10, 12];
        // p = [0, -1, 1, 0, 0, 0];
        // alist = [-1.0, 0.5, 1.0, -2.5, 3.0];
        // flist = [1.0, 0.8, 0.6, 0.4, 0.2];
        // amin = 1.0;
        // amax = 3.0;
        // alp = 2.0;
        // abest = 2.0;
        // fbest = 0.6;
        // fmed = 0.8;
        // up = [false, true, false, true];
        // down = [true, false, true, false];
        // monotone = false;
        // minima = [true, false, true, false, true];
        // nmin = 2;
        // unitlen = 0.5;
        // s = length(up) + 1;
        // saturated = false;
        //
        // lsquart();
        //
        // disp(alist);
        // disp(flist);
        // disp(amin);
        // disp(amax);
        // disp(alp);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);
        // disp(saturated);

        let nloc = 1;
        let small = 0.1;
        let x = SVector::<f64, 6>::from_row_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-1.0, 0.5, 1.0, -2.5, 3.0];
        let mut flist = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let mut amin = 1.0;
        let mut amax = 3.0;
        let mut alp = 2.0;
        let mut abest = 2.0;
        let mut fbest = 0.6;
        let mut fmed = 0.8;
        let mut up = vec![false, true, false, true];
        let mut down = vec![true, false, true, false];
        let mut monotone = false;
        let mut minima = vec![true, false, true, false, true];
        let mut nmin = 2;
        let mut unitlen = 0.5;
        let mut s = up.len() + 1;
        let mut saturated = false;

        lsquart(hm6, nloc, small, &x, &p, &mut alist, &mut flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        assert_eq!(alist, vec![-2.5, -1., 0.5, 1., 2.5480214566643298, 3.]);
        assert_eq!(flist, vec![0.4, 1., 0.8, 0.6, 0., 0.2]);
        assert_eq!(amin, 1.);
        assert_eq!(amax, 3.);
        assert_eq!(alp, 2.5480214566643298);
        assert_eq!(abest, 2.5480214566643298);
        assert_eq!(fbest, 0.0);
        assert_eq!(fmed, 0.5);
        assert_eq!(up, vec![true, false, false, false, true, ]);
        assert_eq!(down, vec![false, true, true, true, false, ]);
        assert_eq!(monotone, false);
        assert_eq!(minima, vec![true, false, false, false, true, false, ]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 5.04802145666433);
        assert_eq!(s, 6);
        assert_eq!(saturated, false);
    }
}

