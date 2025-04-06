mod lsrange;
mod lsinit;
mod lssort;
mod lssplit;
mod lsnew;
mod lsguard;
mod lspar;
mod quartic;
mod lslocal;
mod lsquart;
mod lsdescent;
mod lsconvex;
mod lssat;
mod lssep;
mod helpers;

use lsconvex::lsconvex;
use lsdescent::lsdescent;
use lsinit::lsinit;
use lslocal::lslocal;
use lsnew::lsnew;
use lspar::lspar;
use lsquart::lsquart;
use lsrange::lsrange;
use lssat::lssat;
use lssep::lssep;
use lssort::lssort;
use nalgebra::SVector;

pub fn gls<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    nloc: usize,
    small: f64,
    smax: usize,
) ->
    usize // nf
{
    // Golden section fraction is (3 - sqrt(5)) / 2
    let short = 0.381966;

    // Save information for nf computation and extrapolation decision
    let sinit = alist.len();  // Initial list size

    // Get 5 starting points (needed for lslocal)
    let bend = false;

    // Find range of useful alp
    let (mut amin, mut amax, scale) = lsrange(x, p, u, v, bend);

    // Call `lsinit` to get initial alist, flist, etc.
    let mut alp = lsinit(func, x, p, alist, flist, amin, amax, scale);

    // Sort alist and flist and get relevant values
    let (mut abest, mut fbest, mut fmed,
        mut up, mut down, mut monotone,
        mut minima, mut nmin, mut unitlen, mut s) = lssort(alist, flist);


    // The main search loop
    while s < smax.min(5) {
        if nloc == 1 {
            // Parabolic interpolation STEP
            (fbest, up, down, monotone, minima, nmin) = lspar(func, nloc, small, sinit, short, x, p, alist, flist, amin, amax,
                                                              &mut alp, &mut abest, &mut fmed, &mut unitlen, &mut s);

            // If no true parabolic STEP has been done and it's monotonic
            if s > 3 && monotone && (abest == amin || abest == amax) {
                return s - sinit;
            }
        } else {
            // Extrapolation or split
            alp = lsnew(func, nloc, small, sinit, short, x, p, s, alist, flist, amin, amax, abest, fmed, unitlen);

            (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(alist, flist);
        }
    }

    let mut saturated = false;
    if nmin == 1 {
        if monotone && (abest == amin || abest == amax) {
            return s - sinit;
        }
        if s == 5 {
            lsquart(func, nloc, small, x, p, alist, flist, &mut amin, &mut amax, &mut alp, &mut abest, &mut fbest,
                    &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);
        }

        // Check the descent condition
        lsdescent(func, x, p, alist, flist, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        // Check convexity condition
        if lsconvex(&alist, &flist, nmin, s) {
            return s - sinit;
        }
    }

    let mut sold = 0;
    'inner: loop {
        lsdescent(func, x, p, alist, flist, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up,
                  &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        // Check saturation
        lssat(small, alist, flist, &mut alp, amin, amax, s, &mut saturated);

        if saturated || s == sold || s >= smax {
            break 'inner;
        }

        sold = s;
        let nminold = nmin;

        if !saturated && nloc > 1 {
            lssep(func, nloc, small, sinit, short, x, p, alist, flist, &mut amin,
                  &mut amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down, &mut monotone,
                  &mut minima, &mut nmin, &mut unitlen, &mut s);
        }

        lslocal(func, nloc, small, x, p, alist, flist, amin, amax, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up, &mut down,
                &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s, &mut saturated);

        if nmin > nminold { saturated = false }
    }

    s - sinit
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;
    use approx::assert_relative_eq;

    static TOLERANCE: f64 = 1e-13;

    #[test]
    fn test_random_0() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.09, 0.19, 0.186, 0.59, 0.023, 0.665];
        // p = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // alist = [1.];
        // flist = [2.];
        // u = [-1.0, 0.0, -3.0, 0.0, 0.0, 0.0, ];
        // v = [1.79, 1.66, 1.153, 1.62, 1.08, 1.0];
        // nloc = 10;
        // small = 1e-7;
        // smax = 10;
        //
        // [alist_out, flist_out, nf_out] = gls(func, data, u, v, x, p, alist, flist, nloc, small, smax, prt);
        //
        // alist_out = sprintf('%.16g,', alist_out);
        // flist_out = sprintf('%.16g,', flist_out);
        //
        // disp(alist_out);
        // disp(flist_out);
        // disp(nf_out);

        let x = SVector::<f64, 6>::from_row_slice(&[0.09, 0.19, 0.186, 0.59, 0.023, 0.665]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![1.];
        let mut flist = vec![2.];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.0, -3.0, 0.0, 0.0, 0.0, ]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.79, 1.66, 1.153, 1.62, 1.08, 1.0]);
        let nloc = 10;
        let small = 1e-7;
        let smax = 10;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-1.090000000000000, -0.416342940000000, 0., 0.049580949986973, 0.148742849960918, 0.333333333333333, 1.000000000000000, 1.175000000000000, 1.261764405223204, 1.350000000000000, 1.700000000000000];
        let expected_flist = vec![-0.003176069409058, -0.116697337332934, -0.288227052223000, -0.298542473636854, -0.305550807893646, -0.271229158737147, 2.000000000000000, -0.012411764297438, -0.007146649331859, -0.003917088745748, -0.000319036026492];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 10);
    }

    #[test]
    fn test_random_1() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.19, 0.39, 0.486, 0.09, 0.4023, 0.0665];
        // p = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // alist = [1.];
        // flist = [2.];
        // u = [-1.0, 0.0, -3.0, 0.0, 0.0, 0.0, ];
        // v = [1.79, 1.66, 1.153, 1.62, 1.08, 1.0];
        // nloc = 11;
        // small = 1e-7;
        // smax = 20;
        //
        // [alist_out, flist_out, nf_out] = gls(func, data, u, v, x, p, alist, flist, nloc, small, smax, prt);
        //
        // alist_out = sprintf('%.16g,', alist_out);
        // flist_out = sprintf('%.16g,', flist_out);
        //
        // disp(alist_out);
        // disp(" ");
        // disp(flist_out);
        // disp(nf_out);

        let x = SVector::<f64, 6>::from_row_slice(&[0.19, 0.39, 0.486, 0.09, 0.4023, 0.0665]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![1.];
        let mut flist = vec![2.];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.0, -3.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.79, 1.66, 1.153, 1.62, 1.08, 1.0]);
        let nloc = 11;
        let small = 1e-7;
        let smax = 20;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-1.19, -0.45453953999999996, 0., 0.03570504644416015, 0.06581721440990665, 0.0665995552812526, 0.06661419768084426, 0.1071151393324805, 0.3333333333333333, 1., 1.00234375, 1.00351554775829, 1.0046875, 1.009375, 1.014061324756644, 1.01875, 1.0375, 1.056234722646366, 1.075, 1.15, 1.224903210970236, 1.3, 1.6, ];
        let expected_flist = vec![-0.000795900205426112, -0.04998401289642895, -0.1584547222565412, -0.160850892243273, -0.1615337527074966, -0.1615342022657176, -0.1615342021050759, -0.1602867240702137, -0.1075717931849698, 2., -0.004928499474365494, -0.004895371121026823, -0.004862421944517106, -0.00473245483205075, -0.004605398117693165, -0.004481111483922252, -0.004011515899356581, -0.003583993105764202, -0.003194756685133813, -0.001975930499767486, -0.001182480553159051, -0.000683301031252829, -5.478276420538117e-05];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 22);
    }


    #[test]
    fn test_coverage_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, -0.6, 0.7, 0.8, -0.9, 1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut alist = vec![0.0];
        let mut flist = vec![0.02];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[2.; 6]);
        let nloc = 20;
        let small = 0.01;
        let smax = 6;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.09999999999999998, 0.0, 0.1, 0.14792313333333337, 0.4437694000000001, 1.0];
        let expected_flist = vec![-2.528201273109948e-05, 0.02, -8.21831123481634e-06, -4.964535488273189e-06, -4.2871807159348816e-05, -2.01103722474712e-12];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, 5);
    }

    #[test]
    fn test_coverage_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, -0.6, 0.7, 0.8, -0.9, 1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut alist = vec![0.0];
        let mut flist = vec![0.2];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[2.; 6]);
        let nloc = 1;
        let small = 0.0001;
        let smax = 5;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![0.0, 0.1, 0.16099019295255587, 0.48297057885766764, 0.5499977172405514, 0.8607338767331355, 1.0];
        let expected_flist = vec![0.2, -8.21831123481634e-06, -4.732412711411107e-06, -3.534274081231595e-05, -1.790252032974843e-05, -2.387100054905096e-09, -2.01103722474712e-12];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 6);
    }

    #[test]
    fn test_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, -6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![];
        let mut flist = vec![];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[11.0, 11.0, 11.0, 11.0, 11.0, 10.0]);
        let nloc = 10;
        let small = 1e-20;
        let smax = 60;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-1.0, -0.3333333333333333, 0., 0.045166204243341536, 0.13549861273002461, 0.14654729341594333, 0.14655256966095567, 0.14655260551563967, 0.1465526389548695, 0.1465526392003361, 0.14655263944580268, 0.14655263964772686, 0.14655263982429045, 0.14655263984965106, 0.14655263996655088, 0.1465526400834507, 0.14655264060752235, 0.14655264086955816, 0.14655264094739887, 0.146552641131594, 0.14655264146484273, 0.1465526421797373, 0.14655264322705297, 0.1465526433447804, 0.14655264375131594, 0.146552643927302, 0.14655264450982355, 0.14655264502581478, 0.14655264543947694, 0.14655264554180603, 0.14655264590767106, 0.1465526462735361, 0.14655264633041934, 0.1465526463873026, 0.1465526465512236, 0.14655264681005087, 0.14655264723279915, 0.14655264781034133, 0.14655264813145358, 0.1465526483878835, 0.14655264853442995, 0.14655264868097642, 0.14655264924594935, 0.14655264981092225, 0.14655265033243364, 0.14655265052244165, 0.14655265123396102, 0.14655265237692816, 0.14655265351989533, 0.1465526549920037, 0.14655265685621863, 0.14655265816673121, 0.14655265993962382, 0.14655266409008527, 0.14655266446625598, 0.14655266934747335, 0.14655267260036425, 0.14668779856805622, 0.14949228142785437, 0.3333333333333333, 1.0, 1.5572808507673672, 2.4589802515600003, 3.3606796523526334, 3.917960539822108, 4.819660000000001, 5.413795061176836, 6.375125829882334, 7.336456598587832, 7.9305916988939655, 8.297787391476566, 8.891922530912002, 9.697136449469667, 10.194786028411537, 11.0];
        let expected_flist = vec![-1.9659184167967895e-158, -8.352827819883792e-157, -1.7222819428544745e-156, -1.7904383816084758e-156, -1.8545816882549547e-156, -1.8553670996351082e-156, -1.8553670998210436e-156, -1.8553670998211434e-156, -1.8553670998211434e-156, -1.8553670998211336e-156, -1.8553670998211238e-156, -1.8553670998211341e-156, -1.855367099821195e-156, -1.8553670998211515e-156, -1.8553670998211403e-156, -1.8553670998211292e-156, -1.8553670998211051e-156, -1.855367099821195e-156, -1.8553670998211848e-156, -1.8553670998211794e-156, -1.8553670998211437e-156, -1.8553670998211318e-156, -1.8553670998211678e-156, -1.8553670998211711e-156, -1.855367099821144e-156, -1.8553670998211905e-156, -1.855367099821112e-156, -1.8553670998211473e-156, -1.8553670998211447e-156, -1.8553670998211753e-156, -1.8553670998211207e-156, -1.8553670998211647e-156, -1.8553670998211792e-156, -1.855367099821201e-156, -1.8553670998211362e-156, -1.855367099821161e-156, -1.8553670998211284e-156, -1.8553670998211595e-156, -1.8553670998212017e-156, -1.8553670998211908e-156, -1.8553670998211098e-156, -1.8553670998211346e-156, -1.8553670998211385e-156, -1.8553670998211491e-156, -1.855367099821173e-156, -1.8553670998211704e-156, -1.8553670998211005e-156, -1.8553670998211497e-156, -1.8553670998211918e-156, -1.8553670998211328e-156, -1.8553670998211802e-156, -1.8553670998211243e-156, -1.8553670998211396e-156, -1.855367099821187e-156, -1.8553670998211168e-156, -1.8553670998211318e-156, -1.8553670998211274e-156, -1.8553669824357825e-156, -1.8553115465308355e-156, -1.6441751281725677e-156, -1.4995743515116342e-157, -2.0023677564735018e-159, -3.145116973373622e-164, -7.210983147509425e-171, -6.335205234837438e-176, -8.561087544195914e-186, -1.876923849298989e-193, -8.563403909244088e-208, -1.5265513833668475e-224, -4.248459185835921e-236, -1.0632578750176436e-243, -9.612734950371797e-257, -6.869450759982498e-276, -1.4432434066035165e-288, -1.9043433007926e-310];
        let expected_nf = 75;

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[-7.0, -7.0, -9.0, -8.0, -9.0, -10.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![];
        let mut flist = vec![];
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0, -11.0, -9.0, -8.0, -9.0, -12.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[-5.0, -5.0, -6.0, -7.0, -8.0, -9.0]);
        let nloc = 10;
        let small = 1e-10;
        let smax = 100;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![0., 3.5362427088620996e-46, 1.0608728126586298e-45, 3.182618437975889e-45, 9.547855313927668e-45, 2.8643565941783e-44, 8.593069782534901e-44, 2.5779209347604703e-43, 7.733762804281411e-43, 2.3201288412844232e-42, 6.96038652385327e-42, 2.0881159571559808e-41, 6.264347871467942e-41, 1.8793043614403827e-40, 5.637913084321148e-40, 1.6913739252963442e-39, 5.074121775889032e-39, 1.5222365327667096e-38, 4.566709598300129e-38, 1.3700128794900387e-37, 4.110038638470116e-37, 1.2330115915410348e-36, 3.699034774623104e-36, 1.1097104323869313e-35, 3.329131297160794e-35, 9.987393891482382e-35, 2.996218167444715e-34, 8.988654502334145e-34, 2.6965963507002434e-33, 8.08978905210073e-33, 2.4269367156302191e-32, 7.280810146890657e-32, 2.184243044067197e-31, 6.552729132201592e-31, 1.9658187396604775e-30, 5.897456218981432e-30, 1.7692368656944296e-29, 5.307710597083288e-29, 1.5923131791249865e-28, 4.776939537374959e-28, 1.4330818612124878e-27, 4.2992455836374636e-27, 1.2897736750912391e-26, 3.8693210252737176e-26, 1.1607963075821153e-25, 3.482388922746346e-25, 1.0447166768239038e-24, 3.1341500304717114e-24, 9.402450091415134e-24, 2.82073502742454e-23, 8.46220508227362e-23, 2.538661524682086e-22, 7.615984574046259e-22, 2.2847953722138777e-21, 6.854386116641633e-21, 2.05631583499249e-20, 6.16894750497747e-20, 1.8506842514932412e-19, 5.5520527544797235e-19, 1.6656158263439171e-18, 4.9968474790317514e-18, 1.4990542437095254e-17, 4.4971627311285764e-17, 1.349148819338573e-16, 4.0474464580157187e-16, 1.2142339374047156e-15, 3.642701812214147e-15, 1.092810543664244e-14, 3.278431630992732e-14, 9.835294892978195e-14, 2.9505884678934586e-13, 8.851765403680376e-13, 2.655529621104113e-12, 7.966588863312339e-12, 2.389976658993702e-11, 7.169929976981106e-11, 2.1509789930943317e-10, 6.452936979282995e-10, 1.9358810937848983e-09, 5.807643281354695e-09, 1.7422929844064084e-08, 5.2268789532192255e-08, 1.5680636859657676e-07, 4.7041910578973025e-07, 1.4112573173691908e-06, 4.233771952107572e-06, 1.2701315856322716e-05, 3.8103947568968146e-05, 0.00011431184270690443, 0.0003429355281207133, 0.0010288065843621398, 0.0030864197530864196, 0.009259259259259259, 0.027777777777777776, 0.08333333333333333, 0.25, 0.75, 1.5, 2.25, 3.0];
        let expected_flist = vec![0.0; 100];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, 100);
    }
    #[test]
    fn test_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let mut alist = vec![];
        let mut flist = vec![];
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0, -10.0, -10.0, -10.0, -10.0, -10.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[12.0, 12.0, 12.0, 12.0, 12.0, 12.0]);
        let nloc = 20;
        let small = 1e-10;
        let smax = 15;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-16.0, -13.665631290024416, -12.222912153739788, -9.888543597504, -7.416407389963206, -6.18033938142108, -6.111456, -5.940311294392515, -5.923834345064304, -5.629815484265654, -4.944271798752, -3.055728076869894, 0.0, 2.291796, 3.708203849064, 6.0, ];
        let expected_flist = vec![-0.0, -0.0, -1.1809849115665161e-250, -5.039013617993901e-153, -3.8533839024796616e-72, -1.441391932459853e-59, -2.0554442272196492e-59, -2.792968315142224e-59, -2.754797772928035e-59, -6.003203768759197e-60, -1.4207719174686801e-65, -9.92046588980153e-109, -3.391970967769076e-191, -1.2743440745677273e-295, -0.0, -0.0, ];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, 16);
    }

    #[test]
    fn test_3() {
        let x = SVector::<f64, 6>::from_row_slice(&[-100.0, -200.0, -300.0, -400.0, -500.0, 500.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut alist = vec![];
        let mut flist = vec![];
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0, -11.0, -12.0, -13.0, -10.0, -10.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[12.0, 15.0, 12.0, 12.0, 12.0, 12.0]);
        let nloc = 20;
        let small = 1e-10;
        let smax = 10;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![490.0, 495.0, 500.0, 506.0, 512.0];
        let expected_flist = vec![-0.0, -0.0, -0.0, -0.0, -0.0];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, 5);
    }
    #[test]
    fn test_4() {
        let x = SVector::<f64, 6>::from_row_slice(&[-1.0, -2.0, -3.0, -4.0, 0.0, 0.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let u = SVector::<f64, 6>::from_row_slice(&[-10.0, -11.0, -12.0, -13.0, -10.0, -10.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.1, -0.1, 0.0, 0.0]);
        let nloc = 20;
        let small = 1e-10;
        let smax = 10;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-9.0, -5.90983, -4.0, 0.0, 3.9];
        let expected_flist = vec![-3.441330099253833e-148, -5.1475826387352304e-145, -1.8158117055687204e-143, -1.2977781437703553e-99, -2.1039811496616056e-20];
        let expected_nf = 5;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }


    #[test]
    fn test_minimum_valid_input() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let nloc = 1;
        let small = 1e-5;
        let smax = 10;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-1.0, -0.618034, -0.381966025156, 0.0];
        let expected_flist = vec![-0.00508911288366444, -1.871372446840273e-18, -1.902009314142582e-62, -3.391970967769076e-191, ];
        let expected_nf = 4;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_small_vector_values() {
        let x = SVector::<f64, 6>::from_row_slice(&[-1e-8, 1e-8, -1e-8, 1e-8, -1e-8, 1e-8]);
        let p = SVector::<f64, 6>::from_row_slice(&[1e-8, -1e-8, 1e-8, -1e-8, 1e-8, -1e-8]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let u = SVector::<f64, 6>::from_row_slice(&[-1e-8, -1e-8, -1e-8, -1e-8, -1e-8, -1e-8]);
        let v = SVector::<f64, 6>::from_row_slice(&[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]);
        let nloc = 1;
        let small = 1e-12;
        let smax = 5;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![0.0, 0.12732200000000002, 0.381966, 1.0];
        let expected_flist = vec![-0.00508911309741038, -0.00508911307019582, -0.005089113015766702, -0.00508911288366444, ];
        let expected_nf = 4;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_edge_case_maximum_search_steps() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let nloc = 1;
        let small = 1e-5;
        let smax = 1;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![0.0, 1.0];
        let expected_flist = vec![-3.391970967769076e-191, -0.00508911288366444, ];
        let expected_nf = 2;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }

    #[test]
    fn test_empty_lists() {
        let x = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut alist = Vec::new();
        let mut flist = Vec::new();
        let u = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let nloc = 1;
        let small = 1e-5;
        let smax = 10;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-1.0, -0.618034, -0.381966025156, 0.0];
        let expected_flist = vec![-0.00508911288366444, -1.871372446840273e-18, -1.902009314142582e-62, -3.391970967769076e-191, ];
        let expected_nf = 4;

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, expected_nf);
    }


    #[test]
    fn test_real_mistake_0() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.2, 0.2, 0.45, 0.56, 0.68, 0.72]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![0.0];
        let mut flist = vec![-2.7];
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nloc = 1;
        let small = 0.1;
        let smax = 6;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.2, -0.1, -0.02602472805313486, -0.008674909351044953, 0.0, 0.016084658451990714, 0.048253975355972145, 0.2];
        let expected_flist = vec![-0.3098962997361745, -0.35807529391557985, -0.36128396643179006, -0.3577687209054916, -2.7, -0.3501457040105073, -0.33610446976533986, -0.23322360512233206];
        let expected_nf = 7;

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, expected_nf);
    }
    #[test]
    fn test_real_mistake_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.190983, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![0.0];
        let mut flist = vec![-0.01523227097945881];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let nloc = 1;
        let small = 0.1;
        let smax = 6;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);


        let expected_alist = vec![-1.6, -1.1, -0.6, -0.40767007775845987, -0.30750741391479774, -0.10250247130493258, 0.0];
        let expected_flist = vec![-0.00033007085742366247, -0.005222762849281021, -0.019362461793975085, -0.02327501776110989, -0.023740401281451076, -0.019450678518268937, -0.01523227097945881];
        let expected_nf = 6;

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, expected_nf);
    }
}