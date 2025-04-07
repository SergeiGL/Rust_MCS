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


// Do not forget that in Matlab x, p, u, v should be supplied with ; like this:
// x = [0.25; 0.35; 0.45; 0.55; 0.65; 0.75];
// p = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6];
// u = [0.3; 0.4; 0.5; 0.6; 0.7; 0.8];
// v = [0.8; 0.9; 1.0; 1.1; 1.2; 1.3];
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
    loop {
        lsdescent(func, x, p, alist, flist, &mut alp, &mut abest, &mut fbest, &mut fmed, &mut up,
                  &mut down, &mut monotone, &mut minima, &mut nmin, &mut unitlen, &mut s);

        // Check saturation
        lssat(small, alist, flist, &mut alp, amin, amax, s, &mut saturated);

        if saturated || s == sold || s >= smax {
            break;
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

    static TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_random() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.019; 0.139; 0.1826; 0.549; 0.0223; 0.6465];
        // p = [1.0; 1.0; 1.0; 1.0; 1.0; 1.0];
        // alist = [0.6, 0.7];
        // flist = [2.2, 3.0];
        // u = [-1.0; 0.0; -3.0; 0.0; 0.0; 0.0];
        // v = [1.739; 1.646; 1.1253; 1.642; 1.028; 1.4];
        // nloc = 1;
        // small = 1e-7;
        // smax = 30;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.019, 0.139, 0.1826, 0.549, 0.0223, 0.6465]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut alist = vec![0.6, 0.7];
        let mut flist = vec![2.2, 3.0];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.0, 0.0, -3.0, 0.0, 0.0, 0.0, ]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.739, 1.646, 1.1253, 1.642, 1.028, 1.4]);
        let nloc = 1;
        let small = 1e-7;
        let smax = 30;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.0223, 0.08018569606073983, 0.08439954704129399, 0.1851333333333333, 0.6, 0.7, ];
        let expected_flist = vec![-0.2804054430856644, -0.4381692070159985, -0.4379814420123991, -0.2924682723436872, 2.2, 3.];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 4);
    }

    #[test]
    fn test_random_0() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.09; 0.19; 0.186; 0.59; 0.023; 0.665];
        // p = [1.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        // alist = [1.];
        // flist = [2.];
        // u = [-1.0; 0.0; -3.0; 0.0; 0.0; 0.0];
        // v = [1.79; 1.66; 1.153; 1.62; 1.08; 1.0];
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
        // disp(" ");
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

        let expected_alist = vec![-1.09, -0.4163429400000001, 0., 0.04958094998697257, 0.1487428499609177, 0.3333333333333333, 1., 1.175, 1.261764405223204, 1.35, 1.7, ];
        let expected_flist = vec![-0.003176069409057748, -0.1166973373329338, -0.2882270522230002, -0.2985424736368541, -0.3055508078936456, -0.2712291587371468, 2., -0.01241176429743812, -0.007146649331859215, -0.003917088745748065, -0.0003190360264916434, ];

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
        // x = [0.19; 0.39; 0.486; 0.09; 0.4023; 0.0665];
        // p = [1.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        // alist = [1.];
        // flist = [2.];
        // u = [-1.0; 0.0; -3.0; 0.0; 0.0; 0.0];
        // v = [1.79; 1.66; 1.153; 1.62; 1.08; 1.0];
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

        let expected_alist = vec![-1.19, -0.45453953999999996, 0., 0.03570504644416015, 0.06581721440990665, 0.0665995552812049, 0.06661419768084426, 0.1071151393324805, 0.3333333333333333, 1., 1.00234375, 1.00351554775829, 1.0046875, 1.009375, 1.014061324756644, 1.01875, 1.0375, 1.056234722646366, 1.075, 1.15, 1.224903210970236, 1.3, 1.6, ];
        let expected_flist = vec![-0.000795900205426112, -0.04998401289642895, -0.1584547222565412, -0.160850892243273, -0.1615337527074966, -0.1615342022657176, -0.1615342021050759, -0.1602867240702137, -0.1075717931849698, 2., -0.004928499474365494, -0.004895371121026823, -0.004862421944517106, -0.00473245483205075, -0.004605398117693165, -0.004481111483922252, -0.004011515899356581, -0.003583993105764202, -0.003194756685133813, -0.001975930499767486, -0.001182480553159051, -0.000683301031252829, -5.478276420538117e-05, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 22);
    }

    #[test]
    fn test_random_3() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [-2.25104306; -0.25891982; -0.57220876; -7.00526648; -1.37167218; -3.37688706];
        // p = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0];
        // alist = [-0.7];
        // flist = [0.2];
        // u = [-1.3735792077205666; -1.1435284793384632; -1.1430331519476875; -0.2965544838799241; -1.9311058584546728; -1.570044230736461];
        // v = [-1.0735792077205666; -0.8435284793384632; -0.8430331519476875; 0.0034455161200758755; -1.6311058584546727; -1.270044230736461];
        // nloc = 11;
        // small = 1e-11;
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

        let x = SVector::<f64, 6>::from_row_slice(&[-2.25104306, -0.25891982, -0.57220876, -7.00526648, -1.37167218, -3.37688706]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-0.7];
        let mut flist = vec![0.2];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.3735792077205666, -1.1435284793384632, -1.1430331519476875, -0.2965544838799241, -1.9311058584546728, -1.570044230736461]);
        let v = SVector::<f64, 6>::from_row_slice(&[-1.0735792077205666, -0.8435284793384632, -0.8430331519476875, 0.0034455161200758755, -1.6311058584546727, -1.270044230736461]);
        let nloc = 11;
        let small = 1e-11;
        let smax = 20;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.8846086593384632, -0.8493515437520255, -0.8140944281655877, -0.7855708211241909, -0.7570472140827939, -0.728523607041397, -0.7213927052810477, -0.7142618035206985, -0.7071309017603492, -0.7053481763202618, -0.7035654508801745, -0.7017827254400872, -0.7013370440800654, -0.7008913627200436, -0.7004456813600217, -0.7003342610200163, -0.7002228406800108, -0.7, -0.6286842281655878, -0.6066464437520255, -0.5846086593384632, ];
        let expected_flist = vec![-3.648666658594388e-152, -1.080336674495486e-151, -3.120232329288565e-151, -7.226728668820279e-151, -1.646757915502204e-150, -3.691908794763577e-150, -4.50611297697574e-150, -5.494289172160822e-150, -6.692359693584663e-150, -7.02954480745711e-150, -7.383249207733504e-150, -7.754257998285516e-150, -7.849809425494924e-150, -7.946506711375157e-150, -8.04436319808243e-150, -8.069010022721969e-150, -8.093730352257853e-150, 0.2, -5.480881447124661e-149, -9.678222969295206e-149, -1.692475594480888e-148, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 20);
    }

    #[test]
    fn test_random_4() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [4.094483795775204; 0.9485906963490542; 2.86519204314813; 3.5992828245887476; 2.023854270382081; 4.665965192082808];
        // p = [0.0; 0.0; 0.0; 0.0; 1.0; 0.0];
        // alist = [-1.1];
        // flist = [0.3];
        // u = [0.7910576746302738; 1.6659266237072177; 2.1531189830717956; 2.6279505108000976; 0.08568496163087858; 0.024160521508185373];
        // v = [1.7910576746302738; 2.6659266237072177; 3.1531189830717956; 3.6279505108000976; 1.0856849616308786; 1.0241605215081853];
        // nloc = 10;
        // small = 1e-10;
        // smax = 50;
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

        let x = SVector::<f64, 6>::from_row_slice(&[4.094483795775204, 0.9485906963490542, 2.86519204314813, 3.5992828245887476, 2.023854270382081, 4.665965192082808]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut alist = vec![-1.1];
        let mut flist = vec![0.3];
        let u = SVector::<f64, 6>::from_row_slice(&[0.7910576746302738, 1.6659266237072177, 2.1531189830717956, 2.6279505108000976, 0.08568496163087858, 0.024160521508185373]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.7910576746302738, 2.6659266237072177, 3.1531189830717956, 3.6279505108000976, 1.0856849616308786, 1.0241605215081853]);
        let nloc = 10;
        let small = 1e-10;
        let smax = 50;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-1.938169308751203, -1.618017130564741, -1.420152199271449, -1.1, -1.099999999849284, -1.099999999773925, -1.099999999698567, -1.099999999397134, -1.099999999095701, -1.099999998794267, -1.099999997588535, -1.099999996382802, -1.09999999517707, -1.099999990354139, -1.099999985531209, -1.099999980708278, -1.099999961416557, -1.099999942124835, -1.099999922833113, -1.099999845666226, -1.099999768499339, -1.099999691332452, -1.099999382664905, -1.099999073997357, -1.099998765329809, -1.099997530659619, -1.099996295989428, -1.099995061319237, -1.099990122638474, -1.09998518395771, -1.099980245276947, -1.099960490553894, -1.099940735830842, -1.099920981107789, -1.099841962215578, -1.099762943323366, -1.099683924431155, -1.09936784886231, -1.099051773293464, -1.098735697724619, -1.097471395449238, -1.096207093173856, -1.094942790898475, -1.08988558179695, -1.084828372695426, -1.0797711635939, -1.059542327187801, -1.039313490781701, -1.019084654375601, -0.9381693087512026, ];
        let expected_flist = vec![-5.58604459882631e-115, -2.652941274284988e-115, -7.374919633994484e-116, 0.3, -2.465903843502458e-117, -2.465903841054102e-117, -2.465903838605886e-117, -2.4659038288126e-117, -2.465903819019314e-117, -2.465903809226169e-117, -2.465903770053167e-117, -2.465903730880166e-117, -2.465903691707166e-117, -2.46590353501531e-117, -2.465903378323324e-117, -2.465903221631488e-117, -2.465902594864106e-117, -2.465901968096742e-117, -2.465901341329537e-117, -2.465898834262311e-117, -2.465896327197354e-117, -2.465893820134665e-117, -2.465883791907297e-117, -2.465873763716927e-117, -2.465863735563555e-117, -2.465823623320322e-117, -2.465783511669466e-117, -2.465743400610979e-117, -2.465582962300561e-117, -2.465422533467146e-117, -2.465262114110384e-117, -2.464620531439468e-117, -2.463979100354834e-117, -2.463337820825181e-117, -2.460774217627305e-117, -2.458213036794168e-117, -2.455654276316463e-117, -2.445443397810717e-117, -2.435271084565012e-117, -2.425137208611952e-117, -2.384983530558972e-117, -2.345434702120587e-117, -2.306482699572124e-117, -2.156485531070059e-117, -2.015418217925552e-117, -1.882808262433865e-117, -1.428211741946647e-117, -1.076305723608561e-117, -8.058148137444308e-118, -2.37136880769544e-118, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 49);
    }

    #[test]
    fn test_random_5() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.5; 0.3; 0.1; 0.7; 4.0; -1.0];
        // p = [1.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        // alist = [];
        // flist = [];
        // u = [0.2; 0.1; 0.3; 0.4; 0.3; 0.4];
        // v = [0.7; 0.6; 0.8; 0.9; 0.8; 0.9];
        // nloc = 5;
        // small = 1e-8;
        // smax = 15;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 0.3, 0.1, 0.7, 4.0, -1.0]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![];
        let mut flist = vec![];
        let u = SVector::<f64, 6>::from_row_slice(&[0.2, 0.1, 0.3, 0.4, 0.3, 0.4]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.7, 0.6, 0.8, 0.9, 0.8, 0.9]);
        let nloc = 5;
        let small = 1e-8;
        let smax = 15;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.3, -0.229179602882621, -0.1854101924532, -0.1145898, -0.1069898628193395, -0.09530263583285196, -0.09530000214065119, -0.09524644937944267, -0.09445727806127491, -0.03566328760644651, 0., 0.07639319999999999, 0.1236067949688, 0.1527864019217473, 0.2, ];
        let expected_flist = vec![-5.301049203084008e-09, -7.968821195295377e-09, -9.414046650921805e-09, -1.073935096015936e-08, -1.078242177379952e-08, -1.080749952543113e-08, -1.080749952670756e-08, -1.08074989998379e-08, -1.080736904785538e-08, -1.017342732629142e-08, -9.261303113537727e-09, -6.547641182732688e-09, -4.785531766659876e-09, -3.795983848418976e-09, -2.454209623420858e-09, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 15);
    }

    #[test]
    fn test_with_negative_values() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [-1.2; -0.5; -0.8; -1.0; -0.3; -0.7];
        // p = [-0.2; -0.1; -0.3; -0.4; -0.6; -0.1];
        // alist = [-0.2];
        // flist = [0.8];
        // u = [-2.0; -1.5; -1.1; -1.3; -0.9; -1.2];
        // v = [-0.5; -0.2; -0.3; -0.7; -0.1; -0.4];
        // nloc = 8;
        // small = 1e-9;
        // smax = 25;
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

        let x = SVector::<f64, 6>::from_row_slice(&[-1.2, -0.5, -0.8, -1.0, -0.3, -0.7]);
        let p = SVector::<f64, 6>::from_row_slice(&[-0.2, -0.1, -0.3, -0.4, -0.6, -0.1]);
        let mut alist = vec![-0.2];
        let mut flist = vec![0.8];
        let u = SVector::<f64, 6>::from_row_slice(&[-2.0, -1.5, -1.1, -1.3, -0.9, -1.2]);
        let v = SVector::<f64, 6>::from_row_slice(&[-0.5, -0.2, -0.3, -0.7, -0.1, -0.4]);
        let nloc = 8;
        let small = 1e-9;
        let smax = 25;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.3333333333333333, -0.3, -0.2666666666666667, -0.2, -0.1999755859375, -0.19996337890625, -0.199951171875, -0.19990234375, -0.199853515625, -0.1998046875, -0.199609375, -0.1994140625, -0.19921875, -0.1984375, -0.19765625, -0.196875, -0.19375, -0.190625, -0.1875, -0.175, -0.1625, -0.15, -0.1, -0.05, 0., 0.5, 0.7500000000000001, ];
        let expected_flist = vec![-3.079327311582392e-17, -1.428633305260837e-17, -6.509695368831587e-18, 0.8, -1.279678995704739e-18, -1.279289587836128e-18, -1.278900295374505e-18, -1.277344278933239e-18, -1.275790106346432e-18, -1.274237775490701e-18, -1.268046826994437e-18, -1.261885176942228e-18, -1.255752690572626e-18, -1.231511707668086e-18, -1.207726717916345e-18, -1.184389380696946e-18, -1.095354870502306e-18, -1.01285297469437e-18, -9.364168020046052e-19, -6.830816364074372e-19, -4.970220188550708e-19, -3.607266714098528e-19, -9.758539381132908e-20, -2.535041570826806e-20, -6.323813503627791e-21, -6.347055980559205e-28, -4.396760222516645e-32, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 26);
    }

    #[test]
    fn test_with_large_values() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.5324; 0.3234; 0.7523; 0.2464810; 0.90237; 0.1111345];
        // p = [1.0; 0.0; 1.0; 1.0; 1.0; 1.0];
        // alist = [200.0, 1000.1,];
        // flist = [5.0, -1.0];
        // u = [10.0; 15.0; 20.0; 25.0; 30.0; 35.0];
        // v = [1250.0; 1260.0; 1270.0; 1280.0; 1290.0; 12100.0];
        // nloc = 12;
        // small = 1e-6;
        // smax = 30;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.5324, 0.3234, 0.7523, 0.2464810, 0.90237, 0.1111345]);
        let p = SVector::<f64, 6>::from_row_slice(&[1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let mut alist = vec![200.0, 1000.1];
        let mut flist = vec![5.0, -1.0];
        let u = SVector::<f64, 6>::from_row_slice(&[10.0, 15.0, 20.0, 25.0, 30.0, 35.0]);
        let v = SVector::<f64, 6>::from_row_slice(&[1250.0, 1260.0, 1270.0, 1280.0, 1290.0, 12100.0]);
        let nloc = 12;
        let small = 1e-6;
        let smax = 30;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![34.8888655, 76.16664912500002, 96.80554093750004, 107.12498684375004, 112.28470979687503, 114.86457127343753, 116.15450201171879, 116.79946738085941, 117.44443275, 200., 272.14497856024815, 316.73302530321655, 388.8779991120498, 447.2445079196827, 505.6110167273156, 600.0500100636577, 647.2695067318289, 694.4890034, 749.8344059162557, 805.1798084325113, 860.5252109487669, 915.8706134650226, 968.098910098767, 994.2130584156391, 997.4773269552481, 1000.1, 1000.7415954948572, 1007.2701325740752, 1020.3272067325114, 1072.5555033662556, 1124.7838, 1187.1257, 1249.4676, ];
        let expected_flist = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 31);
    }

    #[test]
    fn test_with_nonzero_starting_lists() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.25; 0.35; 0.45; 0.55; 0.65; 0.75];
        // p = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6];
        // alist = [0.5, 0.5, 0.5, 0.6];
        // flist = [3.0, 2.5, 2.0, 3.5];
        // u = [0.3; 0.4; 0.5; 0.6; 0.7; 0.8];
        // v = [0.8; 0.9; 1.0; 1.1; 1.2; 1.3];
        // nloc = 7;
        // small = 1e-8;
        // smax = 18;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.25, 0.35, 0.45, 0.55, 0.65, 0.75]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut alist = vec![0.5, 0.5, 0.5, 0.6];
        let mut flist = vec![3.0, 2.5, 2.0, 3.5];
        let u = SVector::<f64, 6>::from_row_slice(&[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.8, 0.9, 1.0, 1.1, 1.2, 1.3]);
        let nloc = 7;
        let small = 1e-8;
        let smax = 18;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![0.5, 0.5, 0.5, 0.5432126952758349, 0.54397369927755, 0.5448342054639075, 0.5465584674502496, 0.55, 0.6, 0.6024739583333334, 0.6037108175086641, 0.6049479166666667, 0.6098958333333333, 0.6148419795598689, 0.6197916666666667, 0.6395833333333334, 0.6593547790394046, 0.6791666666666667, 0.7583333333333333, 0.8374300654302065, 0.9166666666666667, ];
        let expected_flist = vec![3., 2.5, 2., -0.01320202006244285, -0.01312293991130451, -0.01303399125728608, -0.01285725548054592, -0.01251041643139657, 3.5, -0.008111115745154726, -0.008025664914064258, -0.007940957925014397, -0.007609663036358585, -0.00729023471041852, -0.006982012598028483, -0.005857593136855228, -0.004892462939473615, -0.004065726825226937, -0.001849851377080479, -0.0007789525093141945, -0.0003024287591470743, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 17);
    }

    #[test]
    fn test_with_very_small_tolerance() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.15; 0.25; 0.35; 0.45; 0.55; 0.65];
        // p = [0.05; 0.15; 0.25; 0.35; 0.45; 0.55];
        // alist = [0.3];
        // flist = [1.2];
        // u = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6];
        // v = [0.6; 0.7; 0.8; 0.9; 1.0; 1.1];
        // nloc = 15;
        // small = 1e-15;
        // smax = 40;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.15, 0.25, 0.35, 0.45, 0.55, 0.65]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.05, 0.15, 0.25, 0.35, 0.45, 0.55]);
        let mut alist = vec![0.3];
        let mut flist = vec![1.2];
        let u = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.6, 0.7, 0.8, 0.9, 1.0, 1.1]);
        let nloc = 15;
        let small = 1e-15;
        let smax = 40;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.09090909090909098, -0.03030303030303033, 0., 0.3, 0.3000000381772496, 0.3000000572658733, 0.3000000763544993, 0.3000001527089986, 0.3000002290634785, 0.3000003054179972, 0.3000006108359944, 0.3000009162536819, 0.3000012216719887, 0.3000024433439775, 0.3000036650110121, 0.300004886687955, 0.3000097733759099, 0.3000146599846029, 0.30001954675182, 0.3000390935036399, 0.3000586389875421, 0.3000781870072798, 0.3001563740145596, 0.3002345407527755, 0.3003127480291193, 0.3006254960582386, 0.3009379209071388, 0.3012509921164772, 0.3025019842329545, 0.3037478767309435, 0.305003968465909, 0.3100079369318182, 0.3149346916366831, 0.3200158738636364, 0.3400317477272727, 0.3590504682852974, 0.3800634954545454, 0.4601269909090909, 0.5325518907438376, 0.6202539818181818, 0.7192179000000001, 0.8181818181818182, ];
        let expected_flist = vec![-1.331208712940245, -1.100183965226246, -0.9868690630994547, 1.2, -0.2580477595062831, -0.2580477364093692, -0.2580477133124545, -0.2580476209248239, -0.2580475285372515, -0.2580474361496672, -0.2580470665997718, -0.2580466970508086, -0.2580463275016535, -0.2580448493121072, -0.2580433711374757, -0.258041892959776, -0.2580359803621572, -0.2580300680031594, -0.2580241555950819, -0.2580005077734872, -0.2579768637688354, -0.2579532189797674, -0.2578586687841939, -0.2577641795964878, -0.2576696779124047, -0.2572921338614723, -0.2569155619028214, -0.2565387934532515, -0.255039078817324, -0.2535546628073115, -0.2520673178699071, -0.246232904124891, -0.2406277690288879, -0.2349880954686717, -0.2140972806867099, -0.1960601226510194, -0.1779736370627656, -0.1232785803071267, -0.087910803818448, -0.0569916346800095, -0.03318640053013867, -0.0179993867046353, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 41);
    }

    #[test]
    fn test_with_minimal_iterations() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.12; 0.22; 0.32; 0.42; 0.52; 0.62];
        // p = [0.01; 0.02; 0.03; 0.04; 0.05; 0.06];
        // alist = [5.25, 3.1];
        // flist = [0.75, 1.0];
        // u = [0.15; 0.25; 0.35; 0.45; 0.55; 0.65];
        // v = [0.65; 0.75; 0.85; 0.95; 1.05; 1.15];
        // nloc = 2;
        // small = 1e-6;
        // smax = 3;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.12, 0.22, 0.32, 0.42, 0.52, 0.62]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]);
        let mut alist = vec![5.25, 3.1];
        let mut flist = vec![0.75, 1.0];
        let u = SVector::<f64, 6>::from_row_slice(&[0.15, 0.25, 0.35, 0.45, 0.55, 0.65]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.65, 0.75, 0.85, 0.95, 1.05, 1.15]);
        let nloc = 2;
        let small = 1e-6;
        let smax = 3;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist =
            vec![3.1, 5.25, 8.833333333333332];
        let expected_flist =
            vec![1., 0.75, -0.009015252684460929];
        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 1);
    }

    #[test]
    fn test_with_many_iterations() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.33; 0.44; 0.55; 0.66; 0.77; 0.88];
        // p = [0.11; 0.22; 0.33; 0.44; 0.55; 0.66];
        // alist = [0.4];
        // flist = [1.4];
        // u = [0.2; 0.3; 0.4; 0.5; 0.6; 0.7];
        // v = [0.7; 0.8; 0.9; 1.0; 1.1; 1.2];
        // nloc = 20;
        // small = 1e-12;
        // smax = 100;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.33, 0.44, 0.55, 0.66, 0.77, 0.88]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.11, 0.22, 0.33, 0.44, 0.55, 0.66]);
        let mut alist = vec![0.4];
        let mut flist = vec![1.4];
        let u = SVector::<f64, 6>::from_row_slice(&[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
        let nloc = 20;
        let small = 1e-12;
        let smax = 100;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.2727272727272728, -0.09090909090909093, 0., 0.1527864, 0.4, 0.4000000000000001, 0.4000000000000002, 0.4000000000000004, 0.4000000000000005, 0.4000000000000006, 0.4000000000000012, 0.4000000000000019, 0.4000000000000025, 0.4000000000000049, 0.4000000000000072, 0.4000000000000097, 0.4000000000000193, 0.400000000000029, 0.4000000000000386, 0.4000000000000772, 0.4000000000001158, 0.4000000000001543, 0.4000000000003087, 0.400000000000463, 0.4000000000006174, 0.4000000000012347, 0.4000000000018521, 0.4000000000024694, 0.4000000000049388, 0.4000000000074083, 0.4000000000098777, 0.4000000000197553, 0.400000000029633, 0.4000000000395107, 0.4000000000790214, 0.400000000118532, 0.4000000001580427, 0.4000000003160853, 0.4000000004741279, 0.4000000006321705, 0.400000001264341, 0.4000000018965115, 0.400000002528682, 0.4000000050573638, 0.4000000075860457, 0.4000000101147276, 0.4000000202294552, 0.4000000303441829, 0.4000000404589104, 0.4000000809178208, 0.400000121376731, 0.4000001618356416, 0.4000003236712831, 0.4000004855069222, 0.4000006473425663, 0.4000012946851326, 0.4000019420276593, 0.4000025893702652, 0.4000051787405303, 0.4000077681101619, 0.4000103574810606, 0.4000207149621212, 0.4000310724330467, 0.4000414299242425, 0.4000828598484849, 0.4001242896107045, 0.4001657196969697, 0.4003314393939394, 0.4004971565074051, 0.4006628787878788, 0.4013257575757576, 0.4019885955905637, 0.4026515151515152, 0.4053030303030303, 0.4079539280915952, 0.4106060606060606, 0.4212121212121212, 0.4318102969542484, 0.4424242424242424, 0.4848484848484848, ];
        let expected_flist = vec![-0.4148772466154029, -0.1378006984278799, -0.07863237832159926, -0.02809250452629235, 1.4, -0.003124123189016663, -0.003124123189016663, -0.003124123189016659, -0.003124123189016652, -0.003124123189016644, -0.003124123189016631, -0.003124123189016608, -0.003124123189016589, -0.00312412318901651, -0.00312412318901643, -0.003124123189016351, -0.003124123189016029, -0.003124123189015709, -0.003124123189015391, -0.003124123189014109, -0.00312412318901283, -0.003124123189011556, -0.003124123189006437, -0.003124123189001319, -0.003124123188996206, -0.003124123188975737, -0.00312412318895527, -0.003124123188934805, -0.003124123188852945, -0.003124123188771085, -0.003124123188689227, -0.003124123188361786, -0.003124123188034347, -0.003124123187706904, -0.003124123186397136, -0.003124123185087373, -0.003124123183777605, -0.003124123178538544, -0.003124123173299479, -0.003124123168060421, -0.003124123147104175, -0.003124123126147927, -0.003124123105191682, -0.003124123021366696, -0.003124122937541707, -0.003124122853716724, -0.003124122518416809, -0.003124122183116929, -0.003124121847817079, -0.003124120506617987, -0.003124119165419401, -0.003124117824221299, -0.003124112459433899, -0.003124107094654546, -0.003124101729883002, -0.003124080270876829, -0.003124058811999455, -0.00312403735324695, -0.003123951519517077, -0.003123865687848, -0.003123779858176686, -0.003123436559972262, -0.003123093294736776, -0.00312275006146008, -0.003121377455952248, -0.003120005377697815, -0.003118633810441926, -0.003113152776957687, -0.003107680163569835, -0.003102215704283668, -0.003080441249751249, -0.003058800499230784, -0.003037288820205381, -0.002952551723590431, -0.002869889728609846, -0.002789205161152755, -0.002485904280861753, -0.002211937174943907, -0.001964385356903246, -0.001200941803919226, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 79);
    }

    #[test]
    fn test_edge_case_identical_vectors() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.5; 0.5; 0.5; 0.5; 0.5; 0.5];
        // p = [0.5; 0.5; 0.5; 0.5; 0.5; 0.5];
        // alist = [0.5];
        // flist = [1.5];
        // u = [0.5; 0.5; 0.5; 0.5; 0.5; 0.5];
        // v = [1.5; 1.5; 1.5; 1.5; 1.5; 1.5];
        // nloc = 5;
        // small = 1e-7;
        // smax = 15;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let mut alist = vec![0.5];
        let mut flist = vec![1.5];
        let u = SVector::<f64, 6>::from_row_slice(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.5, 1.5, 1.5, 1.5, 1.5, 1.5]);
        let nloc = 5;
        let small = 1e-7;
        let smax = 15;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist =
            vec![0., 0.002357814814814815, 0.007073444444444446, 0.02122033333333334, 0.06366100000000001, 0.190983, 0.5, 0.515625, 0.5234310152279966, 0.53125, 0.5625, 0.5936846203147993, 0.625, 0.75, 0.8748576104933676, 1., 1.5, 2.];
        let expected_flist =
            vec![-0.5053149917022333, -0.4982684029025751, -0.4843034800054251, -0.4435058483240779, -0.3324587256685739, -0.1143848161128617, 1.5, -0.005944145628576739, -0.005622468788683764, -0.005318881044098518, -0.004261319728800603, -0.003401049496124226, -0.002686101828098329, -0.0008894161671529885, -0.0002102914481126549, -3.408539273427753e-05, -5.152905323849509e-10, -1.674100549072578e-17, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 17);
    }

    #[test]
    fn test_with_varying_parameters() {
        // Matlab equivalent test
        //
        // func = "hm6";
        // data = "hm6";
        // prt = 0;
        // x = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6];
        // p = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6];
        // alist = [0.0, 0.25, 0.5, 0.75, 0.25];
        // flist = [2.0, 1.5, 1.0, 1.5, 2.0];
        // u = [0.05; 0.15; 0.25; 0.35; 0.45; 0.55];
        // v = [0.55; 0.65; 0.75; 0.85; 0.95; 1.05];
        // nloc = 8;
        // small = 1e-9;
        // smax = 25;
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

        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut alist = vec![0.0, 0.25, 0.5, 0.75, 0.25];
        let mut flist = vec![2.0, 1.5, 1.0, 1.5, 2.0];
        let u = SVector::<f64, 6>::from_row_slice(&[0.05, 0.15, 0.25, 0.35, 0.45, 0.55]);
        let v = SVector::<f64, 6>::from_row_slice(&[0.55, 0.65, 0.75, 0.85, 0.95, 1.05]);
        let nloc = 8;
        let small = 1e-9;
        let smax = 25;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist =
            vec![-0.08333333333333323, -0.0566861062365032, -0.02777777777777775, 0., 0.00048828125, 0.000732039573965991, 0.0009765625, 0.001953125, 0.00390625, 0.005834677614046809, 0.0078125, 0.015625, 0.03125, 0.04564887156786142, 0.0625, 0.125, 0.1309326570439365, 0.25, 0.25, 0.25, 0.25, 0.2841180454794854, 0.375, 0.4950420016536347, 0.5, 0.75, ];
        let expected_flist =
            vec![-1.702585724868347, -1.620374292731841, -1.516783816035308, 2., -1.40491190980703, -1.403913420954234, -1.402911321397702, -1.398904459823381, -1.390868567142536, -1.382906344337312, -1.374712645882838, -1.342100351729808, -1.275957929898487, -1.214372504925084, -1.142172879850576, -0.8837271620933401, -0.8606132389760937, 1.5, 2., -0.4784007916743019, -0.4784007916743019, -0.3997069805795279, -0.2475524111912364, -0.1342687584771292, 1., 1.5, ];

        assert_relative_eq!(alist.as_slice(), expected_alist.as_slice(), epsilon = TOLERANCE);
        assert_relative_eq!(flist.as_slice(), expected_flist.as_slice(), epsilon = TOLERANCE);
        assert_eq!(nf, 21);
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