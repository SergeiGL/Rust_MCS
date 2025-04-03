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

    static TOLERANCE: f64 = 1e-14;

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
    fn test_random_0() {
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

        let expected_alist = vec![-1.9381693087512026, -1.6180171305647408, -1.420152199271449, -1.1, -1.0999998456662263, -1.0999997684993392, -1.0999996913324523, -1.0999993826649046, -1.0999987653298093, -1.099998147994714, -1.0999975306596186, -1.0999950613192369, -1.0999925919788551, -1.0999901226384736, -1.0999802452769472, -1.0999703679154207, -1.0999604905538944, -1.0999209811077888, -1.0998419622155775, -1.0997629433233662, -1.099683924431155, -1.0993678488623095, -1.098735697724619, -1.0981035465869282, -1.0974713954492377, -1.0949427908984752, -1.0898855817969504, -1.0848283726954255, -1.0797711635939005, -1.0595423271878008, -1.0491685697157145, -1.039313490781701, -1.0284873075042742, -1.0190846543756014, -0.9985899654191848, -0.978626981563402, -0.9683334255600031, -0.9653111451110294, -0.9586532268848431, -0.9519953086586569, -0.9516390013465992, -0.9450823087049298, -0.9427779753870207, -0.9409044993830534, -0.9404736420691117, -0.9393214754101571, -0.9387453920806799, -0.938361336527695, -0.9382653226394488, -0.9382173156953257, -0.9381853110659103, -0.9381693087512026];
        let expected_flist = vec![-5.58604459882631e-115, -2.6529412742849885e-115, -7.374919633994484e-116, 0.3, -2.465898834262311e-117, -2.465893820134665e-117, -2.465883791907297e-117, -2.4658637355635548e-117, -2.4658236233203223e-117, -2.4657434006109786e-117, -2.465582962300561e-117, -2.465262114110384e-117, -2.4646205314394678e-117, -2.4633378208251812e-117, -2.4607742176273045e-117, -2.455654276316463e-117, -2.4454433978107165e-117, -2.4251372086119515e-117, -2.3849835305589723e-117, -2.3064826995721238e-117, -2.1564855310700586e-117, -1.8828082624338647e-117, -1.4282117419466468e-117, -8.058148137444308e-118, -2.3713688076954403e-118, -4.428983696129811e-118, -1.0763057236085606e-117, -2.0154182179255524e-117, -2.4049842584556147e-117, -2.4582130367941682e-117, -3.6163057774754522e-118, -2.9443817849340133e-118, -2.6433985694809107e-118, -3.2642517423598656e-118, -2.4649413038245704e-117, -5.97012060227803e-118, -2.458992379572101e-118, -2.4148088292226268e-118, -2.3929966028326985e-118, -2.5496370732386913e-118, -2.4656631802710786e-117, -9.22605984986214e-118, -2.928117286902525e-118, -2.3785576693680694e-118, -2.3749606936539554e-118, -2.3731641148703642e-118, -2.4658436793679434e-117, -1.2363580545504154e-117, -3.7875234552968614e-118, -2.47570838052117e-118, -2.3719671021816795e-118, -2.4658963271973538e-117];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, 51);
    }

    #[test]
    fn test_random_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[-2.25104306, -0.25891982, -0.57220876, -7.00526648, -1.37167218, -3.37688706]);
        let p = SVector::<f64, 6>::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut alist = vec![-0.7];
        let mut flist = vec![0.2];
        let u = SVector::<f64, 6>::from_row_slice(&[-1.3735792077205666, -1.1435284793384632, -1.1430331519476875, -0.2965544838799241, -1.9311058584546728, -1.570044230736461]);
        let v = SVector::<f64, 6>::from_row_slice(&[-1.0735792077205666, -0.8435284793384632, -0.8430331519476875, 0.0034455161200758755, -1.6311058584546727, -1.270044230736461]);
        let nloc = 20;
        let small = 1e-11;
        let smax = 50;

        let nf = gls(hm6, &x, &p, &mut alist, &mut flist, &u, &v, nloc, small, smax);

        let expected_alist = vec![-0.8846086593384632, -0.8493515437520255, -0.8140944281655877, -0.7971870738926792, -0.7902798583320398, -0.7874342253282521, -0.7862703984433798, -0.7855708211241909, -0.785125139764169, -0.7846794584041472, -0.7837880956841035, -0.7820053702440162, -0.7784399193638416, -0.774874468483667, -0.7713090176034924, -0.7677435667233178, -0.7641781158431431, -0.7623953904030558, -0.7606126649629685, -0.7597213022429248, -0.7588299395228812, -0.7583842581628595, -0.7579385768028375, -0.7570472140827939, -0.728523607041397, -0.7142618035206985, -0.701250695905654, -0.7000854208487842, -0.7, -0.698926562791997, -0.6978531255839941, -0.6944905535337613, -0.6889811070675227, -0.6779622141350454, -0.6559244282700909, -0.6318248865218592, -0.6299912246206081, -0.6292088265730588, -0.6286842281655878, -0.6284305072336874, -0.6281767863017871, -0.6276693444379864, -0.6237137371210822, -0.6187432460765767, -0.612694844914301, -0.6066464437520255, -0.6048217486774715, -0.5978919959043376, -0.5947152040079673, -0.5916448807994049, -0.5896619316732152, -0.5846086593384632];
        let expected_flist = vec![-3.6486666585943875e-152, -1.0803366744954857e-151, -3.1202323292885654e-151, -5.143374274850338e-151, -6.298113056111975e-151, -6.844293414832105e-151, -7.080765545344288e-151, -1.6467579155022036e-150, -7.321245517906356e-151, -7.416969067795014e-151, -7.612096498798265e-151, -8.017504576462822e-151, -8.892548959109939e-151, -9.860589844505158e-151, -1.0931231837672606e-150, -1.2115041248076118e-150, -1.3423639183099886e-150, -1.4128678720567895e-150, -1.4869803323158416e-150, -1.52544554854951e-150, -1.5648809162133424e-150, -1.584969825627249e-150, -1.6053102458864408e-150, 0.2, -3.691908794763577e-150, -5.494289172160822e-150, -7.868454054544235e-150, -8.124320622193561e-150, -5.480881447124661e-149, -8.386803466329519e-150, -8.637291212342694e-150, -9.469998774805086e-150, -1.1006033565301753e-149, -1.4838899679685561e-149, -2.677809095340146e-149, -5.050265583349299e-149, -5.297516289370337e-149, -5.4065546511784635e-149, -1.692475594480888e-148, -5.517184296473476e-149, -5.553720448932115e-149, -5.627498474273394e-149, -6.236139983685429e-149, -7.091967074437646e-149, -8.287814949876298e-149, -9.678222969295206e-149, -1.0140356243010133e-148, -1.2098267751338764e-148, -1.3113886237217697e-148, -1.4173693832299761e-148, -1.4901768519684082e-148, -7.226728668820279e-151];

        assert_eq!(alist, expected_alist);
        assert_eq!(flist, expected_flist);
        assert_eq!(nf, 51);
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