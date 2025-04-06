use nalgebra::SVector;

// find first two points and establish correct scale
pub fn lsinit<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x: &SVector<f64, N>,
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    scale: f64,
) ->
    f64  // alp
{
    let mut alp = 0.0;

    match alist.len() {
        0 => {
            // Evaluate at absolutely smallest point
            if amin > 0.0 { alp = amin; }
            if amax < 0.0 { alp = amax; }
            // New function value
            let falp = func(&(x + p.scale(alp)));
            alist.push(alp);
            flist.push(falp);
        }
        1 => {
            // Evaluate at absolutely smallest point
            if amin > 0.0 { alp = amin; }
            if amax < 0.0 { alp = amax; }
            if (alist[0] - alp).abs() > f64::EPSILON {
                // New function value
                let falp = func(&(x + p.scale(alp)));
                alist.push(alp);
                flist.push(falp);
            }
        }
        _ => {} // TODO: It is not clear how to initialize alp!
    }

    // alist and flist are set - now compute min and max
    let aamin = *alist.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let aamax = *alist.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    debug_assert!(!(amin > aamin || amax < aamax));

    // Establish correct scale
    if aamax - aamin <= scale {
        let alp1 = amin.max((-scale).min(amax));
        let alp2 = amin.max(scale.min(amax));
        alp = f64::INFINITY;

        if aamin - alp1 >= alp2 - aamax { alp = alp1; }
        if alp2 - aamax >= aamin - alp1 { alp = alp2; }
        if alp < aamin || alp > aamax {
            // New function value
            let falp = func(&(x + p.scale(alp)));
            alist.push(alp);
            flist.push(falp);
        }
    }

    debug_assert!(alist.len() > 1); // lsinit bug: no second point found
    alp
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_1() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[1., 0., 0., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::new();
        let mut flist: Vec<f64> = Vec::new();
        let amin = -1.0;
        let amax = 1.0;
        let scale = 0.1;

        let alp = lsinit(hm6, &x, &p, &mut alist, &mut flist, amin, amax, scale);

        let alist_expected = Vec::from([0.0, 0.1]);
        let flist_expected = Vec::from([-1.4069105761385299, -1.4664312887853619]);


        assert_eq!(alist, alist_expected);
        assert_eq!(flist, flist_expected);
        assert_eq!(alp, 0.1);
    }

    #[test]
    fn test_2() {
        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[0., 1., 0., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::from([0.0, 0.1]);
        let mut flist: Vec<f64> = Vec::from([0.0, 0.1]);
        let amin = -1.0;
        let amax = 1.0;
        let scale = 0.1;

        let alp = lsinit(hm6, &x, &p, &mut alist, &mut flist, amin, amax, scale);

        let alist_expected = Vec::from([0.0, 0.1, -0.1]);
        let flist_expected = Vec::from([0.0, 0.1, -1.4091800887102848]);

        assert_eq!(alist, alist_expected);
        assert_eq!(flist, flist_expected);
        assert_eq!(alp, -0.1);
    }

    #[test]
    fn test_first_if() {
        // Matlab equivalent test
        //
        // x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        // p = [0., 1., 0., 0., 0., 0.];
        // alist = [];
        // flist = [];
        // amin = 1.0;
        // amax = 2.0;
        // scale = 300.;
        //
        // lsinit;
        //
        // disp(alist);
        // disp(flist);
        // disp(alp);

        let x = SVector::<f64, 6>::from_row_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[0., 1., 0., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::from([]);
        let mut flist: Vec<f64> = Vec::from([]);
        let amin = 1.0;
        let amax = 2.0;
        let scale = 300.;

        let alp = lsinit(hm6, &x, &p, &mut alist, &mut flist, amin, amax, scale);

        assert_eq!(alist, Vec::from([1., 2.]));
        assert_eq!(flist, Vec::from([-0.034770428308850826, -1.1611374653759681e-6]));
        assert_eq!(alp, 2.);
    }

    #[test]
    fn test_second_if() {
        // Matlab equivalent test
        //
        // x = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6];
        // p = [0., 0., 1., 0., 0., 0.];
        // alist = [0.3];
        // flist = [2.1];
        // amin = -2.0;
        // amax = 1.0;
        // scale = 300.;
        //
        // lsinit;
        //
        // disp(alist);
        // disp(flist);
        // disp(alp);

        let x = SVector::<f64, 6>::from_row_slice(&[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]);
        let p = SVector::<f64, 6>::from_row_slice(&[0., 0., 1., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::from([0.3]);
        let mut flist: Vec<f64> = Vec::from([2.1]);
        let amin = -2.0;
        let amax = 1.0;
        let scale = 300.;

        let alp = lsinit(hm6, &x, &p, &mut alist, &mut flist, amin, amax, scale);

        assert_eq!(alist, Vec::from([0.3, 0., -2.]));
        assert_eq!(flist, Vec::from([2.1, -1.5810439431686927e-12, -5.224055753901483e-13]));
        assert_eq!(alp, -2.);
    }
}
