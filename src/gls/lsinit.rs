use crate::feval::feval;
use ndarray::Array1;
use std::process;

pub fn lsinit(
    x: &Array1<f64>,
    p: &Array1<f64>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    scale: f64,
) -> (f64, //alp
      f64, //alp1
      f64, //alp2
      f64, //falp
) {
    let mut alp = 0.0;
    let mut alp1 = 0.0;
    let mut alp2 = 0.0;
    let mut falp = 0.0;

    if alist.is_empty() {
        // Evaluate at absolutely smallest point
        alp = 0.0;
        if amin > 0.0 {
            alp = amin;
        }
        if amax < 0.0 {
            alp = amax;
        }
        // New function value
        let step = p * alp;
        falp = feval(&(x + &step));
        alist.push(alp);
        flist.push(falp);
    } else if alist.len() == 1 {
        // Evaluate at absolutely smallest point
        alp = 0.0;
        if amin > 0.0 {
            alp = amin;
        }
        if amax < 0.0 {
            alp = amax;
        }
        if (alist[0] - alp).abs() > f64::EPSILON {
            // New function value
            let step = p * alp;
            falp = feval(&(x + &step));
            alist.push(alp);
            flist.push(falp);
        }
    }

    // alist and flist are set - now compute min and max
    let aamin = alist.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let aamax = alist.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if amin > aamin || amax < aamax {
        eprintln!("GLS Error: non-admissible step in alist");
        process::exit(1);
    }

    // Establish correct scale
    if aamax - aamin <= scale {
        alp1 = amin.max((-scale).min(amax));
        alp2 = amin.max(scale.min(amax));
        alp = f64::INFINITY;

        if aamin - alp1 >= alp2 - aamax {
            alp = alp1;
        }
        if alp2 - aamax >= aamin - alp1 {
            alp = alp2;
        }
        if alp < aamin || alp > aamax {
            // New function value
            let step = p * alp;
            falp = feval(&(x + &step));
            alist.push(alp);
            flist.push(falp);
        }
    }

    if alist.len() == 1 {
        eprintln!("GLS Error: lsinit bug: no second point found");
        process::exit(1);
    }
    (alp, alp1, alp2, falp)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_1() {
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = Array1::from_vec(vec![1., 0., 0., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::new();
        let mut flist: Vec<f64> = Vec::new();
        let amin = -1.0;
        let amax = 1.0;
        let scale = 0.1;

        let (alp, alp1, alp2, falp) = lsinit(&x, &p, &mut alist, &mut flist, amin, amax, scale);

        let alist_expected = Vec::from([0.0, 0.1]);
        let flist_expected = Vec::from([-1.4069105761385299, -1.4664312887853619]);

        assert_eq!(alist.len(), alist_expected.len());
        assert_eq!(flist.len(), flist_expected.len());

        for (a, e) in alist.iter().zip(alist_expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-10);
        }

        for (a, e) in flist.iter().zip(flist_expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-10);
        }

        assert_eq!((alp, alp1, alp2, falp), (0.1, -0.1, 0.1, -1.4664312887853619));
    }

    #[test]
    fn test_2() {
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = Array1::from_vec(vec![0., 1., 0., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::from([0.0, 0.1]);
        let mut flist: Vec<f64> = Vec::from([0.0, 0.1]);
        let amin = -1.0;
        let amax = 1.0;
        let scale = 0.1;

        let (alp, alp1, alp2, falp) = lsinit(&x, &p, &mut alist, &mut flist, amin, amax, scale);

        let alist_expected = Vec::from([0.0, 0.1, -0.1]);
        let flist_expected = Vec::from([0.0, 0.1, -1.4091800887102848]);

        assert_eq!(alist.len(), alist_expected.len());
        assert_eq!(flist.len(), flist_expected.len());

        for (a, e) in alist.iter().zip(alist_expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-10);
        }

        for (a, e) in flist.iter().zip(flist_expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-10);
        }

        assert_eq!((alp, alp1, alp2, falp), (-0.1, -0.1, 0.1, -1.4091800887102848));
    }
}
