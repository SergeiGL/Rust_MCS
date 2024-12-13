use crate::feval::feval;
use nalgebra::SVector;
use std::process;

pub fn lsinit<const N: usize>(
    x: &[f64; N],
    p: &SVector<f64, N>,
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
    amin: f64,
    amax: f64,
    scale: f64,
) -> (
    f64,  // alp
    f64,  // alp1
    f64,  // alp2
    f64,  // falp
) {
    let (mut alp, mut alp1, mut alp2, mut falp) = (0.0, 0.0, 0.0, 0.0);

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
        let step = p.iter().map(|i| *i * alp).collect::<Vec<f64>>();
        falp = feval(&std::array::from_fn::<f64, N, _>(|i| x[i] + step[i]));
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
            let step = p.iter().map(|i| *i * alp).collect::<Vec<f64>>();
            falp = feval(&std::array::from_fn::<f64, N, _>(|i| x[i] + step[i]));
            alist.push(alp);
            flist.push(falp);
        }
    }

    // alist and flist are set - now compute min and max
    let aamin = alist.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let aamax = alist.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if amin > aamin || amax < aamax {
        eprintln!("GLS Error: non-admissible STEP in alist");
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
            let step = p.iter().map(|i| *i * alp).collect::<Vec<f64>>();
            falp = feval(&std::array::from_fn::<f64, N, _>(|i| x[i] + step[i]));
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

    #[test]
    fn test_1() {
        let x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let p = SVector::<f64, 6>::from_row_slice(&[1., 0., 0., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::new();
        let mut flist: Vec<f64> = Vec::new();
        let amin = -1.0;
        let amax = 1.0;
        let scale = 0.1;

        let (alp, alp1, alp2, falp) = lsinit(&x, &p, &mut alist, &mut flist, amin, amax, scale);

        let alist_expected = Vec::from([0.0, 0.1]);
        let flist_expected = Vec::from([-1.4069105761385299, -1.4664312887853619]);


        assert_eq!(alist, alist_expected);
        assert_eq!(flist, flist_expected);
        assert_eq!((alp, alp1, alp2, falp), (0.1, -0.1, 0.1, -1.4664312887853619));
    }

    #[test]
    fn test_2() {
        let x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let p = SVector::<f64, 6>::from_row_slice(&[0., 1., 0., 0., 0., 0.]);
        let mut alist: Vec<f64> = Vec::from([0.0, 0.1]);
        let mut flist: Vec<f64> = Vec::from([0.0, 0.1]);
        let amin = -1.0;
        let amax = 1.0;
        let scale = 0.1;

        let (alp, alp1, alp2, falp) = lsinit(&x, &p, &mut alist, &mut flist, amin, amax, scale);

        let alist_expected = Vec::from([0.0, 0.1, -0.1]);
        let flist_expected = Vec::from([0.0, 0.1, -1.4091800887102848]);

        assert_eq!(alist, alist_expected);
        assert_eq!(flist, flist_expected);
        assert_eq!((alp, alp1, alp2, falp), (-0.1, -0.1, 0.1, -1.4091800887102848));
    }
}
