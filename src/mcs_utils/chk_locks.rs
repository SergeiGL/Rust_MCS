use nalgebra::SVector;

pub fn chkloc<const N: usize>(xloc: &[SVector<f64, N>], x: &SVector<f64, N>) -> bool {
    !xloc.iter().any(|x_i| x_i == x)
}

pub fn fbestloc<const N: usize>(
    fmi: &Vec<f64>,
    fbest: &mut f64,
    xmin: &Vec<SVector<f64, N>>,
    xbest: &mut SVector<f64, N>,
    nbasket0: usize,
) {
    if fmi[nbasket0] < *fbest {
        *fbest = fmi[nbasket0];
        *xbest = xmin[nbasket0];
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chkloc_test_0() {
        let xloc = vec![];
        let x = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        let result = chkloc(&xloc, &x);
        assert!(result);
    }

    #[test]
    fn test_chkloc_test_1() {
        let xloc = vec![SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]), SVector::<f64, 3>::from_row_slice(&[4.0, 5.0, 6.0])];
        let x = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        let result = chkloc(&xloc, &x);
        assert!(!result);
    }

    #[test]
    fn test_chkloc_test_2() {
        let xloc = vec![SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]), SVector::<f64, 3>::from_row_slice(&[4.0, 5.0, 6.0])];
        let x = SVector::<f64, 3>::from_row_slice(&[7.0, 8.0, 9.0]);
        let result = chkloc(&xloc, &x);
        assert!(result);
    }


    #[test]
    fn test_fbestloc_test_0() {
        let fmi = vec![1.5, 2.0, 3.0];
        let mut fbest = 2.0;
        let xmin = vec![SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]), SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]), SVector::<f64, 2>::from_row_slice(&[5.0, 6.0])];
        let mut xbest = SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]);
        let nbasket0 = 0_usize;

        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);

        assert_eq!(fbest, fmi[nbasket0]);
        assert_eq!(xbest, xmin[nbasket0]);
    }

    #[test]
    fn test_fbestloc_test_1() {
        let fmi = vec![2.5, 3.0, 3.5];
        let mut fbest = 2.0;
        let xmin = vec![SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]), SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]), SVector::<f64, 2>::from_row_slice(&[5.0, 6.0])];
        let mut xbest = SVector::<f64, 2>::from_row_slice(&[7.0, 8.0]);
        let nbasket0 = 0_usize;

        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);

        assert_eq!(fbest, 2.0);
        assert_eq!(xbest.as_slice(), [7.0, 8.0]);
    }

    #[test]
    fn test_fbestloc_test_2() {
        let fmi = vec![2.0, 3.0, 4.0];
        let mut fbest = 2.0;
        let xmin = vec![SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]), SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]), SVector::<f64, 2>::from_row_slice(&[5.0, 6.0])];
        let mut xbest = SVector::<f64, 2>::from_row_slice(&[7.0, 8.0]);
        let nbasket0 = 0_usize;

        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);

        assert_eq!(fbest, 2.0);
        assert_eq!(xbest.as_slice(), [7.0, 8.0]);
    }

    #[test]
    fn test_fbestloc_test_5() {
        let fmi = vec![f64::NAN, 2.0, 3.0];
        let mut fbest = 2.0;
        let xmin = vec![SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]), SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]), SVector::<f64, 2>::from_row_slice(&[5.0, 6.0])];
        let mut xbest = SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]);
        let nbasket0 = 0_usize;

        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);

        assert_eq!(fbest, 2.0);
        assert_eq!(xbest.as_slice(), [3.0, 4.0]);
    }

    #[test]
    fn test_fbestloc_test_6() {
        let fmi = vec![1.5, 2.0, 3.0];
        let mut fbest = f64::NAN;
        let xmin = vec![SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]), SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]), SVector::<f64, 2>::from_row_slice(&[5.0, 6.0])];
        let mut xbest = SVector::<f64, 2>::from_row_slice(&[3.0, 4.0]);
        let nbasket0 = 0_usize;

        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);

        assert!(fbest.is_nan());
        assert_eq!(xbest.as_slice(), [3.0, 4.0]);
    }
}