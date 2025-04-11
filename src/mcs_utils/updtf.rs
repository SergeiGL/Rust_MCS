use nalgebra::SVector;

#[inline]
pub fn updtf<const N: usize>(
    i: usize, // -1 from Matlab
    x1: &SVector<f64, N>,
    x2: &SVector<f64, N>,
    f1: &mut SVector<f64, N>,
    f2: &mut SVector<f64, N>,
    fold: f64,
    f: f64,
) -> f64 {
    let fold_minus_f = fold - f;

    for i1 in 0..N {
        if i1 != i {
            if x1[i1] == f64::INFINITY {
                f1[i1] += fold_minus_f;
            }
            if x2[i1] == f64::INFINITY {
                f2[i1] += fold_minus_f;
            }
        }
    }
    f
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_1() {
        let i = 2;
        let x1 = SVector::<f64, 5>::from_row_slice(&[f64::INFINITY, 1.0, 2.0, f64::INFINITY, 4.0]);
        let x2 = SVector::<f64, 5>::from_row_slice(&[0.0, f64::INFINITY, 2.0, 3.0, f64::INFINITY]);
        let mut f1 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let mut f2 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.5, 1.0, 1.0, 1.5, 1.0];
        let expected_f2 = [1.0, 1.5, 1.0, 1.0, 1.5];

        assert_eq!(f1.as_slice(), expected_f1);
        assert_eq!(f2.as_slice(), expected_f2);
    }

    #[test]
    fn test_2() {
        let i = 2;
        let x1 = SVector::<f64, 5>::from_row_slice(&[0.0, 1.0, f64::INFINITY, 3.0, 4.0]);
        let x2 = SVector::<f64, 5>::from_row_slice(&[0.0, 1.0, f64::INFINITY, 3.0, 4.0]);
        let mut f1 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let mut f2 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.0; 5];
        let expected_f2 = [1.0; 5];

        assert_eq!(f1.as_slice(), expected_f1);
        assert_eq!(f2.as_slice(), expected_f2);
    }

    #[test]
    fn test_3() {
        let i = 1;
        let x1 = SVector::<f64, 3>::from_row_slice(&[f64::INFINITY, 0.0, f64::INFINITY]);
        let x2 = SVector::<f64, 3>::from_row_slice(&[0.0, f64::INFINITY, f64::INFINITY]);
        let mut f1 = SVector::<f64, 3>::from_row_slice(&[0.0; 3]);
        let mut f2 = SVector::<f64, 3>::from_row_slice(&[0.0; 3]);
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [0.5, 0.0, 0.5];
        let expected_f2 = [0.0, 0.0, 0.5];

        assert_eq!(f1.as_slice(), expected_f1);
        assert_eq!(f2.as_slice(), expected_f2);
    }

    #[test]
    fn test_4() {
        let i = 2;
        let x1 = SVector::<f64, 5>::from_row_slice(&[0., 1., 2., 3., 4.]);
        let x2 = SVector::<f64, 5>::from_row_slice(&[0., 1., 2., 3., 4.]);
        let mut f1 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let mut f2 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.0; 5];
        let expected_f2 = [1.0; 5];

        assert_eq!(f1.as_slice(), expected_f1);
        assert_eq!(f2.as_slice(), expected_f2);
    }

    #[test]
    fn test_5() {
        let i = 2;
        let x1 = SVector::<f64, 5>::from_row_slice(&[f64::INFINITY; 5]);
        let x2 = SVector::<f64, 5>::from_row_slice(&[f64::INFINITY; 5]);
        let mut f1 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let mut f2 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.5, 1.5, 1.0, 1.5, 1.5];
        let expected_f2 = [1.5, 1.5, 1.0, 1.5, 1.5];

        assert_eq!(f1.as_slice(), expected_f1);
        assert_eq!(f2.as_slice(), expected_f2);
    }

    #[test]
    fn test_6() {
        let i = 2;
        let x1 = SVector::<f64, 5>::from_row_slice(&[f64::INFINITY; 5]);
        let x2 = SVector::<f64, 5>::from_row_slice(&[f64::INFINITY; 5]);
        let mut f1 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let mut f2 = SVector::<f64, 5>::from_row_slice(&[1.0; 5]);
        let fold = 1.0;
        let f = 1.0;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.0; 5];
        let expected_f2 = [1.0; 5];

        assert_eq!(f1.as_slice(), expected_f1);
        assert_eq!(f2.as_slice(), expected_f2);
    }
}
