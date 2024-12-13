pub fn updtf<const N: usize>(
    i: usize,
    x1: &[f64; N],
    x2: &[f64; N],
    f1: &mut [f64; N],
    f2: &mut [f64; N],
    fold: f64,
    f: f64,
) -> f64 {
    for i1 in 0..N {
        if i1 != i {
            if x1[i1] == f64::INFINITY {
                f1[i1] += fold - f;
            }
            if x2[i1] == f64::INFINITY {
                f2[i1] += fold - f;
            }
        }
    }
    f
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let i = 0;
        let x1 = [];
        let x2 = [];
        let mut f1 = [];
        let mut f2 = [];
        let fold = 0.0;
        let f = 0.0;

        let f_exp = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        assert_eq!(f1, []);
        assert_eq!(f2, []);
        assert_eq!(f, f_exp);
    }

    #[test]
    fn test_1() {
        let i = 2;
        let x1 = [f64::INFINITY, 1.0, 2.0, f64::INFINITY, 4.0];
        let x2 = [0.0, f64::INFINITY, 2.0, 3.0, f64::INFINITY];
        let mut f1 = [1.0; 5];
        let mut f2 = [1.0; 5];
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.5, 1.0, 1.0, 1.5, 1.0];
        let expected_f2 = [1.0, 1.5, 1.0, 1.0, 1.5];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_2() {
        let i = 2;
        let x1 = [0.0, 1.0, f64::INFINITY, 3.0, 4.0];
        let x2 = [0.0, 1.0, f64::INFINITY, 3.0, 4.0];
        let mut f1 = [1.0; 5];
        let mut f2 = [1.0; 5];
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.0; 5];
        let expected_f2 = [1.0; 5];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_3() {
        let i = 1;
        let x1 = [f64::INFINITY, 0.0, f64::INFINITY];
        let x2 = [0.0, f64::INFINITY, f64::INFINITY];
        let mut f1 = [0.0; 3];
        let mut f2 = [0.0; 3];
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [0.5, 0.0, 0.5];
        let expected_f2 = [0.0, 0.0, 0.5];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_4() {
        let i = 2;
        let x1 = [0., 1., 2., 3., 4.];
        let x2 = [0., 1., 2., 3., 4.];
        let mut f1 = [1.0; 5];
        let mut f2 = [1.0; 5];
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        assert_eq!(f1, [1.0; 5]);
        assert_eq!(f2, [1.0; 5]);
    }

    #[test]
    fn test_5() {
        let i = 2;
        let x1 = [f64::INFINITY; 5];
        let x2 = [f64::INFINITY; 5];
        let mut f1 = [1.0; 5];
        let mut f2 = [1.0; 5];
        let fold = 1.0;
        let f = 0.5;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = [1.5, 1.5, 1.0, 1.5, 1.5];
        let expected_f2 = [1.5, 1.5, 1.0, 1.5, 1.5];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_6() {
        let i = 2;
        let x1 = [f64::INFINITY; 5];
        let x2 = [f64::INFINITY; 5];
        let mut f1 = [1.0; 5];
        let mut f2 = [1.0; 5];
        let fold = 1.0;
        let f = 1.0;

        let _ = updtf(i, &x1, &x2, &mut f1, &mut f2, fold, f);

        assert_eq!(f1, [1.0; 5]);
        assert_eq!(f2, [1.0; 5]);
    }
}
