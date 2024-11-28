pub fn updtf(
    n: usize,
    i: usize,
    x1: &[f64],
    x2: &[f64],
    f1: &mut [f64],
    f2: &mut [f64],
    fold: f64,
    f: f64,
) {
    for i1 in 0..n {
        if i1 != i {
            if x1[i1] == f64::INFINITY {
                f1[i1] += fold - f;
            }
            if x2[i1] == f64::INFINITY {
                f2[i1] += fold - f;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let n = 0;
        let i = 0;
        let x1: Vec<f64> = vec![];
        let x2: Vec<f64> = vec![];
        let mut f1: Vec<f64> = vec![];
        let mut f2: Vec<f64> = vec![];
        let fold = 0.0;
        let f = 0.0;

        updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f);

        assert_eq!(f1, vec![]);
        assert_eq!(f2, vec![]);
    }

    #[test]
    fn test_1() {
        let n = 5;
        let i = 2;
        let x1 = vec![f64::INFINITY, 1.0, 2.0, f64::INFINITY, 4.0];
        let x2 = vec![0.0, f64::INFINITY, 2.0, 3.0, f64::INFINITY];
        let mut f1 = vec![1.0; n];
        let mut f2 = vec![1.0; n];
        let fold = 1.0;
        let f = 0.5;

        updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = vec![1.5, 1.0, 1.0, 1.5, 1.0];
        let expected_f2 = vec![1.0, 1.5, 1.0, 1.0, 1.5];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_2() {
        let n = 5;
        let i = 2;
        let x1 = vec![0.0, 1.0, f64::INFINITY, 3.0, 4.0];
        let x2 = vec![0.0, 1.0, f64::INFINITY, 3.0, 4.0];
        let mut f1 = vec![1.0; n];
        let mut f2 = vec![1.0; n];
        let fold = 1.0;
        let f = 0.5;

        updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = vec![1.0; n];
        let expected_f2 = vec![1.0; n];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_3() {
        let n = 3;
        let i = 1;
        let x1 = vec![f64::INFINITY, 0.0, f64::INFINITY];
        let x2 = vec![0.0, f64::INFINITY, f64::INFINITY];
        let mut f1 = vec![0.0; n];
        let mut f2 = vec![0.0; n];
        let fold = 1.0;
        let f = 0.5;

        updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = vec![0.5, 0.0, 0.5];
        let expected_f2 = vec![0.0, 0.0, 0.5];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_4() {
        let n = 5;
        let i = 2;
        let x1 = (0..n).map(|x| x as f64).collect::<Vec<_>>();
        let x2 = (0..n).map(|x| x as f64).collect::<Vec<_>>();
        let mut f1 = vec![1.0; n];
        let mut f2 = vec![1.0; n];
        let fold = 1.0;
        let f = 0.5;

        updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f);

        assert_eq!(f1, vec![1.0; n]);
        assert_eq!(f2, vec![1.0; n]);
    }

    #[test]
    fn test_5() {
        let n = 5;
        let i = 2;
        let x1 = vec![f64::INFINITY; n];
        let x2 = vec![f64::INFINITY; n];
        let mut f1 = vec![1.0; n];
        let mut f2 = vec![1.0; n];
        let fold = 1.0;
        let f = 0.5;

        updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f);

        let expected_f1 = vec![1.5, 1.5, 1.0, 1.5, 1.5];
        let expected_f2 = vec![1.5, 1.5, 1.0, 1.5, 1.5];

        assert_eq!(f1, expected_f1);
        assert_eq!(f2, expected_f2);
    }

    #[test]
    fn test_6() {
        let n = 5;
        let i = 2;
        let x1 = vec![f64::INFINITY; n];
        let x2 = vec![f64::INFINITY; n];
        let mut f1 = vec![1.0; n];
        let mut f2 = vec![1.0; n];
        let fold = 1.0;
        let f = 1.0;

        updtf(n, i, &x1, &x2, &mut f1, &mut f2, fold, f);

        assert_eq!(f1, vec![1.0; n]);
        assert_eq!(f2, vec![1.0; n]);
    }
}
