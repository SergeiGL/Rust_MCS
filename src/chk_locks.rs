pub fn chkloc(nloc: usize, xloc: &Vec<Vec<f64>>, x: &[f64]) -> bool {
    for k in 0..nloc {
        if x == &xloc[k] {
            return false;
        }
    }
    true
}

pub fn addloc(nloc: &mut usize, xloc: &mut Vec<Vec<f64>>, x: &Vec<f64>) {
    *nloc += 1;
    xloc.push(x.clone());
}


pub fn fbestloc(
    fmi: &[f64],
    fbest: &mut f64,
    xmin: &Vec<Vec<f64>>,
    xbest: &mut Vec<f64>,
    nbasket0: usize,
) {
    if fmi[nbasket0] < *fbest {
        *fbest = fmi[nbasket0];
        *xbest = xmin[nbasket0].clone();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chkloc_test_0() {
        let nloc = 0_usize;
        let xloc: Vec<Vec<f64>> = vec![];
        let x = vec![1.0, 2.0, 3.0];
        let result = chkloc(nloc, &xloc, &x);
        assert!(result);
    }

    #[test]
    fn test_chkloc_test_1() {
        let nloc = 2_usize;
        let xloc = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let x = vec![1.0, 2.0, 3.0];
        let result = chkloc(nloc, &xloc, &x);
        assert!(!result);
    }

    #[test]
    fn test_chkloc_test_2() {
        let nloc = 2_usize;
        let xloc = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let x = vec![7.0, 8.0, 9.0];
        let result = chkloc(nloc, &xloc, &x);
        assert!(result);
    }

    #[test]
    fn test_chkloc_test_4() {
        let nloc = 1_usize;
        let xloc = vec![vec![]];
        let x: Vec<f64> = vec![];
        let result = chkloc(nloc, &xloc, &x);
        assert!(!result);
    }

    #[test]
    fn test_chkloc_test_5() {
        let nloc = 1usize;
        let xloc = vec![vec![1.0, 2.0, 3.0]];
        let x: Vec<f64> = vec![];
        let result = chkloc(nloc, &xloc, &x);
        assert!(result);
    }

    #[test]
    fn test_chkloc_test_6() {
        let nloc = 2usize;
        let xloc = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
        let x = vec![1.0, 2.0];
        let result = chkloc(nloc, &xloc, &x);
        assert!(!result);
    }

    // -------------------------
    #[test]
    fn test_addloc_test_0() {
        let mut nloc = 2;
        let mut xloc = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let x = vec![7.0, 8.0, 9.0];
        addloc(&mut nloc, &mut xloc, &x);
        assert_eq!(nloc, 3);
        assert_eq!(xloc.len(), nloc);
        assert_eq!(xloc[nloc - 1], x);
    }

    #[test]
    fn test_addloc_test_1() {
        let mut nloc = 1;
        let mut xloc = vec![vec![1.0, 2.0, 3.0]];
        let mut x = vec![4.0, 5.0, 6.0];
        addloc(&mut nloc, &mut xloc, &x);
        x[0] = 100.0; // Modify x after calling addloc
        assert_ne!(xloc[nloc - 1][0], x[0]); // Since xloc[nloc - 1] is cloned
    }

    #[test]
    fn test_addloc_test_2() {
        let mut nloc = 0usize;
        let mut xloc: Vec<Vec<f64>> = vec![];
        let x: Vec<f64> = vec![];
        addloc(&mut nloc, &mut xloc, &x);
        assert_eq!(nloc, 1);
        assert_eq!(xloc.len(), 1);
        assert_eq!(xloc[nloc - 1], x);
    }

    // -----------------------------------------------------
    #[test]
    fn test_fbestloc_test_0() {
        let fmi = vec![1.5, 2.0, 3.0];
        let mut fbest = 2.0;
        let xmin = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut xbest = vec![3.0, 4.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, fmi[nbasket0]);
        assert_eq!(xbest, xmin[nbasket0]);
    }

    #[test]
    fn test_fbestloc_test_1() {
        let fmi = vec![2.5, 3.0, 3.5];
        let mut fbest = 2.0;
        let xmin = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut xbest = vec![7.0, 8.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, 2.0);
        assert_eq!(xbest, vec![7.0, 8.0]);
    }

    #[test]
    fn test_fbestloc_test_2() {
        let fmi = vec![2.0, 3.0, 4.0];
        let mut fbest = 2.0;
        let xmin = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut xbest = vec![7.0, 8.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, 2.0);
        assert_eq!(xbest, vec![7.0, 8.0]);
    }

    #[test]
    fn test_fbestloc_test_5() {
        let fmi = vec![std::f64::NAN, 2.0, 3.0];
        let mut fbest = 2.0;
        let xmin = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut xbest = vec![3.0, 4.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, 2.0); // NaN comparison should not change fbest
        assert_eq!(xbest, vec![3.0, 4.0]);
    }

    #[test]
    fn test_fbestloc_test_6() {
        let fmi = vec![1.5, 2.0, 3.0];
        let mut fbest = std::f64::NAN;
        let xmin = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut xbest = vec![3.0, 4.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert!(fbest.is_nan()); // fbest remains NaN
        assert_eq!(xbest, vec![3.0, 4.0]); // xbest remains unchanged
    }
}