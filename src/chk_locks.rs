pub fn chkloc<const N: usize>(
    nloc: usize,
    xloc: &Vec<[f64; N]>,
    x: &[f64; N],
) ->
    bool
{
    for k in 0..nloc {
        if x == &xloc[k] {
            return false;
        }
    }
    true
}

pub fn fbestloc<const N: usize>(
    fmi: &Vec<f64>,
    fbest: &mut f64,
    xmin: &Vec<[f64; N]>,
    xbest: &mut [f64; N],
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
        let xloc: Vec<[f64; 3]> = vec![];
        let x = [1.0, 2.0, 3.0];
        let result = chkloc(nloc, &xloc, &x);
        assert!(result);
    }

    #[test]
    fn test_chkloc_test_1() {
        let nloc = 2_usize;
        let xloc = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = [1.0, 2.0, 3.0];
        let result = chkloc(nloc, &xloc, &x);
        assert!(!result);
    }

    #[test]
    fn test_chkloc_test_2() {
        let nloc = 2_usize;
        let xloc = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = [7.0, 8.0, 9.0];
        let result = chkloc(nloc, &xloc, &x);
        assert!(result);
    }

    #[test]
    fn test_chkloc_test_4() {
        let nloc = 1_usize;
        let xloc = vec![[]];
        let x = [];
        let result = chkloc(nloc, &xloc, &x);
        assert!(!result);
    }

    #[test]
    fn test_fbestloc_test_0() {
        let fmi = vec![1.5, 2.0, 3.0];
        let mut fbest = 2.0;
        let xmin = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut xbest = [3.0, 4.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, fmi[nbasket0]);
        assert_eq!(xbest, xmin[nbasket0]);
    }

    #[test]
    fn test_fbestloc_test_1() {
        let fmi = vec![2.5, 3.0, 3.5];
        let mut fbest = 2.0;
        let xmin = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut xbest = [7.0, 8.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, 2.0);
        assert_eq!(xbest, [7.0, 8.0]);
    }

    #[test]
    fn test_fbestloc_test_2() {
        let fmi = vec![2.0, 3.0, 4.0];
        let mut fbest = 2.0;
        let xmin = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut xbest = [7.0, 8.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, 2.0);
        assert_eq!(xbest, [7.0, 8.0]);
    }

    #[test]
    fn test_fbestloc_test_5() {
        let fmi = vec![std::f64::NAN, 2.0, 3.0];
        let mut fbest = 2.0;
        let xmin = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut xbest = [3.0, 4.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert_eq!(fbest, 2.0);
        assert_eq!(xbest, [3.0, 4.0]);
    }

    #[test]
    fn test_fbestloc_test_6() {
        let fmi = vec![1.5, 2.0, 3.0];
        let mut fbest = std::f64::NAN;
        let xmin = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut xbest = [3.0, 4.0];
        let nbasket0 = 0usize;
        fbestloc(&fmi, &mut fbest, &xmin, &mut xbest, nbasket0);
        assert!(fbest.is_nan());
        assert_eq!(xbest, [3.0, 4.0]);
    }
}