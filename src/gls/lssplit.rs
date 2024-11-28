pub fn lssplit(i: usize, alist: &[f64], flist: &[f64], short: f64) -> Result<(f64, f64), String> {
    if (i + 1 >= flist.len()) || (i + 1 >= alist.len()) {
        return Err("Index out of bounds".to_string());
    }

    let fac = if flist[i] < flist[i + 1] {
        short
    } else if flist[i] > flist[i + 1] {
        1.0 - short
    } else {
        0.5
    };

    let alp = alist[i] + fac * (alist[i + 1] - alist[i]);
    Ok((alp, fac))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flist_increasing() {
        let alist = vec![10.0, 20.0];
        let flist = vec![1.0, 2.0];
        let short = 0.3;
        let result = lssplit(0, &alist, &flist, short);
        assert_eq!(result, Ok((13.0, 0.3)));
    }

    #[test]
    fn test_flist_decreasing() {
        let alist = vec![10.0, 20.0];
        let flist = vec![2.0, 1.0];
        let short = 0.3;
        let result = lssplit(0, &alist, &flist, short);
        assert_eq!(result, Ok((17.0, 0.7)));
    }

    #[test]
    fn test_flist_equal() {
        let alist = vec![10.0, 20.0];
        let flist = vec![2.0, 2.0];
        let short = 0.3;
        let result = lssplit(0, &alist, &flist, short);
        assert_eq!(result, Ok((15.0, 0.5)));
    }

    #[test]
    fn test_edge_case_empty_lists() {
        let alist: Vec<f64> = vec![];
        let flist: Vec<f64> = vec![];
        let short = 0.3;
        let result = lssplit(0, &alist, &flist, short);
        assert!(result.is_err());
        assert_eq!(result, Err("Index out of bounds".to_string()));
    }

    #[test]
    fn test_edge_case_single_element() {
        let alist: Vec<f64> = vec![10.0];
        let flist: Vec<f64> = vec![2.0];
        let short = 0.3;
        let result = lssplit(0, &alist, &flist, short);
        assert!(result.is_err());
        assert_eq!(result, Err("Index out of bounds".to_string()));
    }
}