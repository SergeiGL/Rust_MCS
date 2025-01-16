pub fn lssplit(alist_i: f64, alist_i_plus_1: f64, flist_i: f64, flist_i_plus_1: f64, short: f64) -> (f64, f64) {
    let fac = match flist_i.partial_cmp(&flist_i_plus_1).unwrap() {
        std::cmp::Ordering::Less => short,
        std::cmp::Ordering::Greater => 1.0 - short,
        std::cmp::Ordering::Equal => 0.5,
    };
    let alp = alist_i + fac * (alist_i_plus_1 - alist_i);
    (alp, fac)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flist_increasing() {
        let result = lssplit(10.0, 20.0, 1.0, 2.0, 0.3);
        assert_eq!(result, (13.0, 0.3));
    }

    #[test]
    fn test_flist_decreasing() {
        let result = lssplit(10.0, 20.0, 2.0, 1.0, 0.3);
        assert_eq!(result, (17.0, 0.7));
    }

    #[test]
    fn test_flist_equal() {
        let result = lssplit(10.0, 20.0, 2.0, 2.0, 0.3);
        assert_eq!(result, (15.0, 0.5));
    }
}