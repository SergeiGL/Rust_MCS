pub fn check_box_bound(u: Vec<f64>, v: Vec<f64>) -> bool {
    if u.len() != v.len() || u >= v { // shapes differ OR upper bound < lower bound
        true
    } else {
        false
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        assert!(check_box_bound(vec![1.0], vec![0.0]));
    }

    #[test]
    fn test_1() {
        assert!(check_box_bound(vec![1.0, 1.0], vec![0.0, 0.0]));
    }

    #[test]
    fn test_2() {
        assert!(check_box_bound(vec![2.0], vec![10.0, 10.0]));
    }

    #[test]
    fn test_3() {
        assert!(!check_box_bound(vec![2.0, -1.0], vec![10.0, 11.0]));
    }

    #[test]
    fn test_4() {
        assert!(!check_box_bound(vec![1.0, 0.0], vec![1.1, 0.0]));
    }
}