pub fn chrelerr(fbest: f64, stop: &[f64]) -> bool {
    let fglob = stop[1];
    if fbest - fglob <= (stop[0] * fglob.abs()).max(stop[2]) {
        false
    } else {
        true
    }
}


pub fn chvtr(fbest: f64, vtr: f64) -> bool {
    if fbest <= vtr { false } else { true }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        assert!(!chrelerr(1.0, &[-1.0, 2.0, 4.0]));
    }

    #[test]
    fn test_1() {
        assert!(chrelerr(1000.0, &[0.0, -2000.0, -400.0]));
    }

    #[test]
    fn test_2() {
        assert!(chrelerr(0.0, &[0.0, -2.0, -400.0]));
    }
    #[test]
    fn test_3() {
        assert!(!chrelerr(-1.0, &[-13.0, 12.0, 34.0]));
    }

    #[test]
    fn test_4() {
        assert!(!chvtr(1.0, 10.0));
    }
    #[test]
    fn test_5() {
        assert!(!chvtr(-1000.0, 0.0));
    }
    #[test]
    fn test_6() {
        assert!(chvtr(0.0, -2.0));
    }

    #[test]
    fn test_7() {
        assert!(chvtr(-1.0, -13.0));
    }
}